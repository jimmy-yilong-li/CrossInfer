#include "edgeflow_llm/edgeflow_llm.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "llama-kv-cache.h"
#if __has_include("llama-memory-hybrid.h")
#include "llama-memory-hybrid.h"
#define EDGEFLOW_HAS_HYBRID_MEMORY 1
#endif
#include "llama-model.h"
#include "llama.h"
#include "session_logits.h"
#include "qwen25_prefill.h"

#include <CommonCrypto/CommonDigest.h>
#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef EDGEFLOW_LLM_RUNTIME_SOURCE_FINGERPRINT_SHA256
#error                                                                         \
    "EDGEFLOW_LLM_RUNTIME_SOURCE_FINGERPRINT_SHA256 must identify selected source inputs"
#endif

extern "C" uint32_t edgeflow_llm_abi_version(void) {
  return EDGEFLOW_LLM_ABI_VERSION;
}

extern "C" uint32_t edgeflow_llm_init_params_sizeof(void) {
  return static_cast<uint32_t>(sizeof(edgeflow_llm_init_params));
}

extern "C" uint32_t edgeflow_llm_request_sizeof(void) {
  return static_cast<uint32_t>(sizeof(edgeflow_llm_request));
}

extern "C" uint32_t edgeflow_llm_request_v2_sizeof(void) {
  return static_cast<uint32_t>(sizeof(edgeflow_llm_request_v2));
}

extern "C" uint32_t edgeflow_llm_result_sizeof(void) {
  return static_cast<uint32_t>(sizeof(edgeflow_llm_result));
}

extern "C" uint32_t edgeflow_llm_session_abi_version(void) {
  return EDGEFLOW_LLM_SESSION_ABI_VERSION;
}

extern "C" uint32_t edgeflow_llm_session_params_sizeof(void) {
  return static_cast<uint32_t>(sizeof(edgeflow_llm_session_params));
}

extern "C" uint32_t edgeflow_llm_session_signature_sizeof(void) {
  return static_cast<uint32_t>(sizeof(edgeflow_llm_session_signature));
}

extern "C" uint32_t edgeflow_llm_sequence_info_sizeof(void) {
  return static_cast<uint32_t>(sizeof(edgeflow_llm_sequence_info));
}

extern "C" uint32_t edgeflow_llm_decode_step_result_sizeof(void) {
  return static_cast<uint32_t>(sizeof(edgeflow_llm_decode_step_result));
}

namespace {

using Clock = std::chrono::steady_clock;

constexpr char kImageTag[] = "[img-0]";
constexpr char kVisionStartText[] = "<|vision_start|>";
constexpr char kVisionEndText[] = "<|vision_end|>";
constexpr int kQwenSpatialMergeSize = 2;

struct ModelDeleter {
  void operator()(llama_model * model) const noexcept {
    if (model != nullptr) {
      llama_model_free(model);
    }
  }
};

struct ContextDeleter {
  void operator()(llama_context * ctx) const noexcept {
    if (ctx != nullptr) {
      llama_free(ctx);
    }
  }
};

class LlamaError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class LlamaDecodeError : public LlamaError {
public:
  explicit LlamaDecodeError(int32_t status)
      : LlamaError("llama_decode failed with status " + std::to_string(status)),
        status(status) {}
  // The vendored llama_decode contract restores KV for these two rejections.
  // Abort (2) and fatal errors may retain processed ubatches.
  bool state_unchanged() const { return status == 1 || status == -1; }
  const int32_t status;
};

void ensure_backend_initialized() {
  static std::once_flag once;
  std::call_once(once, []() { llama_backend_init(); });
}

float elapsed_ms(const Clock::time_point start, const Clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(end - start).count();
}

bool trace_decode_enabled() {
  const char * raw = std::getenv("EDGEFLOW_LLM_TRACE_DECODE");
  return raw != nullptr && raw[0] != '\0' && std::strcmp(raw, "0") != 0;
}

bool memory_clear_enabled() {
  const char * raw = std::getenv("EDGEFLOW_LLM_MEMORY_CLEAR");
  if (raw == nullptr || raw[0] == '\0') {
    return true;
  }
  return std::strcmp(raw, "0") != 0;
}

void trace_decode_call(
    const bool trace_enabled,
    llama_context * ctx,
    const char * phase,
    const int32_t n_tokens,
    const llama_pos pos_0,
    const Clock::time_point start) {
  if (!trace_enabled) {
    return;
  }
  // TRACE_DECODE is intrusive: fencing each batch changes overlap and timing.
  llama_synchronize(ctx);
  const float elapsed = elapsed_ms(start, Clock::now());
  std::fprintf(
      stderr,
      "edgeflow_llm trace phase=%s n_tokens=%d pos0=%d elapsed_ms=%.3f\n",
      phase,
      n_tokens,
      static_cast<int>(pos_0),
      elapsed);
  std::fflush(stderr);
}

void trace_runtime_config(const bool trace_enabled, const llama_context * ctx, const bool memory_clear) {
  if (!trace_enabled) {
    return;
  }
  std::fprintf(
      stderr,
      "edgeflow_llm trace config n_ctx=%u memory_clear=%s\n",
      static_cast<unsigned>(llama_n_ctx(ctx)),
      memory_clear ? "true" : "false");
  std::fflush(stderr);
}

std::vector<llama_token> concat_tokens(
    const std::vector<llama_token> & first,
    const std::vector<llama_token> & second) {
  std::vector<llama_token> out;
  out.reserve(first.size() + second.size());
  out.insert(out.end(), first.begin(), first.end());
  out.insert(out.end(), second.begin(), second.end());
  return out;
}

std::vector<llama_token> tokenize_prompt(const llama_model * model, const std::string & text, const bool add_special, const bool parse_special) {
  const llama_vocab * vocab = llama_model_get_vocab(model);
  if (vocab == nullptr) {
    throw LlamaError("model vocabulary is unavailable");
  }
  if (text.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    throw LlamaError("prompt is too large to tokenize");
  }
  const int32_t n_text = static_cast<int32_t>(text.size());
  const int32_t required = llama_tokenize(vocab, text.c_str(), n_text, nullptr, 0, add_special, parse_special);
  if (required == std::numeric_limits<int32_t>::min()) {
    throw LlamaError("prompt tokenization overflowed int32");
  }
  if (required == 0) {
    return {};
  }
  const int32_t capacity = required < 0 ? -required : required;
  std::vector<llama_token> tokens(static_cast<size_t>(capacity));
  const int32_t written = llama_tokenize(vocab, text.c_str(), n_text, tokens.data(), capacity, add_special, parse_special);
  if (written < 0 || written > capacity) {
    throw LlamaError("prompt tokenization failed");
  }
  tokens.resize(static_cast<size_t>(written));
  return tokens;
}

std::string render_user_prompt(const llama_model * model, const std::string & prompt) {
  const char * tmpl = llama_model_chat_template(model, nullptr);
  const llama_chat_message message = {
      .role = "user",
      .content = prompt.c_str(),
  };

  const int32_t required = llama_chat_apply_template(tmpl, &message, 1, true, nullptr, 0);
  if (required <= 0) {
    throw LlamaError("chat template rendering failed");
  }

  std::string rendered(static_cast<size_t>(required) + 1, '\0');
  const int32_t written = llama_chat_apply_template(
      tmpl,
      &message,
      1,
      true,
      rendered.data(),
      static_cast<int32_t>(rendered.size()));
  if (written <= 0 || written > required) {
    throw LlamaError("chat template rendering returned an invalid length");
  }
  rendered.resize(static_cast<size_t>(written));
  return rendered;
}

// FastVLM-specific prompt renderer. Matches Apple LLaVA `qwen_2` conv_template
// exactly: prepends a "You are a helpful assistant." system message and
// embeds the user content with a [img-0] marker immediately followed by a '\n'
// (Apple uses DEFAULT_IMAGE_TOKEN + '\n' + prompt). The split point is the
// [img-0] marker itself, so the '\n' that follows naturally lands in the
// suffix half (matching Apple's `tokenizer_image_token` behavior where the
// -200 sentinel replaces `<image>` but keeps the trailing '\n' as part of
// the text after).
//
// Prompt layout:
//   prefix tokens after this helper + subsequent [img-0] split =
//     <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n
//   suffix tokens =
//     \n{user_prompt}<|im_end|>\n<|im_start|>assistant\n
std::string render_user_prompt_with_system(
    const llama_model * model,
    const std::string & user_content,
    const char * system_content) {
  const char * tmpl = llama_model_chat_template(model, nullptr);
  const std::array<llama_chat_message, 2> messages = {{
      {/*role=*/"system", /*content=*/system_content},
      {/*role=*/"user",   /*content=*/user_content.c_str()},
  }};

  const int32_t required = llama_chat_apply_template(
      tmpl, messages.data(), messages.size(), /*add_ass=*/true, nullptr, 0);
  if (required <= 0) {
    throw LlamaError("chat template rendering failed (with system)");
  }

  std::string rendered(static_cast<size_t>(required) + 1, '\0');
  const int32_t written = llama_chat_apply_template(
      tmpl,
      messages.data(),
      messages.size(),
      /*add_ass=*/true,
      rendered.data(),
      static_cast<int32_t>(rendered.size()));
  if (written <= 0 || written > required) {
    throw LlamaError("chat template rendering returned an invalid length (with system)");
  }
  rendered.resize(static_cast<size_t>(written));
  return rendered;
}

std::pair<std::string, std::string> split_single_image_prompt(const std::string & prompt) {
  const std::size_t image_pos = prompt.find(kImageTag);
  if (image_pos == std::string::npos) {
    throw LlamaError("multimodal prompt is missing [img-0]");
  }
  if (prompt.find(kImageTag, image_pos + std::strlen(kImageTag)) != std::string::npos) {
    throw LlamaError("multimodal prompt must contain exactly one [img-0]");
  }
  return {
      prompt.substr(0, image_pos),
      prompt.substr(image_pos + std::strlen(kImageTag)),
  };
}

std::string token_to_piece(const llama_model * model, const llama_token token) {
  const llama_vocab * vocab = llama_model_get_vocab(model);
  if (vocab == nullptr) {
    throw LlamaError("model vocabulary is unavailable");
  }
  int32_t required = llama_token_to_piece(vocab, token, nullptr, 0, 0, true);
  if (required == std::numeric_limits<int32_t>::min()) {
    throw LlamaError("token piece length overflowed int32");
  }
  if (required == 0) {
    return {};
  }
  const int32_t capacity = required < 0 ? -required : required;
  std::string piece(static_cast<size_t>(capacity), '\0');
  required = llama_token_to_piece(vocab, token, piece.data(), capacity, 0, true);
  if (required < 0 || required > capacity) {
    throw LlamaError("token piece decoding failed");
  }
  piece.resize(static_cast<size_t>(required));
  return piece;
}

llama_token greedy_sample(llama_context * ctx, const llama_model * model, bool ignore_eog = false) {
  const llama_vocab * vocab = llama_model_get_vocab(model);
  if (vocab == nullptr) {
    throw LlamaError("model vocabulary is unavailable");
  }
  const int32_t n_vocab = llama_vocab_n_tokens(vocab);
  float * logits = llama_get_logits_ith(ctx, -1);
  if (logits == nullptr || n_vocab <= 0) {
    throw LlamaError("logits are unavailable");
  }

  int32_t best = -1;
  float best_value = -std::numeric_limits<float>::infinity();
  for (int32_t i = 0; i < n_vocab; ++i) {
    if (logits[i] > best_value && (!ignore_eog || !llama_vocab_is_eog(vocab, i))) {
      best_value = logits[i];
      best = i;
    }
  }
  if (best < 0) {
    throw LlamaError("no eligible token has a finite logit");
  }
  return static_cast<llama_token>(best);
}

void require_decode_batch_capacity(const llama_context *ctx,
                                   const size_t token_count) {
  const size_t capacity = static_cast<size_t>(llama_n_batch(ctx));
  if (token_count > capacity) {
    throw LlamaError("decode batch contains " + std::to_string(token_count) +
                     " tokens but the context n_batch capacity is " +
                     std::to_string(capacity));
  }
}

void decode_token_batch_at_sequence(llama_context *ctx,
                                    const std::vector<llama_token> &tokens,
                                    const llama_pos pos_0,
                                    const llama_seq_id sequence_id,
                                    const bool logits_last) {
  if (tokens.empty()) {
    throw LlamaError("cannot decode an empty token batch");
  }
  require_decode_batch_capacity(ctx, tokens.size());

  std::vector<llama_pos> positions(tokens.size());
  std::vector<int32_t> n_seq_id(tokens.size(), 1);
  std::array<llama_seq_id, 1> seq_id_storage = {sequence_id};
  std::vector<llama_seq_id *> seq_ids(tokens.size() + 1, nullptr);
  std::vector<int8_t> logits(tokens.size(), 0);
  for (size_t i = 0; i < tokens.size(); ++i) {
    positions[i] = pos_0 + static_cast<llama_pos>(i);
    seq_ids[i] = seq_id_storage.data();
  }
  logits.back() = logits_last ? 1 : 0;

  const llama_batch batch = {
      /*n_tokens =*/ static_cast<int32_t>(tokens.size()),
      /*tokens   =*/ const_cast<llama_token *>(tokens.data()),
      /*embd     =*/ nullptr,
      /*pos      =*/ positions.data(),
      /*n_seq_id =*/ n_seq_id.data(),
      /*seq_id   =*/ seq_ids.data(),
      /*logits   =*/ logits.data(),
  };
  const int32_t rc = llama_decode(ctx, batch);
  if (rc != 0) {
    throw LlamaDecodeError(rc);
  }
}

void decode_token_batch_at(llama_context *ctx,
                           const std::vector<llama_token> &tokens,
                           const llama_pos pos_0, const bool logits_last) {
  decode_token_batch_at_sequence(ctx, tokens, pos_0, 0, logits_last);
}

void decode_embedding_batch_2d_sequence(llama_context *ctx, const float *embd,
                                        const int32_t n_tokens,
                                        const int32_t n_embd,
                                        const llama_pos pos_0, const int nx,
                                        const int ny,
                                        const llama_seq_id sequence_id,
                                        uint64_t *borrowed_pointer = nullptr) {
  if (embd == nullptr || n_tokens <= 0 || n_embd <= 0) {
    throw LlamaError("embedding batch inputs must be non-null and positive");
  }
  if (nx <= 0 || ny <= 0 || nx * ny != n_tokens) {
    throw LlamaError("embedding batch grid shape does not match token count");
  }
  require_decode_batch_capacity(ctx, static_cast<size_t>(n_tokens));

  std::vector<llama_pos> positions(static_cast<size_t>(n_tokens) * 4, 0);
  std::vector<int32_t> n_seq_id(static_cast<size_t>(n_tokens), 1);
  std::array<llama_seq_id, 1> seq_id_storage = {sequence_id};
  std::vector<llama_seq_id *> seq_ids(static_cast<size_t>(n_tokens) + 1, nullptr);
  std::vector<int8_t> logits(static_cast<size_t>(n_tokens), 0);

  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      const int i = y * nx + x;
      positions[static_cast<size_t>(i)] = pos_0;
      positions[static_cast<size_t>(i) + static_cast<size_t>(n_tokens)] = pos_0 + y;
      positions[static_cast<size_t>(i) + static_cast<size_t>(n_tokens) * 2] = pos_0 + x;
      positions[static_cast<size_t>(i) + static_cast<size_t>(n_tokens) * 3] = 0;
      seq_ids[static_cast<size_t>(i)] = seq_id_storage.data();
    }
  }

  const llama_batch batch = {
      /*n_tokens =*/ n_tokens,
      /*tokens   =*/ nullptr,
      /*embd     =*/ const_cast<float *>(embd),
      /*pos      =*/ positions.data(),
      /*n_seq_id =*/ n_seq_id.data(),
      /*seq_id   =*/ seq_ids.data(),
      /*logits   =*/ logits.data(),
  };
  if (borrowed_pointer != nullptr) {
    *borrowed_pointer = reinterpret_cast<uintptr_t>(batch.embd);
  }
  const int32_t rc = llama_decode(ctx, batch);
  if (rc != 0) {
    throw LlamaError("llama_decode failed for embedding batch with status " + std::to_string(rc));
  }
}

// 1D positional embedding batch decode. Each of the `n_tokens` embedding rows
// gets exactly one position value, sequential from `pos_0`. Used by the
// FastVLM/LLaVA-style multimodal path where the text backbone applies
// standard 1D RoPE and the visual tokens occupy sequential positions in the
// text sequence (no spatial-grid / mRoPE layout).
void decode_embedding_batch_1d(
    llama_context * ctx,
    const float * embd,
    const int32_t n_tokens,
    const int32_t n_embd,
    const llama_pos pos_0) {
  if (embd == nullptr || n_tokens <= 0 || n_embd <= 0) {
    throw LlamaError("1d embedding batch inputs must be non-null and positive");
  }
  require_decode_batch_capacity(ctx, static_cast<size_t>(n_tokens));

  std::vector<llama_pos> positions(static_cast<size_t>(n_tokens));
  std::vector<int32_t> n_seq_id(static_cast<size_t>(n_tokens), 1);
  std::array<llama_seq_id, 1> seq_id_storage = {0};
  std::vector<llama_seq_id *> seq_ids(static_cast<size_t>(n_tokens) + 1, nullptr);
  std::vector<int8_t> logits(static_cast<size_t>(n_tokens), 0);

  for (int32_t i = 0; i < n_tokens; ++i) {
    positions[static_cast<size_t>(i)] = pos_0 + i;
    seq_ids[static_cast<size_t>(i)] = seq_id_storage.data();
  }

  const llama_batch batch = {
      /*n_tokens =*/ n_tokens,
      /*tokens   =*/ nullptr,
      /*embd     =*/ const_cast<float *>(embd),
      /*pos      =*/ positions.data(),
      /*n_seq_id =*/ n_seq_id.data(),
      /*seq_id   =*/ seq_ids.data(),
      /*logits   =*/ logits.data(),
  };
  const int32_t rc = llama_decode(ctx, batch);
  if (rc != 0) {
    throw LlamaError("llama_decode failed for 1d embedding batch with status " + std::to_string(rc));
  }
}

void validate_visual_request(const edgeflow_llm_request * req, const llama_model * model) {
  if (req->visual_embd == nullptr) {
    throw LlamaError("visual_embd must be non-null for multimodal generation");
  }
  if (req->num_visual_tokens <= 0 || req->hidden_size <= 0) {
    throw LlamaError("multimodal generation requires positive num_visual_tokens and hidden_size");
  }
  if (req->hidden_size != llama_model_n_embd_inp(model)) {
    throw LlamaError("visual hidden_size does not match model input embedding width");
  }

  const int t = req->image_grid_thw[0];
  const int h = req->image_grid_thw[1];
  const int w = req->image_grid_thw[2];
  if (t != 1) {
    throw LlamaError("This native entry point supports only single-image inputs");
  }
  if (h <= 0 || w <= 0 || h % kQwenSpatialMergeSize != 0 || w % kQwenSpatialMergeSize != 0) {
    throw LlamaError("image_grid_thw is incompatible with Qwen2.5-VL spatial merge");
  }

  const int merged_h = h / kQwenSpatialMergeSize;
  const int merged_w = w / kQwenSpatialMergeSize;
  if (static_cast<int64_t>(merged_h) * merged_w != req->num_visual_tokens) {
    throw LlamaError("num_visual_tokens does not match image_grid_thw after spatial merge");
  }
}

void validate_visual_request_fastvlm(const edgeflow_llm_request_v2 * req, const llama_model * model) {
  if (req->visual_embd == nullptr) {
    throw LlamaError("visual_embd must be non-null for multimodal generation");
  }
  if (req->num_visual_tokens <= 0 || req->hidden_size <= 0) {
    throw LlamaError("FastVLM request requires positive num_visual_tokens and hidden_size");
  }
  if (req->hidden_size != llama_model_n_embd_inp(model)) {
    throw LlamaError("FastVLM visual hidden_size does not match text model input embedding width");
  }
  if (req->visual_dtype != EDGEFLOW_DTYPE_FP16 && req->visual_dtype != EDGEFLOW_DTYPE_FP32) {
    throw LlamaError("FastVLM visual_dtype must be EDGEFLOW_DTYPE_FP16 or EDGEFLOW_DTYPE_FP32");
  }
  // image_grid_thw is intentionally NOT validated for FastVLM. It is decorative.
  // num_visual_tokens is taken at face value; no spatial merge math is performed.
}

void validate_visual_request_llava15(const edgeflow_llm_request_v2 * req, const llama_model * model) {
  validate_visual_request_fastvlm(req, model);
  if (req->num_visual_tokens != 576) {
    throw LlamaError("LLaVA-1.5 request requires num_visual_tokens == 576");
  }
  if (req->hidden_size != 4096) {
    throw LlamaError("LLaVA-1.5 request requires hidden_size == 4096");
  }
  if (req->image_grid_thw[0] != 1 || req->image_grid_thw[1] != 24 || req->image_grid_thw[2] != 24) {
    throw LlamaError("LLaVA-1.5 image_grid_thw must be (1, 24, 24)");
  }
}

// Ingest the reusable part of a Qwen image turn. The branch-local user query,
// closing turn markers, and assistant header are deliberately excluded.
void run_qwen25vl_visual_prefix_ingest(llama_context *llctx,
                                       const llama_model *model,
                                       const edgeflow_llm_request *req,
                                       const std::string &prefix_prompt,
                                       llama_pos &next_position,
                                       const bool trace_enabled,
                                       const llama_seq_id sequence_id = 0) {
  validate_visual_request(req, model);

  const std::vector<llama_token> prefix_tokens =
      tokenize_prompt(model, prefix_prompt, true, true);
  const std::vector<llama_token> vision_start_tokens =
      tokenize_prompt(model, kVisionStartText, false, true);
  const std::vector<llama_token> prefix_and_vision_start_tokens =
      concat_tokens(prefix_tokens, vision_start_tokens);

  if (!prefix_and_vision_start_tokens.empty()) {
    const auto start = Clock::now();
    decode_token_batch_at_sequence(llctx, prefix_and_vision_start_tokens,
                                   next_position, sequence_id, false);
    trace_decode_call(
        trace_enabled,
        llctx,
        "prefix_and_vision_start",
        static_cast<int32_t>(prefix_and_vision_start_tokens.size()),
        next_position,
        start);
    next_position += static_cast<llama_pos>(prefix_and_vision_start_tokens.size());
  }

  const int merged_h = req->image_grid_thw[1] / kQwenSpatialMergeSize;
  const int merged_w = req->image_grid_thw[2] / kQwenSpatialMergeSize;
  {
    const auto start = Clock::now();
    decode_embedding_batch_2d_sequence(
        llctx, req->visual_embd, req->num_visual_tokens, req->hidden_size,
        next_position, merged_w, merged_h, sequence_id);
    trace_decode_call(trace_enabled, llctx, "visual_embd", req->num_visual_tokens, next_position, start);
  }
  next_position += static_cast<llama_pos>(std::max(merged_h, merged_w));
}

std::vector<llama_token>
tokenize_qwen25vl_query_suffix(const llama_model *model,
                               const std::string &suffix_prompt) {
  const std::vector<llama_token> vision_end_tokens =
      tokenize_prompt(model, kVisionEndText, false, true);
  const std::vector<llama_token> suffix_tokens =
      tokenize_prompt(model, suffix_prompt, false, true);
  return concat_tokens(vision_end_tokens, suffix_tokens);
}

std::pair<std::vector<llama_token>, std::vector<llama_token>>
qwen25_prompt_tokens(const llama_model *model, const char *prompt) {
  const auto [before, after] = split_single_image_prompt(
      render_user_prompt(model, std::string(kImageTag) + prompt));
  return {concat_tokens(tokenize_prompt(model, before, true, true),
                        tokenize_prompt(model, kVisionStartText, false, true)),
          tokenize_qwen25vl_query_suffix(model, after)};
}

void run_qwen25_packed_ingest(llama_context *context, const llama_model *model,
                             const edgeflow_llm_request *req, const float *packed,
                             int prefix, int suffix, llama_pos &next_position,
                             bool trace_enabled) {
  validate_visual_request(req, model);
  const auto tokens = qwen25_prompt_tokens(model, req->prompt);
  if (model->arch != LLM_ARCH_QWEN2VL || !packed ||
      prefix != int(tokens.first.size()) || suffix != int(tokens.second.size())) {
    throw LlamaError("packed prompt does not match Qwen2.5 text boundaries");
  }
  auto layout = qwen25_packed_layout(
      prefix, req->image_grid_thw[1] / kQwenSpatialMergeSize,
      req->image_grid_thw[2] / kQwenSpatialMergeSize, suffix);
  require_decode_batch_capacity(context, layout.tokens);
  if (layout.tokens > int(llama_n_ctx(context))) {
    throw LlamaError("packed prompt exceeds context capacity");
  }
  std::vector<int32_t> counts(layout.tokens, 1);
  llama_seq_id sequence = 0;
  std::vector<llama_seq_id *> sequences(layout.tokens, &sequence);
  std::vector<int8_t> logits(layout.tokens, 0);
  logits.back() = 1;
  const llama_batch batch = {layout.tokens, nullptr, const_cast<float *>(packed),
                             layout.positions.data(), counts.data(), sequences.data(), logits.data()};
  const auto start = Clock::now();
  const int rc = llama_decode(context, batch);
  if (rc != 0) throw LlamaDecodeError(rc);
  trace_decode_call(trace_enabled, context, "qwen25_packed", layout.tokens, 0, start);
  next_position = layout.next_position;
}

int run_qwen25vl_query_suffix_ingest(llama_context *llctx,
                                     const llama_model *model,
                                     const std::string &suffix_prompt,
                                     llama_pos &next_position,
                                     const bool trace_enabled,
                                     const llama_seq_id sequence_id = 0) {
  const std::vector<llama_token> vision_end_and_suffix_tokens =
      tokenize_qwen25vl_query_suffix(model, suffix_prompt);

  if (!vision_end_and_suffix_tokens.empty()) {
    const auto start = Clock::now();
    decode_token_batch_at_sequence(llctx, vision_end_and_suffix_tokens,
                                   next_position, sequence_id, true);
    trace_decode_call(trace_enabled, llctx, "vision_end_and_suffix",
                      static_cast<int32_t>(vision_end_and_suffix_tokens.size()),
                      next_position, start);
    next_position +=
        static_cast<llama_pos>(vision_end_and_suffix_tokens.size());
  }
  return static_cast<int>(vision_end_and_suffix_tokens.size());
}

// Shared Qwen2.5-VL ingestion helper for whole-request and stateful-session
// entry points. Keep this composition byte-for-byte equivalent to the original
// one-shot path while exposing the query boundary to local sequence forks.
void run_qwen25vl_multimodal_ingest(llama_context *llctx,
                                    const llama_model *model,
                                    const edgeflow_llm_request *req,
                                    llama_pos &next_position,
                                    const bool trace_enabled,
                                    const llama_seq_id sequence_id = 0) {
  const std::string rendered_prompt =
      render_user_prompt(model, std::string(kImageTag) + req->prompt);
  const auto [prefix_prompt, suffix_prompt] =
      split_single_image_prompt(rendered_prompt);
  run_qwen25vl_visual_prefix_ingest(llctx, model, req, prefix_prompt,
                                    next_position, trace_enabled, sequence_id);
  run_qwen25vl_query_suffix_ingest(llctx, model, suffix_prompt, next_position,
                                   trace_enabled, sequence_id);
}

// FastVLM / LLaVA-style multimodal ingestion.
//   - No <|vision_start|> / <|vision_end|> wrapper tokens.
//   - No spatial merge math; num_visual_tokens is taken at face value.
//   - Visual embedding batch decoded via 1D positional layout.
//   - Accepts fp16 or fp32 visual_embd; fp16 is converted to fp32 once here
//     (llama_batch.embd is float*, upstream llama.cpp constraint).
//
// Token position layout (for num_visual_tokens = 256, prefix_len = L):
//   [0,  L)                        - prefix text tokens
//   [L,  L + 256)                  - visual embedding rows (sequential 1D positions)
//   [L + 256, L + 256 + suffix_len) - suffix text tokens
void run_fastvlm_multimodal_ingest(
    llama_context * llctx,
    const llama_model * model,
    const edgeflow_llm_request_v2 * req,
    llama_pos & next_position,
    const bool trace_enabled) {
  validate_visual_request_fastvlm(req, model);

  // FastVLM parity with Apple's LLaVA `qwen_2` conv_template requires:
  //   1. A prepended "You are a helpful assistant." system message.
  //   2. The user content built as `[img-0]\n<prompt>` — the '\n' after the
  //      image marker mirrors Apple's `DEFAULT_IMAGE_TOKEN + "\n" + prompt`
  //      construction in predict.py, and lands the newline token in the
  //      suffix after the split.
  std::string user_content;
  user_content.reserve(std::strlen(kImageTag) + 1 + std::strlen(req->prompt));
  user_content.append(kImageTag);
  user_content.push_back('\n');
  user_content.append(req->prompt);

  const std::string rendered_prompt = render_user_prompt_with_system(
      model, user_content, "You are a helpful assistant.");
  const auto [prefix_prompt, suffix_prompt] = split_single_image_prompt(rendered_prompt);

  const std::vector<llama_token> prefix_tokens =
      tokenize_prompt(model, prefix_prompt, /*add_special=*/true,  /*parse_special=*/true);
  const std::vector<llama_token> suffix_tokens =
      tokenize_prompt(model, suffix_prompt, /*add_special=*/false, /*parse_special=*/true);

  // 1. Prefix text tokens at positions [0, prefix_len).
  if (!prefix_tokens.empty()) {
    const auto start = Clock::now();
    decode_token_batch_at(llctx, prefix_tokens, next_position, /*logits_last=*/false);
    trace_decode_call(
        trace_enabled,
        llctx,
        "fastvlm_prefix",
        static_cast<int32_t>(prefix_tokens.size()),
        next_position,
        start);
    next_position += static_cast<llama_pos>(prefix_tokens.size());
  }

  // 2. Convert fp16 -> fp32 if needed, then decode visual embedding batch with
  // 1D positional layout starting at the current next_position.
  const int32_t n_tokens = req->num_visual_tokens;
  const int32_t n_embd   = req->hidden_size;
  std::vector<float> embd_fp32_staging;
  const float * embd_fp32 = nullptr;
  if (req->visual_dtype == EDGEFLOW_DTYPE_FP32) {
    embd_fp32 = static_cast<const float *>(req->visual_embd);
  } else {
    embd_fp32_staging.resize(static_cast<size_t>(n_tokens) * static_cast<size_t>(n_embd));
    ggml_fp16_to_fp32_row(
        static_cast<const ggml_fp16_t *>(req->visual_embd),
        embd_fp32_staging.data(),
        static_cast<int64_t>(n_tokens) * static_cast<int64_t>(n_embd));
    embd_fp32 = embd_fp32_staging.data();
  }

  {
    const auto start = Clock::now();
    decode_embedding_batch_1d(llctx, embd_fp32, n_tokens, n_embd, next_position);
    trace_decode_call(trace_enabled, llctx, "fastvlm_visual_embd", n_tokens, next_position, start);
  }
  next_position += static_cast<llama_pos>(n_tokens);

  // 3. Suffix text tokens at positions [prefix_len + n_tokens, ...).
  if (!suffix_tokens.empty()) {
    const auto start = Clock::now();
    decode_token_batch_at(llctx, suffix_tokens, next_position, /*logits_last=*/true);
    trace_decode_call(
        trace_enabled,
        llctx,
        "fastvlm_suffix",
        static_cast<int32_t>(suffix_tokens.size()),
        next_position,
        start);
    next_position += static_cast<llama_pos>(suffix_tokens.size());
  }
}

// LLaVA-1.5 multimodal ingestion. Same -200 single-sentinel + 1D positional
// embedding layout as FastVLM, but the text backbone is LLaMA-2-7B-chat which
// does NOT carry the official LLaVA-1.5 `conv_llava_v1` template in its GGUF
// metadata. We therefore bypass `llama_chat_apply_template` and hand-build
// the Vicuna-style prompt string verbatim from LLaVA's conversation.py.
//
// Pre-split string:
//   "A chat between a curious human and an artificial intelligence assistant. "
//   "The assistant gives helpful, detailed, and polite answers to the human's "
//   "questions. USER: [img-0]\n{user_prompt} ASSISTANT:"
//
// After split_single_image_prompt:
//   prefix = "...USER: "
//   suffix = "\n{user_prompt} ASSISTANT:"
void run_llava15_multimodal_ingest(
    llama_context * llctx,
    const llama_model * model,
    const edgeflow_llm_request_v2 * req,
    llama_pos & next_position,
    const bool trace_enabled) {
  validate_visual_request_llava15(req, model);

  static constexpr char kLLaVA15System[] =
      "A chat between a curious human and an artificial intelligence assistant. "
      "The assistant gives helpful, detailed, and polite answers to the human's "
      "questions. USER: ";
  static constexpr char kLLaVA15AssistantSuffix[] = " ASSISTANT:";

  std::string raw_prompt;
  raw_prompt.reserve(
      std::strlen(kLLaVA15System)
      + std::strlen(kImageTag)
      + 1
      + std::strlen(req->prompt)
      + std::strlen(kLLaVA15AssistantSuffix));
  raw_prompt.append(kLLaVA15System);
  raw_prompt.append(kImageTag);
  raw_prompt.push_back('\n');
  raw_prompt.append(req->prompt);
  raw_prompt.append(kLLaVA15AssistantSuffix);

  const auto [prefix_prompt, suffix_prompt] = split_single_image_prompt(raw_prompt);

  const std::vector<llama_token> prefix_tokens =
      tokenize_prompt(model, prefix_prompt, /*add_special=*/true,  /*parse_special=*/true);
  const std::vector<llama_token> suffix_tokens =
      tokenize_prompt(model, suffix_prompt, /*add_special=*/false, /*parse_special=*/true);

  // 1. Prefix text tokens at positions [0, prefix_len).
  if (!prefix_tokens.empty()) {
    const auto start = Clock::now();
    decode_token_batch_at(llctx, prefix_tokens, next_position, /*logits_last=*/false);
    trace_decode_call(
        trace_enabled,
        llctx,
        "llava15_prefix",
        static_cast<int32_t>(prefix_tokens.size()),
        next_position,
        start);
    next_position += static_cast<llama_pos>(prefix_tokens.size());
  }

  // 2. Convert fp16 -> fp32 if needed, then decode visual embedding batch with
  // 1D positional layout starting at the current next_position.
  const int32_t n_tokens = req->num_visual_tokens;
  const int32_t n_embd   = req->hidden_size;
  std::vector<float> embd_fp32_staging;
  const float * embd_fp32 = nullptr;
  if (req->visual_dtype == EDGEFLOW_DTYPE_FP32) {
    embd_fp32 = static_cast<const float *>(req->visual_embd);
  } else {
    embd_fp32_staging.resize(static_cast<size_t>(n_tokens) * static_cast<size_t>(n_embd));
    ggml_fp16_to_fp32_row(
        static_cast<const ggml_fp16_t *>(req->visual_embd),
        embd_fp32_staging.data(),
        static_cast<int64_t>(n_tokens) * static_cast<int64_t>(n_embd));
    embd_fp32 = embd_fp32_staging.data();
  }

  {
    const auto start = Clock::now();
    decode_embedding_batch_1d(llctx, embd_fp32, n_tokens, n_embd, next_position);
    trace_decode_call(trace_enabled, llctx, "llava15_visual_embd", n_tokens, next_position, start);
  }
  next_position += static_cast<llama_pos>(n_tokens);

  // 3. Suffix text tokens at positions [prefix_len + n_tokens, ...).
  if (!suffix_tokens.empty()) {
    const auto start = Clock::now();
    decode_token_batch_at(llctx, suffix_tokens, next_position, /*logits_last=*/true);
    trace_decode_call(
        trace_enabled,
        llctx,
        "llava15_suffix",
        static_cast<int32_t>(suffix_tokens.size()),
        next_position,
        start);
    next_position += static_cast<llama_pos>(suffix_tokens.size());
  }
}

// Text-only ingestion (shared between v1 and v2 entry points). Pure
// refactor of the existing else-branch in edgeflow_llm_generate.
void run_text_only_ingest(
    llama_context * llctx,
    const llama_model * model,
    const char * prompt_cstr,
    llama_pos & next_position,
    const bool trace_enabled) {
  const std::string rendered_prompt = render_user_prompt(model, prompt_cstr);
  const std::vector<llama_token> prompt_tokens = tokenize_prompt(model, rendered_prompt, true, true);
  if (prompt_tokens.empty()) {
    throw LlamaError("prompt tokenization produced no tokens");
  }
  const auto start = Clock::now();
  decode_token_batch_at(llctx, prompt_tokens, next_position, true);
  trace_decode_call(trace_enabled, llctx, "prompt_only", static_cast<int32_t>(prompt_tokens.size()), next_position, start);
  next_position += static_cast<llama_pos>(prompt_tokens.size());
}

void reset_result(edgeflow_llm_result * out) {
  if (out == nullptr) {
    return;
  }
  out->text = nullptr;
  out->first_token_ms = 0.0f;
  out->prompt_eval_ms = 0.0f;
  out->eval_ms = 0.0f;
  out->eval_count = 0;
}

}  // namespace

struct edgeflow_llm_ctx {
  std::unique_ptr<llama_model, ModelDeleter> model;
  std::unique_ptr<llama_context, ContextDeleter> context;
  bool ignore_eog = false;
  edgeflow_llm_token_callback token_callback = nullptr;
  void *token_callback_data = nullptr;
};

edgeflow_llm_ctx * edgeflow_llm_init(const char * model_path, edgeflow_llm_init_params params) {
  if (model_path == nullptr || model_path[0] == '\0') {
    return nullptr;
  }

  try {
    ensure_backend_initialized();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = params.n_gpu_layers;
    mparams.use_mmap = true;
    mparams.use_mlock = false;
    mparams.vocab_only = false;

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (model == nullptr) {
      return nullptr;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = params.n_ctx > 0 ? static_cast<uint32_t>(params.n_ctx) : 0;
    cparams.n_batch = params.n_batch > 0 ? static_cast<uint32_t>(params.n_batch) : cparams.n_batch;
    cparams.n_ubatch = cparams.n_batch;
    cparams.n_seq_max = 1;
    cparams.n_threads = params.n_threads > 0 ? params.n_threads : cparams.n_threads;
    cparams.n_threads_batch = cparams.n_threads;
    cparams.embeddings = false;
    cparams.offload_kqv = true;
    cparams.no_perf = false;

    llama_context * context = llama_init_from_model(model, cparams);
    if (context == nullptr) {
      llama_model_free(model);
      return nullptr;
    }

    auto * out = new edgeflow_llm_ctx();
    out->model.reset(model);
    out->context.reset(context);
    return out;
  } catch (...) {
    return nullptr;
  }
}

void edgeflow_llm_free(edgeflow_llm_ctx * ctx) {
  delete ctx;
}

int edgeflow_llm_set_generation_options(edgeflow_llm_ctx *ctx, int ignore_eog,
                                        edgeflow_llm_token_callback callback, void *user_data) {
  if (ctx == nullptr || (ignore_eog != 0 && ignore_eog != 1)) {
    return 1;
  }
  ctx->ignore_eog = ignore_eog != 0;
  ctx->token_callback = callback;
  ctx->token_callback_data = user_data;
  return 0;
}

namespace {

// Shared generation loop — greedy sampling, reused by both v1 and v2 entry
// points. Assumes ingestion has already advanced `next_position` past the
// prompt, completed its work, and set `out->prompt_eval_ms`. Populates `out->text`,
// `out->first_token_ms`, `out->eval_ms`, and `out->eval_count` on success.
// first_token_ms measures the first decision, including immediate EOG with
// zero emitted tokens (eval_count == 0), not delivery of a text piece.
// eval_ms includes sampling, detokenization, and required continuation decodes.
// Returns the result code to forward from the entry point (0 on success,
// nonzero on allocation failure).
int run_generation_loop(
    edgeflow_llm_ctx * ctx,
    const int max_new_tokens,
    const bool trace_enabled,
    llama_pos next_position,
    edgeflow_llm_result * out) {
  std::string text;
  const auto loop_start = Clock::now();
  bool saw_first = false;
  std::vector<llama_token> one_token(1);
  const char *logit_directory = std::getenv("EDGEFLOW_LLM_TRACE_LOGITS_DIR");

  for (int generated = 0; generated < max_new_tokens; ++generated) {
    const llama_token next = greedy_sample(ctx->context.get(), ctx->model.get(), ctx->ignore_eog);
    if (logit_directory && logit_directory[0]) {
      const auto path = std::string(logit_directory) + "/" + std::to_string(generated) + ".f32";
      FILE *file = std::fopen(path.c_str(), "wb");
      if (!file) throw LlamaError("cannot create diagnostic logits file");
      const size_t count = llama_vocab_n_tokens(llama_model_get_vocab(ctx->model.get()));
      const size_t written = std::fwrite(llama_get_logits_ith(ctx->context.get(), -1), sizeof(float), count, file);
      std::fclose(file);
      if (written != count) throw LlamaError("incomplete diagnostic logits file");
    }
    // Record the first decision even if it emits no text because it is EOG.
    if (!saw_first) {
      out->first_token_ms = out->prompt_eval_ms + elapsed_ms(loop_start, Clock::now());
      saw_first = true;
    }
    if (llama_vocab_is_eog(llama_model_get_vocab(ctx->model.get()), next)) {
      break;
    }
    const std::string piece = token_to_piece(ctx->model.get(), next);
    text += piece;
    out->eval_count += 1;
    if (ctx->token_callback &&
        ctx->token_callback(next, piece.data(), piece.size(), ctx->token_callback_data) != 0) {
      throw LlamaError("token consumer aborted generation");
    }
    // Only compute logits that a subsequent sampling decision will consume.
    if (generated + 1 == max_new_tokens) {
      break;
    }
    one_token[0] = next;
    const auto start = Clock::now();
    decode_token_batch_at(ctx->context.get(), one_token, next_position, true);
    trace_decode_call(trace_enabled, ctx->context.get(), "decode_step", 1, next_position, start);
    next_position += 1;
  }

  // Complete required continuation work before reporting loop timing.
  llama_synchronize(ctx->context.get());
  out->eval_ms = elapsed_ms(loop_start, Clock::now());
  if (!saw_first) {
    out->first_token_ms = out->prompt_eval_ms;
  }

  out->text = static_cast<char *>(std::malloc(text.size() + 1));
  if (out->text == nullptr) {
    reset_result(out);
    return 6;
  }
  std::memcpy(out->text, text.c_str(), text.size() + 1);
  return 0;
}

}  // namespace

static int generate_qwen_request(edgeflow_llm_ctx * ctx, const edgeflow_llm_request * req,
                                edgeflow_llm_result * out, const float *packed = nullptr,
                                int prefix = 0, int suffix = 0) {
  reset_result(out);
  if (ctx == nullptr || req == nullptr || out == nullptr) {
    return 1;
  }
  if (ctx->model == nullptr || ctx->context == nullptr) {
    return 2;
  }
  if (req->prompt == nullptr) {
    return 3;
  }
  if (req->max_new_tokens <= 0) {
    return 4;
  }

  try {
    const bool clear_memory = memory_clear_enabled();
    const bool trace_enabled = trace_decode_enabled();
    trace_runtime_config(trace_enabled, ctx->context.get(), clear_memory);
    llama_memory_clear(llama_get_memory(ctx->context.get()), clear_memory);

    llama_pos next_position = 0;
    const auto prompt_start = Clock::now();

    const bool has_visual = req->visual_embd != nullptr;
    if (packed) {
      run_qwen25_packed_ingest(ctx->context.get(), ctx->model.get(), req, packed,
                              prefix, suffix, next_position, trace_enabled);
    } else if (has_visual) {
      run_qwen25vl_multimodal_ingest(
          ctx->context.get(), ctx->model.get(), req, next_position, trace_enabled);
    } else {
      run_text_only_ingest(
          ctx->context.get(), ctx->model.get(), req->prompt, next_position, trace_enabled);
    }

    llama_synchronize(ctx->context.get());
    const auto prompt_end = Clock::now();
    out->prompt_eval_ms = elapsed_ms(prompt_start, prompt_end);

    return run_generation_loop(ctx, req->max_new_tokens, trace_enabled, next_position, out);
  } catch (...) {
    // Failed ingestion or decoding may leave work queued on this context.
    llama_synchronize(ctx->context.get());
    reset_result(out);
    return 7;
  }
}

int edgeflow_llm_generate(edgeflow_llm_ctx *ctx, const edgeflow_llm_request *req,
                         edgeflow_llm_result *out) {
  return generate_qwen_request(ctx, req, out);
}

int edgeflow_llm_generate_packed(edgeflow_llm_ctx *ctx, const edgeflow_llm_request *req,
                                const float *packed, int prefix, int suffix,
                                edgeflow_llm_result *out) {
  reset_result(out);
  if (!packed || prefix <= 0 || suffix <= 0) return 1;
  return generate_qwen_request(ctx, req, out, packed, prefix, suffix);
}

int edgeflow_llm_qwen25_prompt_embeddings(edgeflow_llm_ctx *ctx, const char *prompt,
                                         float *embeddings, uint64_t capacity,
                                         int32_t sizes[3]) {
  if (!ctx || !ctx->model || !prompt || !sizes) return 1;
  try {
    const auto *model = ctx->model.get();
    if (model->arch != LLM_ARCH_QWEN2VL) return 2;
    const auto tokens = qwen25_prompt_tokens(model, prompt);
    const auto *table = model->tok_embd;
    sizes[0] = int32_t(tokens.first.size());
    sizes[1] = int32_t(tokens.second.size());
    sizes[2] = int32_t(table->ne[0]);
    if (!embeddings) return capacity == 0 ? 0 : 3;
    const auto all = concat_tokens(tokens.first, tokens.second);
    if (capacity < uint64_t(all.size()) * sizes[2]) return 3;
    const size_t row_bytes = ggml_row_size(table->type, sizes[2]);
    std::vector<uint8_t> row(row_bytes);
    const auto *traits = ggml_get_type_traits(table->type);
    if (table->type != GGML_TYPE_F32 && !traits->to_float) return 4;
    for (size_t i = 0; i < all.size(); ++i) {
      ggml_backend_tensor_get(table, row.data(), size_t(all[i]) * table->nb[1], row_bytes);
      float *target = embeddings + i * sizes[2];
      if (table->type == GGML_TYPE_F32) std::memcpy(target, row.data(), row_bytes);
      else traits->to_float(row.data(), target, sizes[2]);
    }
    return 0;
  } catch (...) {
    return 5;
  }
}

int edgeflow_llm_generate_v2(
    edgeflow_llm_ctx * ctx,
    const edgeflow_llm_request_v2 * req,
    edgeflow_llm_result * out) {
  reset_result(out);
  if (ctx == nullptr || req == nullptr || out == nullptr) {
    return 1;
  }
  if (ctx->model == nullptr || ctx->context == nullptr) {
    return 2;
  }
  if (req->prompt == nullptr) {
    return 3;
  }
  if (req->max_new_tokens <= 0) {
    return 4;
  }

  try {
    const bool clear_memory = memory_clear_enabled();
    const bool trace_enabled = trace_decode_enabled();
    trace_runtime_config(trace_enabled, ctx->context.get(), clear_memory);
    llama_memory_clear(llama_get_memory(ctx->context.get()), clear_memory);

    llama_pos next_position = 0;
    const auto prompt_start = Clock::now();

    const bool has_visual = req->visual_embd != nullptr;
    if (!has_visual) {
      run_text_only_ingest(
          ctx->context.get(), ctx->model.get(), req->prompt, next_position, trace_enabled);
    } else {
      switch (req->model_family) {
        case EDGEFLOW_MODEL_FAMILY_QWEN25VL: {
          // v2 shell dispatches Qwen2.5-VL back to the existing ingest helper.
          // Qwen2.5-VL requires fp32 visual_embd; reject fp16 explicitly.
          if (req->visual_dtype != EDGEFLOW_DTYPE_FP32) {
            throw LlamaError("Qwen2.5-VL via v2 ABI requires EDGEFLOW_DTYPE_FP32 visual_embd");
          }
          edgeflow_llm_request v1_view = {};
          v1_view.prompt            = req->prompt;
          v1_view.visual_embd       = static_cast<const float *>(req->visual_embd);
          v1_view.num_visual_tokens = req->num_visual_tokens;
          v1_view.hidden_size       = req->hidden_size;
          v1_view.image_grid_thw[0] = req->image_grid_thw[0];
          v1_view.image_grid_thw[1] = req->image_grid_thw[1];
          v1_view.image_grid_thw[2] = req->image_grid_thw[2];
          v1_view.max_new_tokens    = req->max_new_tokens;
          run_qwen25vl_multimodal_ingest(
              ctx->context.get(), ctx->model.get(), &v1_view, next_position, trace_enabled);
          break;
        }
        case EDGEFLOW_MODEL_FAMILY_FASTVLM: {
          run_fastvlm_multimodal_ingest(
              ctx->context.get(), ctx->model.get(), req, next_position, trace_enabled);
          break;
        }
        case EDGEFLOW_MODEL_FAMILY_LLAVA15: {
          run_llava15_multimodal_ingest(
              ctx->context.get(), ctx->model.get(), req, next_position, trace_enabled);
          break;
        }
        default:
          throw LlamaError("unknown model_family in edgeflow_llm_generate_v2");
      }
    }

    llama_synchronize(ctx->context.get());
    const auto prompt_end = Clock::now();
    out->prompt_eval_ms = elapsed_ms(prompt_start, prompt_end);

    return run_generation_loop(ctx, req->max_new_tokens, trace_enabled, next_position, out);
  } catch (...) {
    // Failed ingestion or decoding may leave work queued on this context.
    llama_synchronize(ctx->context.get());
    reset_result(out);
    return 7;
  }
}

void edgeflow_llm_free_result(edgeflow_llm_result * result) {
  if (result == nullptr) {
    return;
  }
  if (result->text != nullptr) {
    std::free(result->text);
  }
  reset_result(result);
}

namespace {

thread_local std::string g_session_init_error;

struct SessionSequenceState {
  bool active = false;
  llama_pos next_position = 0;
  struct VisualBoundary {
    llama_pos kv_position_max;
    int64_t token_count;
    llama_pos visual_start;
    bool forkable;
  };
  // M-RoPE position extent is not the number of occupied KV cells. Retain image
  // boundaries for capacity checks; historical multi-image forks stay gated.
  std::map<llama_pos, VisualBoundary> visual_boundaries;
  edgeflow::SessionLogits logits;
};

std::string sha256_file(const char *path) {
  std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(path, "rb"),
                                                     &std::fclose);
  if (file == nullptr) {
    throw LlamaError("failed to open model for identity hashing");
  }
  CC_SHA256_CTX context;
  CC_SHA256_Init(&context);
  std::array<unsigned char, 1024 * 1024> buffer = {};
  while (true) {
    const size_t count =
        std::fread(buffer.data(), 1, buffer.size(), file.get());
    if (count > 0) {
      CC_SHA256_Update(&context, buffer.data(), static_cast<CC_LONG>(count));
    }
    if (count != buffer.size()) {
      if (std::ferror(file.get()) != 0) {
        throw LlamaError("failed while hashing model contents");
      }
      break;
    }
  }
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest = {};
  CC_SHA256_Final(digest.data(), &context);
  constexpr char kHex[] = "0123456789abcdef";
  std::string output(digest.size() * 2, '0');
  for (size_t index = 0; index < digest.size(); ++index) {
    output[index * 2] = kHex[digest[index] >> 4];
    output[index * 2 + 1] = kHex[digest[index] & 0x0f];
  }
  return output;
}

std::string runtime_binary_sha256() {
  Dl_info info = {};
  if (dladdr(reinterpret_cast<const void *>(&edgeflow_llm_session_abi_version),
             &info) == 0 ||
      info.dli_fname == nullptr || info.dli_fname[0] == '\0') {
    throw LlamaError("failed to resolve the loaded runtime binary path");
  }
  return sha256_file(info.dli_fname);
}

void copy_sha256(char out[EDGEFLOW_LLM_SHA256_HEX_SIZE],
                 const std::string &value) {
  if (value.size() + 1 != EDGEFLOW_LLM_SHA256_HEX_SIZE) {
    throw LlamaError("runtime produced an invalid SHA-256 identity");
  }
  std::memcpy(out, value.c_str(), EDGEFLOW_LLM_SHA256_HEX_SIZE);
}

edgeflow_backend_placement observed_model_placement(const llama_model *model,
                                                    int &gpu_layers) {
  gpu_layers = 0;
  const int layer_count = llama_model_n_layer(model);
  auto is_gpu = [](ggml_backend_dev_t device) {
    if (device == nullptr) {
      return false;
    }
    const enum ggml_backend_dev_type type = ggml_backend_dev_type(device);
    return type == GGML_BACKEND_DEVICE_TYPE_GPU ||
           type == GGML_BACKEND_DEVICE_TYPE_IGPU;
  };
  for (int layer = 0; layer < layer_count; ++layer) {
    gpu_layers += is_gpu(model->dev_layer(layer)) ? 1 : 0;
  }
  gpu_layers += is_gpu(model->dev_output()) ? 1 : 0;
  if (gpu_layers == 0) {
    return EDGEFLOW_BACKEND_PLACEMENT_CPU_ONLY;
  }
  if (gpu_layers == layer_count + 1) {
    return EDGEFLOW_BACKEND_PLACEMENT_GPU_FULL;
  }
  return EDGEFLOW_BACKEND_PLACEMENT_MIXED;
}

bool valid_model_family(const edgeflow_model_family family) {
  return family == EDGEFLOW_MODEL_FAMILY_QWEN25VL ||
         family == EDGEFLOW_MODEL_FAMILY_FASTVLM ||
         family == EDGEFLOW_MODEL_FAMILY_LLAVA15 ||
         family == EDGEFLOW_MODEL_FAMILY_QWEN35;
}

bool valid_position_contract(const edgeflow_position_contract contract) {
  return contract == EDGEFLOW_POSITION_CONTRACT_TEXT_1D ||
         contract == EDGEFLOW_POSITION_CONTRACT_QWEN25VL_MROPE ||
         contract == EDGEFLOW_POSITION_CONTRACT_QWEN35_IMROPE;
}

} // namespace

struct edgeflow_llm_session {
  std::unique_ptr<llama_model, ModelDeleter> model;
  std::unique_ptr<llama_context, ContextDeleter> context;
  edgeflow_llm_session_signature signature = {};
  std::vector<SessionSequenceState> sequences;
  std::thread::id owner;
  std::string last_error;
  bool poisoned = false;
  bool ignore_eog = false;
  std::vector<int32_t> eog_ids;
};

struct edgeflow_llm_root_snapshot {
  std::vector<uint8_t> state;
  SessionSequenceState root;
  edgeflow_llm_session_signature signature = {};
};

namespace {

int session_fail(edgeflow_llm_session *session, const int code,
                 std::string message) {
  if (session != nullptr) {
    session->last_error = std::move(message);
  }
  return code;
}

int require_session_owner(edgeflow_llm_session *session) {
  if (session == nullptr) {
    return EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT;
  }
  if (session->owner != std::this_thread::get_id()) {
    return EDGEFLOW_LLM_SESSION_WRONG_THREAD;
  }
  if (session->poisoned) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_STATE,
                        "stateful session is unusable after a decode failure");
  }
  session->last_error.clear();
  return EDGEFLOW_LLM_SESSION_OK;
}

bool has_live_branches(const edgeflow_llm_session *session) {
  return std::any_of(
      session->sequences.begin() + 1, session->sequences.end(),
      [](const SessionSequenceState &sequence) { return sequence.active; });
}

bool root_signatures_match(const edgeflow_llm_session_signature &source,
                           const edgeflow_llm_session_signature &target) {
  return source.session_abi_version == target.session_abi_version &&
         source.n_ctx == target.n_ctx &&
         source.kv_unified == target.kv_unified &&
         source.model_family == target.model_family &&
         source.kv_type_k == target.kv_type_k &&
         source.kv_type_v == target.kv_type_v &&
         source.position_contract == target.position_contract &&
         source.prompt_contract_version == target.prompt_contract_version &&
         std::memcmp(source.model_sha256, target.model_sha256,
                     EDGEFLOW_LLM_SHA256_HEX_SIZE) == 0 &&
         std::memcmp(source.runtime_sha256, target.runtime_sha256,
                     EDGEFLOW_LLM_SHA256_HEX_SIZE) == 0;
}

void clear_session_root(edgeflow_llm_session *session) {
  llama_synchronize(session->context.get());
  llama_memory_clear(llama_get_memory(session->context.get()), true);
  for (auto &sequence : session->sequences) {
    sequence = SessionSequenceState{};
  }
  session->sequences.front().active = true;
}

SessionSequenceState *require_active_sequence(edgeflow_llm_session *session,
                                              const int sequence_id,
                                              int &status) {
  if (sequence_id < 0 ||
      sequence_id >= static_cast<int>(session->sequences.size())) {
    status =
        session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
                     "sequence_id is outside the configured session range");
    return nullptr;
  }
  SessionSequenceState &sequence =
      session->sequences[static_cast<size_t>(sequence_id)];
  if (!sequence.active) {
    status = session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_STATE,
                          "sequence is not active");
    return nullptr;
  }
  status = EDGEFLOW_LLM_SESSION_OK;
  return &sequence;
}

void refresh_current_logits(edgeflow_llm_session *session,
                            edgeflow::SessionLogits &logits) {
  const llama_vocab *vocab = llama_model_get_vocab(session->model.get());
  if (vocab == nullptr) {
    throw LlamaError("model vocabulary is unavailable");
  }
  const int32_t n_vocab = llama_vocab_n_tokens(vocab);
  float *values = llama_get_logits_ith(session->context.get(), -1);
  if (values == nullptr || n_vocab <= 0) {
    throw LlamaError("session logits are unavailable");
  }
  logits.assign(values, static_cast<size_t>(n_vocab));
}

// The pinned fork exposes exact occupancy; newer upstream does not. Detect the
// member rather than assuming a llama version or equating M-RoPE span with cells.
template <typename Cache>
auto cache_used_cells(const Cache *cache, int)
    -> decltype(cache->get_used_cells(), uint64_t{}) {
  return cache->get_used_cells();
}
template <typename Cache>
uint64_t cache_used_cells(const Cache *, long) {
  return std::numeric_limits<uint64_t>::max();
}

uint64_t conservative_session_cells(const edgeflow_llm_session *session) {
  uint64_t cells = 0;
  for (const auto &sequence : session->sequences) {
    if (!sequence.active) continue;
    const auto end = sequence.visual_boundaries.upper_bound(sequence.next_position);
    if (end == sequence.visual_boundaries.begin()) {
      cells += sequence.next_position;
    } else {
      const auto &boundary = *std::prev(end);
      cells += boundary.second.token_count + sequence.next_position - boundary.first;
    }
  }
  // Shared prefixes are deliberately counted per active sequence. This is a
  // safe upper bound, not a measurement of occupancy, and may reject early.
  return cells;
}

bool has_session_kv_capacity(edgeflow_llm_session *session,
                             uint64_t new_cells) {
  auto *memory = llama_get_memory(session->context.get());
  const auto *cache = dynamic_cast<const llama_kv_cache *>(memory);
#ifdef EDGEFLOW_HAS_HYBRID_MEMORY
  if (cache == nullptr) {
    const auto *hybrid = dynamic_cast<const llama_memory_hybrid *>(memory);
    if (hybrid != nullptr) cache = hybrid->get_mem_attn();
  }
#endif
  uint64_t used = cache ? cache_used_cells(cache, 0)
                        : std::numeric_limits<uint64_t>::max();
  if (used == std::numeric_limits<uint64_t>::max()) {
    used = conservative_session_cells(session);
  }
  const uint64_t capacity = llama_n_ctx(session->context.get());
  return used <= capacity && new_cells <= capacity - used;
}

int ingest_session_tokens(edgeflow_llm_session *session, const int sequence_id,
                          const int32_t *tokens, const int token_count) {
  int status = EDGEFLOW_LLM_SESSION_OK;
  SessionSequenceState *sequence =
      require_active_sequence(session, sequence_id, status);
  if (sequence == nullptr) {
    return status;
  }
  if (tokens == nullptr || token_count <= 0) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
                        "token input must be non-null and non-empty");
  }
  if (token_count > static_cast<int>(llama_n_batch(session->context.get()))) {
    return session_fail(
        session, EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
        "token input exceeds the configured logical batch size");
  }
  const int64_t end_position = static_cast<int64_t>(sequence->next_position) +
                               static_cast<int64_t>(token_count);
  if (end_position >
      static_cast<int64_t>(llama_n_ctx_seq(session->context.get()))) {
    return session_fail(
        session, EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
        "token input exceeds the configured per-sequence context size");
  }
  if (!has_session_kv_capacity(session, static_cast<uint64_t>(token_count))) {
    return session_fail(
        session, EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
        "token input exceeds available physical KV cell capacity");
  }
  const int n_vocab =
      llama_vocab_n_tokens(llama_model_get_vocab(session->model.get()));
  for (int i = 0; i < token_count; ++i) {
    if (tokens[i] < 0 || tokens[i] >= n_vocab) {
      return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
                          "token input contains an out-of-vocabulary id");
    }
  }

  bool decode_started = false;
  try {
    std::vector<llama_token> native_tokens(
        tokens, tokens + static_cast<size_t>(token_count));
    decode_started = true;
    decode_token_batch_at_sequence(
        session->context.get(), native_tokens, sequence->next_position,
        static_cast<llama_seq_id>(sequence_id), true);
    sequence->next_position = static_cast<llama_pos>(end_position);
    refresh_current_logits(session, sequence->logits);
    return EDGEFLOW_LLM_SESSION_OK;
  } catch (const LlamaDecodeError &exc) {
    session->poisoned = !exc.state_unchanged();
    return session_fail(session, EDGEFLOW_LLM_SESSION_RUNTIME_ERROR,
                        std::string("session token ingest rejected: ") +
                            exc.what());
  } catch (const std::exception &exc) {
    session->poisoned = decode_started;
    return session_fail(session, EDGEFLOW_LLM_SESSION_RUNTIME_ERROR,
                        std::string("session token ingest failed: ") +
                            exc.what());
  } catch (...) {
    session->poisoned = decode_started;
    return session_fail(session, EDGEFLOW_LLM_SESSION_RUNTIME_ERROR,
                        "session token ingest failed with an unknown error");
  }
}

int ingest_session_qwen(edgeflow_llm_session *session, int sequence_id,
                        SessionSequenceState &sequence,
                        const edgeflow_llm_request *request, bool completed,
                        uint64_t *borrowed_pointer = nullptr) {
  bool decode_started = false;
  try {
    // Validate the entire multi-batch operation before the first KV write.
    validate_visual_request(request, session->model.get());
    const auto [prefix_prompt, suffix_prompt] =
        split_single_image_prompt(render_user_prompt(
            session->model.get(),
            std::string(kImageTag) + (completed ? request->prompt : "")));
    const auto prefix = concat_tokens(
        tokenize_prompt(session->model.get(), prefix_prompt, true, true),
        tokenize_prompt(session->model.get(), kVisionStartText, false, true));
    const auto suffix = completed ? tokenize_qwen25vl_query_suffix(
                                        session->model.get(), suffix_prompt)
                                  : std::vector<llama_token>{};
    const int merged_h = request->image_grid_thw[1] / kQwenSpatialMergeSize;
    const int merged_w = request->image_grid_thw[2] / kQwenSpatialMergeSize;
    const int64_t image_end =
        static_cast<int64_t>(prefix.size()) + std::max(merged_h, merged_w);
    const int64_t end_position = image_end + suffix.size();
    const uint64_t image_cells = prefix.size() + request->num_visual_tokens;
    require_decode_batch_capacity(session->context.get(), prefix.size());
    require_decode_batch_capacity(session->context.get(),
                                  request->num_visual_tokens);
    require_decode_batch_capacity(session->context.get(), suffix.size());
    if (end_position > llama_n_ctx_seq(session->context.get()) ||
        !has_session_kv_capacity(session, image_cells + suffix.size())) {
      return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
                          "Qwen ingest exceeds available physical KV cell "
                          "capacity or context size");
    }
    const size_t values = static_cast<size_t>(request->num_visual_tokens) *
                          static_cast<size_t>(request->hidden_size);
    if (!std::all_of(request->visual_embd, request->visual_embd + values,
                     [](float value) { return std::isfinite(value); })) {
      return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
                          "Qwen ingest requires finite embeddings");
    }
    const auto visual_start = static_cast<llama_pos>(prefix.size());
    std::map<llama_pos, SessionSequenceState::VisualBoundary> boundaries;
    boundaries.emplace(static_cast<llama_pos>(image_end),
                       SessionSequenceState::VisualBoundary{
                           visual_start, static_cast<int64_t>(image_cells),
                           visual_start, !completed});
    decode_started = true;
    if (!prefix.empty()) {
      decode_token_batch_at_sequence(session->context.get(), prefix, 0,
                                     sequence_id, false);
    }
    decode_embedding_batch_2d_sequence(
        session->context.get(), request->visual_embd,
        request->num_visual_tokens, request->hidden_size, visual_start,
        merged_w, merged_h, sequence_id, borrowed_pointer);
    if (!suffix.empty()) {
      decode_token_batch_at_sequence(session->context.get(), suffix,
                                     static_cast<llama_pos>(image_end),
                                     sequence_id, true);
    }
    // A published image prefix must represent completed work, not enqueue time.
    llama_synchronize(session->context.get());
    if (completed) {
      refresh_current_logits(session, sequence.logits);
    } else {
      sequence.logits.clear();
    }
    sequence.next_position = static_cast<llama_pos>(end_position);
    sequence.visual_boundaries.swap(boundaries);
    return EDGEFLOW_LLM_SESSION_OK;
  } catch (const std::exception &exc) {
    // Once a visual ingest has started, a later rejected batch does not undo
    // earlier batches. Preserve fail-closed behavior for partial execution.
    if (decode_started) llama_synchronize(session->context.get());
    session->poisoned = decode_started;
    return session_fail(session,
                        decode_started ? EDGEFLOW_LLM_SESSION_RUNTIME_ERROR
                                       : EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
                        std::string("Qwen session ingest failed: ") +
                            exc.what());
  } catch (...) {
    if (decode_started) llama_synchronize(session->context.get());
    session->poisoned = decode_started;
    return session_fail(session, EDGEFLOW_LLM_SESSION_RUNTIME_ERROR,
                        "Qwen session ingest failed with an unknown error");
  }
}

} // namespace

extern "C" edgeflow_llm_session *
edgeflow_llm_session_init(const char *model_path,
                          const edgeflow_llm_session_params *params) {
  g_session_init_error.clear();
  if (model_path == nullptr || model_path[0] == '\0' || params == nullptr) {
    g_session_init_error = "model_path and session params are required";
    return nullptr;
  }
  if (params->n_ctx <= 0 || params->n_batch <= 0 || params->n_ubatch <= 0 ||
      params->n_ubatch > params->n_batch || params->n_seq_max < 2 ||
      params->n_threads <= 0) {
    g_session_init_error = "session dimensions and thread count are invalid";
    return nullptr;
  }
  if (params->kv_unified != 1) {
    g_session_init_error = "stateful local forks require kv_unified=1";
    return nullptr;
  }
  if (params->kv_type_k != EDGEFLOW_KV_DTYPE_FP16 ||
      params->kv_type_v != EDGEFLOW_KV_DTYPE_FP16) {
    g_session_init_error = "Session storage supports only fp16 K/V cache";
    return nullptr;
  }
  if (!valid_model_family(params->model_family) ||
      !valid_position_contract(params->position_contract) ||
      params->prompt_contract_version == 0) {
    g_session_init_error = "session compatibility metadata is invalid";
    return nullptr;
  }
  const bool qwen35 = params->model_family == EDGEFLOW_MODEL_FAMILY_QWEN35;
  if (qwen35 != (params->position_contract == EDGEFLOW_POSITION_CONTRACT_QWEN35_IMROPE) ||
      (qwen35 && params->prompt_contract_version != 1)) {
    g_session_init_error = "Qwen3.5 requires its IMRoPE and prompt-v1 contract";
    return nullptr;
  }

  try {
    ensure_backend_initialized();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params->n_gpu_layers;
    model_params.use_mmap = true;
    model_params.use_mlock = false;
    model_params.vocab_only = false;
    llama_model *model = llama_model_load_from_file(model_path, model_params);
    if (model == nullptr) {
      g_session_init_error = "failed to load the session model";
      return nullptr;
    }
    std::unique_ptr<llama_model, ModelDeleter> model_owner(model);
    if (qwen35) {
      char architecture[32] = {};
      llama_model_meta_val_str(model, "general.architecture", architecture, sizeof(architecture));
      if (std::strcmp(architecture, "qwen35") != 0 || !llama_model_is_hybrid(model)) {
        g_session_init_error = "Qwen3.5 contract requires a qwen35 hybrid GGUF";
        return nullptr;
      }
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = static_cast<uint32_t>(params->n_ctx);
    context_params.n_batch = static_cast<uint32_t>(params->n_batch);
    context_params.n_ubatch = static_cast<uint32_t>(params->n_ubatch);
    context_params.n_seq_max = static_cast<uint32_t>(params->n_seq_max);
    context_params.n_threads = params->n_threads;
    context_params.n_threads_batch = params->n_threads;
    context_params.type_k = GGML_TYPE_F16;
    context_params.type_v = GGML_TYPE_F16;
    context_params.embeddings = false;
    const bool use_gpu = params->n_gpu_layers > 0;
    context_params.offload_kqv = use_gpu;
    context_params.op_offload = use_gpu;
    context_params.no_perf = false;
    context_params.kv_unified = true;

    llama_context *context = llama_init_from_model(model, context_params);
    if (context == nullptr) {
      g_session_init_error = "failed to create the stateful llama context";
      return nullptr;
    }
    std::unique_ptr<llama_context, ContextDeleter> context_owner(context);

    auto session = std::make_unique<edgeflow_llm_session>();
    session->model = std::move(model_owner);
    session->context = std::move(context_owner);
    session->owner = std::this_thread::get_id();
    session->sequences.resize(llama_n_seq_max(context));
    session->sequences.front().active = true;

    session->signature.session_abi_version = EDGEFLOW_LLM_SESSION_ABI_VERSION;
    session->signature.requested_gpu_layers = params->n_gpu_layers;
    session->signature.backend_placement = observed_model_placement(
        session->model.get(), session->signature.observed_gpu_layers);
    session->signature.n_ctx = static_cast<int>(llama_n_ctx(context));
    session->signature.n_batch = static_cast<int>(llama_n_batch(context));
    session->signature.n_ubatch = static_cast<int>(llama_n_ubatch(context));
    session->signature.n_seq_max = static_cast<int>(llama_n_seq_max(context));
    session->signature.n_threads = params->n_threads;
    session->signature.kv_unified = 1;
    session->signature.model_family = params->model_family;
    session->signature.kv_type_k = params->kv_type_k;
    session->signature.kv_type_v = params->kv_type_v;
    session->signature.position_contract = params->position_contract;
    session->signature.prompt_contract_version =
        params->prompt_contract_version;
    copy_sha256(session->signature.model_sha256, sha256_file(model_path));
    copy_sha256(session->signature.runtime_sha256, runtime_binary_sha256());
    copy_sha256(session->signature.runtime_source_fingerprint_sha256,
                EDGEFLOW_LLM_RUNTIME_SOURCE_FINGERPRINT_SHA256);
    return session.release();
  } catch (const std::exception &exc) {
    g_session_init_error = exc.what();
    return nullptr;
  } catch (...) {
    g_session_init_error =
        "stateful session initialization failed with an unknown error";
    return nullptr;
  }
}

extern "C" void edgeflow_llm_session_free(edgeflow_llm_session *session) {
  delete session;
}

extern "C" const char *edgeflow_llm_session_last_init_error(void) {
  return g_session_init_error.c_str();
}

extern "C" const char *
edgeflow_llm_session_last_error(const edgeflow_llm_session *session) {
  if (session == nullptr) {
    return "stateful session is null";
  }
  if (session->owner != std::this_thread::get_id()) {
    return "stateful session called from wrong thread";
  }
  return session->last_error.c_str();
}

extern "C" int
edgeflow_llm_session_get_signature(edgeflow_llm_session *session,
                                   edgeflow_llm_session_signature *out) {
  const int status = require_session_owner(session);
  if (status != EDGEFLOW_LLM_SESSION_OK) {
    return status;
  }
  if (out == nullptr) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
                        "signature output is required");
  }
  *out = session->signature;
  return EDGEFLOW_LLM_SESSION_OK;
}

extern "C" int edgeflow_llm_session_vocab_size(edgeflow_llm_session *session) {
  const int status = require_session_owner(session);
  if (status != EDGEFLOW_LLM_SESSION_OK) {
    return -status;
  }
  const llama_vocab *vocab = llama_model_get_vocab(session->model.get());
  const int n_vocab = vocab == nullptr ? 0 : llama_vocab_n_tokens(vocab);
  if (n_vocab <= 0) {
    session_fail(session, EDGEFLOW_LLM_SESSION_RUNTIME_ERROR,
                 "session model vocabulary is unavailable");
    return -EDGEFLOW_LLM_SESSION_RUNTIME_ERROR;
  }
  return n_vocab;
}

extern "C" int
edgeflow_llm_session_get_sequence_info(edgeflow_llm_session *session,
                                       const int sequence_id,
                                       edgeflow_llm_sequence_info *out) {
  const int owner_status = require_session_owner(session);
  if (owner_status != EDGEFLOW_LLM_SESSION_OK) {
    return owner_status;
  }
  if (out == nullptr || sequence_id < 0 ||
      sequence_id >= static_cast<int>(session->sequences.size())) {
    return session_fail(
        session, EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
        "sequence info requires a valid output and sequence_id");
  }
  const SessionSequenceState &sequence =
      session->sequences[static_cast<size_t>(sequence_id)];
  out->active = sequence.active ? 1 : 0;
  out->next_position = static_cast<int>(sequence.next_position);
  out->has_logits = sequence.logits.values().empty() ? 0 : 1;
  out->state_bytes = sequence.active
                         ? static_cast<uint64_t>(llama_state_seq_get_size(
                               session->context.get(), sequence_id))
                         : 0;
  return EDGEFLOW_LLM_SESSION_OK;
}

extern "C" int edgeflow_llm_session_export_root(
    edgeflow_llm_session *session, edgeflow_llm_root_snapshot **out) {
  const int status = require_session_owner(session);
  if (status != EDGEFLOW_LLM_SESSION_OK) {
    return status;
  }
  if (out == nullptr) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
                        "root snapshot output is required");
  }
  *out = nullptr;
  try {
    auto snapshot = std::make_unique<edgeflow_llm_root_snapshot>();
    snapshot->signature = session->signature;
    snapshot->root.active = true;
    snapshot->root.next_position = session->sequences.front().next_position;
    snapshot->root.visual_boundaries =
        session->sequences.front().visual_boundaries;
    llama_synchronize(session->context.get());
    const size_t size = llama_state_seq_get_size(session->context.get(), 0);
    if (size == 0) {
      throw LlamaError("llama.cpp failed to size the root state");
    }
    snapshot->state.resize(size);
    if (llama_state_seq_get_data(session->context.get(), snapshot->state.data(),
                                 size, 0) != size) {
      throw LlamaError("llama.cpp failed to export the complete root state");
    }
    *out = snapshot.release();
    return EDGEFLOW_LLM_SESSION_OK;
  } catch (const std::exception &exc) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_RUNTIME_ERROR,
                        std::string("root export failed: ") + exc.what());
  } catch (...) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_RUNTIME_ERROR,
                        "root export failed with an unknown error");
  }
}

extern "C" int edgeflow_llm_session_import_root(
    edgeflow_llm_session *session,
    const edgeflow_llm_root_snapshot *snapshot) {
  const int status = require_session_owner(session);
  if (status != EDGEFLOW_LLM_SESSION_OK) {
    return status;
  }
  if (snapshot == nullptr) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
                        "root snapshot is required");
  }
  if (has_live_branches(session)) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_STATE,
                        "root import requires no active response branches");
  }
  if (!root_signatures_match(snapshot->signature, session->signature)) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
                        "root snapshot compatibility contract does not match");
  }
  bool mutation_started = false;
  try {
    // Copy before clearing so an allocation failure preserves the old root.
    SessionSequenceState root = snapshot->root;
    root.logits.clear();
    mutation_started = true;
    clear_session_root(session);
    if (llama_state_seq_set_data(session->context.get(), snapshot->state.data(),
                                 snapshot->state.size(), 0) !=
        snapshot->state.size()) {
      throw LlamaError("llama.cpp failed to import the complete root state");
    }
    llama_synchronize(session->context.get());
    session->sequences.front() = std::move(root);
    return EDGEFLOW_LLM_SESSION_OK;
  } catch (const std::exception &exc) {
    session->poisoned = mutation_started;
    return session_fail(session, EDGEFLOW_LLM_SESSION_RUNTIME_ERROR,
                        std::string("root import failed: ") + exc.what());
  } catch (...) {
    session->poisoned = mutation_started;
    return session_fail(session, EDGEFLOW_LLM_SESSION_RUNTIME_ERROR,
                        "root import failed with an unknown error");
  }
}

extern "C" void
edgeflow_llm_root_snapshot_free(edgeflow_llm_root_snapshot *snapshot) {
  delete snapshot;
}

extern "C" uint64_t edgeflow_llm_root_snapshot_size(
    const edgeflow_llm_root_snapshot *snapshot) {
  return snapshot == nullptr ? 0 : static_cast<uint64_t>(snapshot->state.size());
}

extern "C" int edgeflow_llm_root_snapshot_equal(
    const edgeflow_llm_root_snapshot *left,
    const edgeflow_llm_root_snapshot *right) {
  if (!left || !right) return -EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT;
  if (!root_signatures_match(left->signature, right->signature) ||
      left->state != right->state || left->root.active != right->root.active ||
      left->root.next_position != right->root.next_position ||
      left->root.visual_boundaries.size() != right->root.visual_boundaries.size()) {
    return 0;
  }
  auto other = right->root.visual_boundaries.begin();
  for (const auto &[position, boundary] : left->root.visual_boundaries) {
    const auto &[other_position, b] = *other++;
    if (position != other_position || boundary.kv_position_max != b.kv_position_max ||
        boundary.token_count != b.token_count || boundary.visual_start != b.visual_start ||
        boundary.forkable != b.forkable) return 0;
  }
  return 1;
}

extern "C" int
edgeflow_llm_session_reset_root(edgeflow_llm_session *session) {
  const int status = require_session_owner(session);
  if (status != EDGEFLOW_LLM_SESSION_OK) {
    return status;
  }
  if (has_live_branches(session)) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_STATE,
                        "root reset requires no active response branches");
  }
  try {
    clear_session_root(session);
    return EDGEFLOW_LLM_SESSION_OK;
  } catch (const std::exception &exc) {
    session->poisoned = true;
    return session_fail(session, EDGEFLOW_LLM_SESSION_RUNTIME_ERROR,
                        std::string("root reset failed: ") + exc.what());
  } catch (...) {
    session->poisoned = true;
    return session_fail(session, EDGEFLOW_LLM_SESSION_RUNTIME_ERROR,
                        "root reset failed with an unknown error");
  }
}

extern "C" int edgeflow_llm_session_ingest_text(
    edgeflow_llm_session *session, const int sequence_id, const char *text,
    const int add_special, const int parse_special, int *out_token_count) {
  const int owner_status = require_session_owner(session);
  if (owner_status != EDGEFLOW_LLM_SESSION_OK) {
    return owner_status;
  }
  if (text == nullptr || text[0] == '\0' || out_token_count == nullptr) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
                        "text and token-count output are required");
  }
  try {
    const std::vector<llama_token> tokens = tokenize_prompt(
        session->model.get(), text, add_special != 0, parse_special != 0);
    if (tokens.empty()) {
      return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
                          "text tokenization produced no tokens");
    }
    const int status = ingest_session_tokens(
        session, sequence_id, reinterpret_cast<const int32_t *>(tokens.data()),
        static_cast<int>(tokens.size()));
    if (status == EDGEFLOW_LLM_SESSION_OK) {
      *out_token_count = static_cast<int>(tokens.size());
    }
    return status;
  } catch (const std::exception &exc) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_RUNTIME_ERROR,
                        std::string("session text ingest failed: ") +
                            exc.what());
  } catch (...) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_RUNTIME_ERROR,
                        "session text ingest failed with an unknown error");
  }
}

extern "C" int edgeflow_llm_session_ingest_user_prompt(
    edgeflow_llm_session *session, const int sequence_id, const char *prompt,
    int *out_token_count) {
  const int owner_status = require_session_owner(session);
  if (owner_status != EDGEFLOW_LLM_SESSION_OK) {
    return owner_status;
  }
  if (prompt == nullptr || prompt[0] == '\0' || out_token_count == nullptr) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
                        "user prompt and token-count output are required");
  }
  try {
    const std::string rendered =
        render_user_prompt(session->model.get(), prompt);
    const std::vector<llama_token> tokens =
        tokenize_prompt(session->model.get(), rendered, true, true);
    if (tokens.empty()) {
      return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
                          "rendered user prompt produced no tokens");
    }
    const int status = ingest_session_tokens(
        session, sequence_id, reinterpret_cast<const int32_t *>(tokens.data()),
        static_cast<int>(tokens.size()));
    if (status == EDGEFLOW_LLM_SESSION_OK) {
      *out_token_count = static_cast<int>(tokens.size());
    }
    return status;
  } catch (const std::exception &exc) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_RUNTIME_ERROR,
                        std::string("session user-prompt ingest failed: ") +
                            exc.what());
  } catch (...) {
    return session_fail(
        session, EDGEFLOW_LLM_SESSION_RUNTIME_ERROR,
        "session user-prompt ingest failed with an unknown error");
  }
}

extern "C" int edgeflow_llm_session_ingest_tokens(edgeflow_llm_session *session,
                                                  const int sequence_id,
                                                  const int32_t *tokens,
                                                  const int token_count) {
  const int owner_status = require_session_owner(session);
  if (owner_status != EDGEFLOW_LLM_SESSION_OK) {
    return owner_status;
  }
  return ingest_session_tokens(session, sequence_id, tokens, token_count);
}

extern "C" int
edgeflow_llm_session_ingest_qwen25vl(edgeflow_llm_session *session,
                                     const int sequence_id,
                                     const edgeflow_llm_request *request) {
  const int owner_status = require_session_owner(session);
  if (owner_status != EDGEFLOW_LLM_SESSION_OK) {
    return owner_status;
  }
  int sequence_status = EDGEFLOW_LLM_SESSION_OK;
  SessionSequenceState *sequence =
      require_active_sequence(session, sequence_id, sequence_status);
  if (sequence == nullptr) {
    return sequence_status;
  }
  if (request == nullptr || request->prompt == nullptr) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
                        "Qwen session ingest requires a request and prompt");
  }
  if (session->signature.model_family != EDGEFLOW_MODEL_FAMILY_QWEN25VL ||
      session->signature.position_contract !=
          EDGEFLOW_POSITION_CONTRACT_QWEN25VL_MROPE) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_STATE,
                        "Qwen session ingest requires the Qwen2.5-VL M-RoPE "
                        "compatibility contract");
  }
  if (sequence->next_position != 0) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_STATE,
                        "Completed-image Qwen ingest requires "
                        "an empty sequence");
  }

  return ingest_session_qwen(session, sequence_id, *sequence, request, true);
}

extern "C" int edgeflow_llm_session_ingest_qwen25vl_visual_prefix(
    edgeflow_llm_session *session, const int sequence_id,
    const edgeflow_llm_request *request) {
  const int owner_status = require_session_owner(session);
  if (owner_status != EDGEFLOW_LLM_SESSION_OK) {
    return owner_status;
  }
  int sequence_status = EDGEFLOW_LLM_SESSION_OK;
  SessionSequenceState *sequence =
      require_active_sequence(session, sequence_id, sequence_status);
  if (sequence == nullptr) {
    return sequence_status;
  }
  if (request == nullptr) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
                        "Qwen visual-prefix ingest requires a request");
  }
  if (session->signature.model_family != EDGEFLOW_MODEL_FAMILY_QWEN25VL ||
      session->signature.position_contract !=
          EDGEFLOW_POSITION_CONTRACT_QWEN25VL_MROPE) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_STATE,
                        "Qwen visual-prefix ingest requires the Qwen2.5-VL "
                        "M-RoPE compatibility contract");
  }
  if (sequence->next_position != 0) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_STATE,
                        "Qwen visual-prefix ingest requires an empty sequence");
  }

  return ingest_session_qwen(session, sequence_id, *sequence, request, false);
}

static int append_session_qwen_visual(
    edgeflow_llm_session *session, const int sequence_id,
    const edgeflow_llm_request *request, bool qwen35,
    uint64_t *borrowed_pointer) {
  const int owner_status = require_session_owner(session);
  if (owner_status != EDGEFLOW_LLM_SESSION_OK) {
    return owner_status;
  }
  int sequence_status = EDGEFLOW_LLM_SESSION_OK;
  SessionSequenceState *sequence =
      require_active_sequence(session, sequence_id, sequence_status);
  if (sequence == nullptr) {
    return sequence_status;
  }
  if (request == nullptr) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
                        "Qwen visual append requires a request");
  }
  const bool first = qwen35 && sequence->next_position == 0 &&
                     sequence->visual_boundaries.empty();
  if (session->signature.model_family != (qwen35 ? EDGEFLOW_MODEL_FAMILY_QWEN35 : EDGEFLOW_MODEL_FAMILY_QWEN25VL) ||
      session->signature.position_contract !=
          (qwen35 ? EDGEFLOW_POSITION_CONTRACT_QWEN35_IMROPE : EDGEFLOW_POSITION_CONTRACT_QWEN25VL_MROPE) ||
      (!first && sequence->visual_boundaries.count(sequence->next_position) == 0)) {
    return session_fail(
        session, EDGEFLOW_LLM_SESSION_INVALID_STATE,
        "Qwen visual append requires an unmodified image boundary");
  }
  if (sequence_id != 0 ||
      std::any_of(
          session->sequences.begin() + 1, session->sequences.end(),
          [](const SessionSequenceState &state) { return state.active; })) {
    return session_fail(
        session, EDGEFLOW_LLM_SESSION_INVALID_STATE,
        "visual append requires root zero and no active response "
        "branches; interleaved append fidelity is not validated");
  }

  bool decode_started = false;
  try {
    validate_visual_request(request, session->model.get());
    const int merged_h = request->image_grid_thw[1] / kQwenSpatialMergeSize;
    const int merged_w = request->image_grid_thw[2] / kQwenSpatialMergeSize;
    const std::vector<llama_token> markers = tokenize_prompt(
        session->model.get(), first ? "<|im_start|>user\n<|vision_start|>"
                                   : std::string(kVisionEndText) + kVisionStartText,
        false, true);
    require_decode_batch_capacity(session->context.get(), markers.size());
    require_decode_batch_capacity(session->context.get(),
                                  request->num_visual_tokens);
    const int64_t token_count =
        (first ? 0 : sequence->visual_boundaries.at(sequence->next_position).token_count) +
        static_cast<int64_t>(markers.size()) + request->num_visual_tokens;
    const int64_t end_position = static_cast<int64_t>(sequence->next_position) +
                                 markers.size() + std::max(merged_h, merged_w);
    if (token_count > llama_n_ctx_seq(session->context.get()) ||
        end_position > llama_n_ctx_seq(session->context.get()) ||
        !has_session_kv_capacity(session,
                                 markers.size() + request->num_visual_tokens)) {
      return session_fail(
          session, EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
          "Qwen visual append exceeds the bounded context capacity");
    }
    const size_t values = static_cast<size_t>(request->num_visual_tokens) *
                          static_cast<size_t>(request->hidden_size);
    if (!std::all_of(request->visual_embd, request->visual_embd + values,
                     [](float value) { return std::isfinite(value); })) {
      return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
                          "Qwen visual append requires finite embeddings");
    }
    auto boundaries = sequence->visual_boundaries;
    const llama_pos visual_start = sequence->next_position + markers.size();
    boundaries.emplace(static_cast<llama_pos>(end_position),
                       SessionSequenceState::VisualBoundary{
                           visual_start, token_count, visual_start, true});
    decode_started = true;
    decode_token_batch_at_sequence(session->context.get(), markers,
                                   sequence->next_position, sequence_id, false);
    decode_embedding_batch_2d_sequence(
        session->context.get(), request->visual_embd,
        request->num_visual_tokens, request->hidden_size, visual_start,
        merged_w, merged_h, sequence_id, borrowed_pointer);
    llama_synchronize(session->context.get());
    const llama_pos actual_max = llama_memory_seq_pos_max(
        llama_get_memory(session->context.get()), sequence_id);
    if (actual_max != visual_start) {
      throw LlamaError("Qwen visual append produced an unexpected KV position");
    }
    sequence->next_position = static_cast<llama_pos>(end_position);
    sequence->visual_boundaries.swap(boundaries);
    sequence->logits.clear();
    return EDGEFLOW_LLM_SESSION_OK;
  } catch (const std::exception &exc) {
    if (decode_started) llama_synchronize(session->context.get());
    session->poisoned = session->poisoned || decode_started;
    return session_fail(session,
                        decode_started ? EDGEFLOW_LLM_SESSION_RUNTIME_ERROR
                                       : EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
                        std::string("Qwen visual append failed: ") +
                            exc.what());
  } catch (...) {
    if (decode_started) llama_synchronize(session->context.get());
    session->poisoned = session->poisoned || decode_started;
    return session_fail(session, EDGEFLOW_LLM_SESSION_RUNTIME_ERROR,
                        "Qwen visual append failed with an unknown error");
  }
}

extern "C" int edgeflow_llm_session_append_qwen25vl_visual(
    edgeflow_llm_session *session, int sequence_id, const edgeflow_llm_request *request) {
  return append_session_qwen_visual(session, sequence_id, request, false, nullptr);
}

extern "C" uint32_t edgeflow_llm_surface_append_receipt_sizeof(void) {
  return sizeof(edgeflow_llm_surface_append_receipt);
}

extern "C" int edgeflow_llm_session_append_qwen25vl_visual_borrowed(
    edgeflow_llm_session *session, int sequence_id, const edgeflow_llm_request *request,
    edgeflow_llm_surface_append_receipt *receipt) {
  if (receipt != nullptr) *receipt = {};
  const int owner = require_session_owner(session);
  if (owner != EDGEFLOW_LLM_SESSION_OK) return owner;
  if (request == nullptr || receipt == nullptr || sequence_id != 0)
    return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
                        "borrowed visual append requires root zero and outputs");
  if (session->signature.model_family != EDGEFLOW_MODEL_FAMILY_QWEN25VL ||
      session->signature.position_contract != EDGEFLOW_POSITION_CONTRACT_QWEN25VL_MROPE ||
      has_live_branches(session))
    return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_STATE,
                        "borrowed Qwen2.5 append requires its contract and no live branches");
  auto &root = session->sequences[0];
  const int rc = root.next_position == 0
      ? ingest_session_qwen(session, 0, root, request, false, &receipt->decoder_pointer)
      : append_session_qwen_visual(session, 0, request, false, &receipt->decoder_pointer);
  if (rc == 0) {
    receipt->payload_bytes = uint64_t(request->num_visual_tokens) * request->hidden_size * sizeof(float);
    receipt->next_position = root.next_position;
  }
  return rc;
}

extern "C" int edgeflow_llm_session_append_qwen35_visual(
    edgeflow_llm_session *session, int sequence_id, const edgeflow_llm_request *request,
    edgeflow_llm_surface_append_receipt *receipt) {
  if (receipt == nullptr) return EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT;
  *receipt = {};
  const int rc = append_session_qwen_visual(session, sequence_id, request, true,
                                           &receipt->decoder_pointer);
  if (rc == 0) {
    receipt->payload_bytes = uint64_t(request->num_visual_tokens) * request->hidden_size * sizeof(float);
    receipt->next_position = session->sequences[sequence_id].next_position;
  }
  return rc;
}

static int ingest_session_qwen_query(
    edgeflow_llm_session *session, const int sequence_id, const char *prompt,
    int *out_token_count, bool qwen35) {
  const int owner_status = require_session_owner(session);
  if (owner_status != EDGEFLOW_LLM_SESSION_OK) {
    return owner_status;
  }
  int sequence_status = EDGEFLOW_LLM_SESSION_OK;
  SessionSequenceState *sequence =
      require_active_sequence(session, sequence_id, sequence_status);
  if (sequence == nullptr) {
    return sequence_status;
  }
  if (prompt == nullptr || prompt[0] == '\0' || out_token_count == nullptr) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
                        "Qwen query ingest requires a prompt and token-count "
                        "output");
  }
  if (session->signature.model_family != (qwen35 ? EDGEFLOW_MODEL_FAMILY_QWEN35 : EDGEFLOW_MODEL_FAMILY_QWEN25VL) ||
      session->signature.position_contract !=
          (qwen35 ? EDGEFLOW_POSITION_CONTRACT_QWEN35_IMROPE : EDGEFLOW_POSITION_CONTRACT_QWEN25VL_MROPE) ||
      (qwen35 && sequence_id == 0)) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_STATE,
                        "Qwen query ingest requires the Qwen2.5-VL M-RoPE "
                        "compatibility contract");
  }
  if (sequence->visual_boundaries.count(sequence->next_position) == 0) {
    return session_fail(
        session, EDGEFLOW_LLM_SESSION_INVALID_STATE,
        "Qwen query ingest requires an unmodified visual-prefix boundary");
  }

  try {
    const std::string rendered_prompt = qwen35 ? "" : render_user_prompt(
        session->model.get(), std::string(kImageTag) + prompt);
    const auto [prefix_prompt, suffix_prompt] = qwen35
        ? std::make_pair(std::string{}, std::string(prompt))
        : split_single_image_prompt(rendered_prompt);
    (void)prefix_prompt;
    const std::vector<llama_token> tokens =
        qwen35 ? tokenize_prompt(session->model.get(),
                    std::string(kVisionEndText) + suffix_prompt +
                    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", false, true)
               : tokenize_qwen25vl_query_suffix(session->model.get(), suffix_prompt);
    // Pre-decode validation must not invalidate the root or sibling branches.
    // The shared token-ingest path owns poisoning around state-changing decode.
    const int status = ingest_session_tokens(
        session, sequence_id, reinterpret_cast<const int32_t *>(tokens.data()),
        static_cast<int>(tokens.size()));
    if (status != EDGEFLOW_LLM_SESSION_OK) {
      return status;
    }
    *out_token_count = static_cast<int>(tokens.size());
    return EDGEFLOW_LLM_SESSION_OK;
  } catch (const std::exception &exc) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_RUNTIME_ERROR,
                        std::string("Qwen query ingest failed: ") + exc.what());
  } catch (...) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_RUNTIME_ERROR,
                        "Qwen query ingest failed with an unknown error");
  }
}

extern "C" int edgeflow_llm_session_ingest_qwen25vl_query(
    edgeflow_llm_session *session, int sequence_id, const char *prompt, int *count) {
  return ingest_session_qwen_query(session, sequence_id, prompt, count, false);
}

extern "C" int edgeflow_llm_session_ingest_qwen35_query(
    edgeflow_llm_session *session, int sequence_id, const char *prompt, int *count) {
  return ingest_session_qwen_query(session, sequence_id, prompt, count, true);
}

extern "C" int edgeflow_llm_session_fork_prefix(
    edgeflow_llm_session *session, const int source_sequence_id,
    const int destination_sequence_id, const int end_position) {
  const int owner_status = require_session_owner(session);
  if (owner_status != EDGEFLOW_LLM_SESSION_OK) {
    return owner_status;
  }
  int source_status = EDGEFLOW_LLM_SESSION_OK;
  SessionSequenceState *source =
      require_active_sequence(session, source_sequence_id, source_status);
  if (source == nullptr) {
    return source_status;
  }
  if (destination_sequence_id < 0 ||
      destination_sequence_id >= static_cast<int>(session->sequences.size()) ||
      destination_sequence_id == source_sequence_id || end_position <= 0 ||
      end_position > source->next_position) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
                        "fork requires distinct valid sequences and a source "
                        "prefix end position");
  }
  if (session->model &&
      (llama_model_is_recurrent(session->model.get()) ||
       llama_model_is_hybrid(session->model.get())) &&
      (source_sequence_id != 0 || end_position != source->next_position)) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_STATE,
                        "recurrent models require a full-current-root fork; "
                        "historical ranges and branch-source forks are unsupported");
  }
  if (source->visual_boundaries.size() > 1 &&
      end_position != source->next_position) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_STATE,
                        "historical multi-image forks have not passed the "
                        "strict local fidelity gate");
  }
  for (const auto &[image_end, boundary] : source->visual_boundaries) {
    if (end_position > boundary.visual_start && end_position < image_end) {
      return session_fail(
          session, EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
          "fork prefix cannot end inside an image position interval");
    }
    if (end_position == image_end && !boundary.forkable) {
      return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_STATE,
                          "completed-image ingest does not expose a "
                          "visual-prefix fork boundary");
    }
  }
  SessionSequenceState &destination =
      session->sequences[static_cast<size_t>(destination_sequence_id)];
  if (destination.active) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_STATE,
                        "fork destination sequence is already active");
  }

  bool destination_may_have_state = false;
  try {
    // Allocate before mutating llama.cpp state so allocation failure cannot
    // strand a partially initialized destination sequence.
    edgeflow::SessionLogits fork_logits;
    if (end_position == source->next_position) {
      fork_logits = source->logits;
    }
    const auto boundary = source->visual_boundaries.find(end_position);
    std::map<llama_pos, SessionSequenceState::VisualBoundary> fork_boundaries(
        source->visual_boundaries.begin(),
        source->visual_boundaries.upper_bound(end_position));

    llama_memory_t memory = llama_get_memory(session->context.get());
    destination_may_have_state = true;
    llama_memory_seq_cp(memory, static_cast<llama_seq_id>(source_sequence_id),
                        static_cast<llama_seq_id>(destination_sequence_id), 0,
                        static_cast<llama_pos>(end_position));
    const llama_pos copied_max = llama_memory_seq_pos_max(
        memory, static_cast<llama_seq_id>(destination_sequence_id));
    const llama_pos expected_copied_max =
        boundary != source->visual_boundaries.end()
            ? boundary->second.kv_position_max
            : static_cast<llama_pos>(end_position - 1);
    if (copied_max != expected_copied_max) {
      if (!llama_memory_seq_rm(memory, destination_sequence_id, -1, -1)) {
        session->poisoned = true;
        return session_fail(
            session, EDGEFLOW_LLM_SESSION_RUNTIME_ERROR,
            "fork prefix verification and destination cleanup both failed; "
            "the session is unusable");
      }
      return session_fail(
          session, EDGEFLOW_LLM_SESSION_RUNTIME_ERROR,
          "llama.cpp did not materialize the requested fork prefix");
    }

    destination.active = true;
    destination.next_position = static_cast<llama_pos>(end_position);
    destination.visual_boundaries = std::move(fork_boundaries);
    destination.logits = std::move(fork_logits);
    return EDGEFLOW_LLM_SESSION_OK;
  } catch (const std::exception &exc) {
    if (destination_may_have_state &&
        !llama_memory_seq_rm(llama_get_memory(session->context.get()),
                             destination_sequence_id, -1, -1)) {
      session->poisoned = true;
      return session_fail(session, EDGEFLOW_LLM_SESSION_RUNTIME_ERROR,
                          "fork failure and destination cleanup both failed; "
                          "the session is unusable");
    }
    destination = SessionSequenceState{};
    return session_fail(
        session, EDGEFLOW_LLM_SESSION_RUNTIME_ERROR,
        std::string("session fork failed before publication: ") + exc.what());
  } catch (...) {
    if (destination_may_have_state &&
        !llama_memory_seq_rm(llama_get_memory(session->context.get()),
                             destination_sequence_id, -1, -1)) {
      session->poisoned = true;
      return session_fail(session, EDGEFLOW_LLM_SESSION_RUNTIME_ERROR,
                          "fork failure and destination cleanup both failed; "
                          "the session is unusable");
    }
    destination = SessionSequenceState{};
    return session_fail(session, EDGEFLOW_LLM_SESSION_RUNTIME_ERROR,
                        "session fork failed before publication");
  }
}

extern "C" int edgeflow_llm_session_set_ignore_eog(
    edgeflow_llm_session *session, int ignore_eog) {
  const int status = require_session_owner(session);
  if (status != EDGEFLOW_LLM_SESSION_OK) return status;
  if (ignore_eog != 0 && ignore_eog != 1)
    return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
                        "ignore_eog must be zero or one");
  try {
    if (ignore_eog && session->eog_ids.empty()) {
      const auto *vocab = llama_model_get_vocab(session->model.get());
      std::vector<int32_t> ids;
      for (int32_t id = 0; id < llama_vocab_n_tokens(vocab); ++id)
        if (llama_vocab_is_eog(vocab, id)) ids.push_back(id);
      session->eog_ids = std::move(ids);
    }
    session->ignore_eog = ignore_eog != 0;
    return EDGEFLOW_LLM_SESSION_OK;
  } catch (const std::exception &exc) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_RUNTIME_ERROR, exc.what());
  }
}

static llama_token session_argmax(edgeflow_llm_session *session,
                                  edgeflow::SessionLogits &logits) {
  const auto token = logits.argmax();
  // Keep the cached SIMD decision on the usual non-EOG path.
  return session->ignore_eog &&
                 std::binary_search(session->eog_ids.begin(), session->eog_ids.end(), token)
             ? logits.argmax_excluding(session->eog_ids)
             : token;
}

extern "C" int edgeflow_llm_session_peek_token(
    edgeflow_llm_session *session, const int sequence_id, int *out_token_id,
    int *out_is_eog) {
  const int status = require_session_owner(session);
  if (status != EDGEFLOW_LLM_SESSION_OK) return status;
  int sequence_status = EDGEFLOW_LLM_SESSION_OK;
  SessionSequenceState *sequence =
      require_active_sequence(session, sequence_id, sequence_status);
  if (sequence == nullptr) return sequence_status;
  if (out_token_id == nullptr || out_is_eog == nullptr)
    return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
                        "token and EOG outputs are required");
  if (sequence->logits.values().empty())
    return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_STATE,
                        "peek requires fresh sequence logits");
  try {
    const llama_token token = session_argmax(session, sequence->logits);
    *out_is_eog =
        llama_vocab_is_eog(llama_model_get_vocab(session->model.get()), token)
            ? 1 : 0;
    *out_token_id = static_cast<int>(token);
    return EDGEFLOW_LLM_SESSION_OK;
  } catch (const std::exception &exc) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_RUNTIME_ERROR,
                        std::string("session peek failed: ") + exc.what());
  } catch (...) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_RUNTIME_ERROR,
                        "session peek failed with an unknown error");
  }
}

extern "C" int
edgeflow_llm_session_decode_step(edgeflow_llm_session *session,
                                 const int sequence_id,
                                 edgeflow_llm_decode_step_result *out,
                                 float *out_logits, const int logits_capacity) {
  const int owner_status = require_session_owner(session);
  if (owner_status != EDGEFLOW_LLM_SESSION_OK) {
    return owner_status;
  }
  int sequence_status = EDGEFLOW_LLM_SESSION_OK;
  SessionSequenceState *sequence =
      require_active_sequence(session, sequence_id, sequence_status);
  if (sequence == nullptr) {
    return sequence_status;
  }
  if (out == nullptr || sequence->logits.values().empty() ||
      logits_capacity < 0 || (out_logits == nullptr && logits_capacity != 0)) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
                        "decode step requires active sequence logits and a "
                        "valid output buffer");
  }
  const int n_vocab = static_cast<int>(sequence->logits.values().size());
  if (out_logits != nullptr && logits_capacity < n_vocab) {
    return session_fail(
        session, EDGEFLOW_LLM_SESSION_BUFFER_TOO_SMALL,
        "decode-step logits buffer is smaller than the model vocabulary");
  }

  try {
    const llama_token token = session_argmax(session, sequence->logits);
    out->token_id = token;
    out->n_vocab = n_vocab;
    out->is_eog =
        llama_vocab_is_eog(llama_model_get_vocab(session->model.get()), token)
            ? 1
            : 0;
    if (out->is_eog != 0) {
      out->token_position = -1;
      out->next_token_id = token;
    } else {
      out->token_position = static_cast<int>(sequence->next_position);
      const int32_t native_token = token;
      const int status =
          ingest_session_tokens(session, sequence_id, &native_token, 1);
      if (status != EDGEFLOW_LLM_SESSION_OK) {
        return status;
      }
      out->next_token_id = session_argmax(session, sequence->logits);
    }
    if (out_logits != nullptr) {
      const auto &values = sequence->logits.values();
      std::copy(values.begin(), values.end(), out_logits);
    }
    return EDGEFLOW_LLM_SESSION_OK;
  } catch (const std::exception &exc) {
    session->poisoned = true;
    return session_fail(session, EDGEFLOW_LLM_SESSION_RUNTIME_ERROR,
                        std::string("session decode step failed: ") +
                            exc.what());
  } catch (...) {
    session->poisoned = true;
    return session_fail(session, EDGEFLOW_LLM_SESSION_RUNTIME_ERROR,
                        "session decode step failed with an unknown error");
  }
}

extern "C" int edgeflow_llm_session_token_piece(edgeflow_llm_session *session,
                                                const int token_id, char *out,
                                                const int capacity,
                                                int *out_size) {
  const int owner_status = require_session_owner(session);
  if (owner_status != EDGEFLOW_LLM_SESSION_OK) {
    return owner_status;
  }
  const llama_vocab *vocab = llama_model_get_vocab(session->model.get());
  const int n_vocab = vocab == nullptr ? 0 : llama_vocab_n_tokens(vocab);
  if (out_size == nullptr || token_id < 0 || token_id >= n_vocab ||
      capacity < 0 || (out == nullptr && capacity != 0)) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
                        "token piece requires a valid token and output buffer");
  }
  try {
    const std::string piece = token_to_piece(
        session->model.get(), static_cast<llama_token>(token_id));
    *out_size = static_cast<int>(piece.size());
    if (out == nullptr) {
      return EDGEFLOW_LLM_SESSION_OK;
    }
    if (capacity < static_cast<int>(piece.size())) {
      return session_fail(session, EDGEFLOW_LLM_SESSION_BUFFER_TOO_SMALL,
                          "token-piece output buffer is too small");
    }
    std::memcpy(out, piece.data(), piece.size());
    return EDGEFLOW_LLM_SESSION_OK;
  } catch (const std::exception &exc) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_RUNTIME_ERROR,
                        std::string("token-piece decoding failed: ") +
                            exc.what());
  } catch (...) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_RUNTIME_ERROR,
                        "token-piece decoding failed with an unknown error");
  }
}

extern "C" int
edgeflow_llm_session_remove_sequence(edgeflow_llm_session *session,
                                     const int sequence_id) {
  const int owner_status = require_session_owner(session);
  if (owner_status != EDGEFLOW_LLM_SESSION_OK) {
    return owner_status;
  }
  if (sequence_id == 0) {
    return session_fail(session, EDGEFLOW_LLM_SESSION_INVALID_STATE,
                        "sequence zero is the reserved root and cannot be "
                        "removed");
  }
  int sequence_status = EDGEFLOW_LLM_SESSION_OK;
  SessionSequenceState *sequence =
      require_active_sequence(session, sequence_id, sequence_status);
  if (sequence == nullptr) {
    return sequence_status;
  }
  const bool removed =
      llama_memory_seq_rm(llama_get_memory(session->context.get()),
                          static_cast<llama_seq_id>(sequence_id), -1, -1);
  if (!removed) {
    session->poisoned = true;
    return session_fail(session, EDGEFLOW_LLM_SESSION_RUNTIME_ERROR,
                        "llama.cpp failed to remove the complete sequence");
  }
  *sequence = SessionSequenceState{};
  return EDGEFLOW_LLM_SESSION_OK;
}
