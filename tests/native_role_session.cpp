// Execute production session functions with tiny storage/vocabulary stubs.
// No model, backend initialization or accelerator submission occurs.
#define EDGEFLOW_LLM_RUNTIME_SOURCE_FINGERPRINT_SHA256 \
  "0000000000000000000000000000000000000000000000000000000000000000"
#include "../tools/edgeflow_llm_runtime/src/edgeflow_llm.cpp"
#include <iostream>

namespace {
float next_logits[] = {1, 10, 9};
int forwards = 0, fences = 0;
int last_token = -1;
llama_pos last_position = -1;
const float *last_embedding = nullptr;
bool fail_visual = false;
void check(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
}

extern "C" void llama_model_free(llama_model *) {}
extern "C" void llama_free(llama_context *) {}
extern "C" const llama_vocab *llama_model_get_vocab(const llama_model *) {
  return reinterpret_cast<const llama_vocab *>(1);
}
extern "C" int32_t llama_vocab_n_tokens(const llama_vocab *) { return 3; }
extern "C" bool llama_vocab_is_eog(const llama_vocab *, llama_token id) { return id == 1; }
extern "C" int32_t llama_model_n_embd_inp(const llama_model *) { return 8; }
extern "C" uint32_t llama_n_batch(const llama_context *) { return 512; }
extern "C" uint32_t llama_n_ctx(const llama_context *) { return 8192; }
extern "C" uint32_t llama_n_ctx_seq(const llama_context *) { return 8192; }
extern "C" llama_memory_t llama_get_memory(const llama_context *) { return nullptr; }
extern "C" llama_pos llama_memory_seq_pos_max(llama_memory_t, llama_seq_id) { return last_position; }
extern "C" void llama_synchronize(llama_context *) { ++fences; }
extern "C" float *llama_get_logits_ith(llama_context *, int32_t) { return next_logits; }
extern "C" const char *llama_model_chat_template(const llama_model *, const char *) { return "chatml"; }
extern "C" int32_t llama_chat_apply_template(const char *, const llama_chat_message *messages,
    size_t, bool, char *output, int32_t capacity) {
  std::string text = std::string("header") + messages[0].content + "suffix";
  if (output && capacity > int(text.size())) std::memcpy(output, text.data(), text.size());
  return text.size();
}
extern "C" int32_t llama_tokenize(const llama_vocab *, const char *, int32_t,
    llama_token *tokens, int32_t capacity, bool, bool) {
  if (!tokens || capacity < 1) return -1;
  tokens[0] = 2;
  return 1;
}
extern "C" int32_t llama_decode(llama_context *, llama_batch batch) {
  ++forwards;
  last_position = batch.pos[0];
  if (batch.embd) {
    last_embedding = batch.embd;
    if (fail_visual) return -1;
    for (int i = 0; i < batch.n_tokens; ++i)
      check(batch.pos[3 * batch.n_tokens + i] == 0, "fourth position plane not zero");
  } else {
    last_token = batch.token[0];
  }
  return 0;
}

int main(int argc, char **argv) {
  try {
    check(argc == 2, "case required");
    edgeflow_llm_session session;
    session.owner = std::this_thread::get_id();
    session.model.reset(reinterpret_cast<llama_model *>(1));
    session.context.reset(reinterpret_cast<llama_context *>(1));
    session.sequences.resize(2);
    session.sequences[0].active = true;
    session.signature.model_family = EDGEFLOW_MODEL_FAMILY_QWEN25VL;
    session.signature.position_contract = EDGEFLOW_POSITION_CONTRACT_QWEN25VL_MROPE;
    const std::string test = argv[1];
    if (test == "eog") {
      float logits[] = {9, 10, 8};
      session.sequences[0].logits.assign(logits, 3);
      int token = -1, eog = -1;
      check(edgeflow_llm_session_peek_token(&session, 0, &token, &eog) == 0 && token == 1 && eog,
            "natural EOG changed");
      check(edgeflow_llm_session_set_ignore_eog(&session, 1) == 0, "setter failed");
      check(edgeflow_llm_session_peek_token(&session, 0, &token, &eog) == 0 && token == 0 && !eog,
            "peek failed to mask EOG");
      edgeflow_llm_decode_step_result out{};
      check(edgeflow_llm_session_decode_step(&session, 0, &out, nullptr, 0) == 0 &&
            out.token_id == 0 && out.next_token_id == 2 && !out.is_eog &&
            forwards == 1 && last_token == 0, "decode did not use masked choice on both sides");
      check(edgeflow_llm_session_set_ignore_eog(&session, 0) == 0, "disable failed");
      check(edgeflow_llm_session_peek_token(&session, 0, &token, &eog) == 0 && token == 1,
            "natural EOG not restored");
      check(edgeflow_llm_session_set_ignore_eog(&session, 2) == EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
            "invalid flag accepted");
    } else if (test == "snapshot") {
      edgeflow_llm_root_snapshot left, right;
      left.signature = right.signature = session.signature;
      left.state = right.state = {0, 1, 2, 3};
      left.root.active = right.root.active = true;
      left.root.next_position = right.root.next_position = 4;
      left.root.visual_boundaries[4] = right.root.visual_boundaries[4] = {3, 4, 1, true};
      check(edgeflow_llm_root_snapshot_equal(&left, &right) == 1, "equal snapshot rejected");
      right.state[2] = 9;
      check(edgeflow_llm_root_snapshot_equal(&left, &right) == 0, "same-size corrupt KV accepted");
      right.state[2] = 2;
      right.root.next_position = 5;
      check(edgeflow_llm_root_snapshot_equal(&left, &right) == 0, "wrong position accepted");
      right.root.next_position = 4;
      right.root.visual_boundaries[4].token_count = 5;
      check(edgeflow_llm_root_snapshot_equal(&left, &right) == 0, "wrong boundary accepted");
      right.root.visual_boundaries[4].token_count = 4;
      right.signature.position_contract = EDGEFLOW_POSITION_CONTRACT_TEXT_1D;
      check(edgeflow_llm_root_snapshot_equal(&left, &right) == 0, "wrong contract accepted");
      check(edgeflow_llm_root_snapshot_equal(nullptr, &right) < 0, "null snapshot accepted");
    } else {
      float visual[4 * 8] = {};
      edgeflow_llm_request request{};
      request.visual_embd = visual;
      request.num_visual_tokens = 4;
      request.hidden_size = 8;
      request.image_grid_thw[0] = 1;
      request.image_grid_thw[1] = request.image_grid_thw[2] = 4;
      edgeflow_llm_surface_append_receipt receipt{};
      fail_visual = test == "failure";
      int rc = edgeflow_llm_session_append_qwen25vl_visual_borrowed(&session, 0, &request, &receipt);
      if (fail_visual) {
        check(rc != 0 && session.poisoned && fences > 0, "failed first ingest returned before fencing");
      } else {
        check(rc == 0 && last_embedding == visual && receipt.decoder_pointer == uintptr_t(visual)
                  && fences > 0 && receipt.payload_bytes == sizeof(visual), "prefix was not borrowed/completed");
        const int first = receipt.next_position;
        rc = edgeflow_llm_session_append_qwen25vl_visual_borrowed(&session, 0, &request, &receipt);
        check(rc == 0 && receipt.next_position > first && receipt.decoder_pointer == uintptr_t(visual),
              "incremental append failed");
        const int before = forwards;
        session.sequences[1].active = true;
        check(edgeflow_llm_session_append_qwen25vl_visual_borrowed(&session, 0, &request, &receipt)
                  == EDGEFLOW_LLM_SESSION_INVALID_STATE && forwards == before,
              "live branch allowed mutation of root");
      }
    }
    std::cout << "PASS\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
