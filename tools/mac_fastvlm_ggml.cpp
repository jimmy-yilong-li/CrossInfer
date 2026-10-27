// Retained FastVLM language path using upstream llama.cpp.
#include "ggml-backend.h"
#include "llama.h"
#include "ane_bridge.h"
#include "mac_fastvlm_borrowed_batch.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <time.h>

// Additive C ABI records, mirrored by the small ctypes adapter.
struct fv_query_info {
    double prefill_finished_s;  // Python monotonic epoch; NaN if prefill did not finish.
    int32_t actual_count;
    int32_t root;
    int32_t cleanup_ok;         // Also true for preflight rejection with unchanged KV.
    int32_t logits_captured;
};

struct fv_handoff_info {
    double text_ingest_started_s;
    double text_ingest_finished_s;
    double visual_ingest_started_s;
    double visual_ingest_finished_s;
    double surface_setup_s;    // Callback entry/validation, excluding language ingest.
    double surface_teardown_s; // Callback exit/unlock, excluding language ingest.
};

namespace {
thread_local std::string error;

double now() {
    timespec stamp{};
    // CPython on macOS uses mach_absolute_time, exposed as UPTIME_RAW here.
    clock_gettime(CLOCK_UPTIME_RAW, &stamp);
    return stamp.tv_sec + stamp.tv_nsec * 1e-9;
}

struct State {
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    llama_sampler * sampler = nullptr;
    int n_ctx = 4096;
    int position = 0;
    int pending_root = -1;
    int pending_visual_count = 0;
    bool needs_reset = false;
    uintptr_t embedding_input_pointer = 0;
    fv_handoff_info last_handoff{};
    ~State() {
        if (sampler) llama_sampler_free(sampler);
        if (context) llama_free(context);
        if (model) llama_model_free(model);
    }
};

State & ready(void * pointer) {
    if (!pointer) throw std::runtime_error("language context is null");
    auto & state = *static_cast<State *>(pointer);
    if (state.needs_reset) throw std::runtime_error("suffix cleanup failed; reset required");
    return state;
}

void preflight(const State & state, int64_t count) {
    if (count <= 0 || count > int64_t(state.n_ctx) - state.position)
        throw std::runtime_error("input is empty or exceeds context");
}

// Even a failed decode may have touched KV. Never reuse an unclean context.
bool restore_root(State & state, int root) {
    llama_synchronize(state.context);
    const bool removed = llama_memory_seq_rm(llama_get_memory(state.context), 0, root, -1);
    llama_sampler_reset(state.sampler);
    llama_synchronize(state.context);
    if (removed) state.position = root;
    else state.needs_reset = true;
    return removed;
}

void ingest(State & state, const int32_t * tokens, const float * features,
            int count, int width, bool logits_last = true) {
    preflight(state, count);
    const int batch_size = int(llama_n_batch(state.context));
    for (int offset = 0; offset < count; offset += batch_size) {
        const int size = std::min(batch_size, count - offset);
        BorrowedBatch batch(tokens ? tokens + offset : nullptr,
                            features ? features + size_t(offset) * width : nullptr,
                            size, state.position, logits_last && offset + size == count);
        if (features && offset == 0)
            state.embedding_input_pointer = reinterpret_cast<uintptr_t>(batch.value.embd);
        const int status = llama_decode(state.context, batch.value);
        if (status != 0) throw std::runtime_error("llama_decode failed");
        state.position += size;
    }
    llama_synchronize(state.context);
}

struct SurfaceAppend {
    State * state;
    int count;
    int width;
    double started_s;
};

int consume_embeddings(const void * base, size_t size, void * context,
                       char * message, size_t capacity) {
    auto & request = *static_cast<SurfaceAppend *>(context);
    try {
        if (!base || size < size_t(request.count) * request.width * sizeof(float))
            throw std::runtime_error("leased surface is smaller than visual payload");
        auto & timing = request.state->last_handoff;
        timing.visual_ingest_started_s = now();
        timing.surface_setup_s = timing.visual_ingest_started_s - request.started_s;
        ingest(*request.state, nullptr, static_cast<const float *>(base),
               request.count, request.width, false);
        timing.visual_ingest_finished_s = now();
        return 0;
    } catch (const std::exception & exception) {
        llama_synchronize(request.state->context);
        std::snprintf(message, capacity, "%s", exception.what());
        return 1;
    }
}

int append(void * pointer, const int32_t * tokens, int token_count,
           const float * features, ANESurfaceLeaseHandle * lease,
           int visual_count, int width, bool surface) {
    auto * state = static_cast<State *>(pointer);
    const bool staged = state && state->pending_root >= 0;
    const int root = state ? (staged ? state->pending_root : state->position) : 0;
    bool started = staged;
    error.clear();
    if (state) {
        if (!staged) state->last_handoff = {};
        state->embedding_input_pointer = 0;
    }
    try {
        ready(pointer);
        if (staged && (tokens || token_count != 0 || visual_count != state->pending_visual_count))
            throw std::runtime_error("pending frame requires its reserved visual rows only");
        if (!staged && (!tokens || token_count <= 0))
            throw std::runtime_error("text input is empty");
        if (visual_count <= 0 || width != llama_model_n_embd(state->model) || width <= 0)
            throw std::runtime_error("visual shape does not match language model");
        if (surface ? !lease : !features)
            throw std::runtime_error("invalid visual input");
        preflight(*state, int64_t(token_count) + visual_count);
        started = true;
        auto & timing = state->last_handoff;
        if (!staged) {
            timing.text_ingest_started_s = now();
            ingest(*state, tokens, nullptr, token_count, 0, false);
            timing.text_ingest_finished_s = now();
        }
        if (surface) {
            SurfaceAppend request{state, visual_count, width, now()};
            char message[512]{};
            if (ane_bridge_consume_surface_lease(lease, consume_embeddings, &request,
                                                message, sizeof(message)) != 0)
                throw std::runtime_error(message[0] ? message : "surface consume failed");
            timing.surface_teardown_s = now() - timing.visual_ingest_finished_s;
        } else {
            timing.visual_ingest_started_s = now();
            ingest(*state, nullptr, features, visual_count, width, false);
            timing.visual_ingest_finished_s = now();
        }
        state->pending_root = -1;
        state->pending_visual_count = 0;
        return state->position;
    } catch (const std::exception & exception) {
        error = exception.what();
        if (started && !restore_root(*state, root))
            error += "; failed to remove append suffix; reset required";
        if (state) {
            state->pending_root = -1;
            state->pending_visual_count = 0;
            state->embedding_input_pointer = 0;
            state->last_handoff = {};
        }
        return -1;
    }
}
} // namespace

extern "C" {
double fv_now() { return now(); }
const char * fv_error() { return error.c_str(); }

void * fv_init_with_context(const char * model_path, const char * backend_directory, int n_ctx) {
    error.clear();
    try {
        if (n_ctx <= 0 || n_ctx > 32768)
            throw std::runtime_error("n_ctx must be in [1, 32768]");
        if (!model_path || !backend_directory)
            throw std::runtime_error("model and backend paths are required");
        ggml_backend_load_all_from_path(backend_directory);
        llama_backend_init();
        auto state = std::make_unique<State>();
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = 999;
        state->model = llama_model_load_from_file(model_path, model_params);
        if (!state->model) throw std::runtime_error("model load failed");
        if (n_ctx > llama_model_n_ctx_train(state->model))
            throw std::runtime_error("n_ctx exceeds model context limit");
        state->n_ctx = n_ctx;
        auto params = llama_context_default_params();
        params.n_ctx = n_ctx;
        params.n_batch = params.n_ubatch = 512;
        params.n_seq_max = 1;
        params.n_threads = params.n_threads_batch = 4;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        params.kv_unified = true;
        params.type_k = params.type_v = GGML_TYPE_F16;
        state->context = llama_init_from_model(state->model, params);
        if (!state->context) throw std::runtime_error("context creation failed");
        if (llama_n_ctx(state->context) < uint32_t(n_ctx))
            throw std::runtime_error("allocated context is smaller than n_ctx");
        state->sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
        if (!state->sampler) throw std::runtime_error("sampler creation failed");
        auto * greedy = llama_sampler_init_greedy();
        if (!greedy) throw std::runtime_error("greedy sampler creation failed");
        llama_sampler_chain_add(state->sampler, greedy);
        return state.release();
    } catch (const std::exception & exception) {
        error = exception.what();
        return nullptr;
    }
}

void * fv_init(const char * model_path, const char * backend_directory) {
    return fv_init_with_context(model_path, backend_directory, 4096);
}

void fv_free(void * pointer) { delete static_cast<State *>(pointer); }

int fv_reset(void * pointer) {
    error.clear();
    if (!pointer) { error = "language context is null"; return -1; }
    auto & state = *static_cast<State *>(pointer);
    llama_synchronize(state.context);
    llama_memory_clear(llama_get_memory(state.context), true);
    llama_sampler_reset(state.sampler);
    state.position = 0;
    state.pending_root = -1;
    state.pending_visual_count = 0;
    state.needs_reset = false;
    state.embedding_input_pointer = 0;
    state.last_handoff = {};
    return 0;
}

int fv_begin_append(void * pointer, const int32_t * tokens, int token_count, int visual_count) {
    auto * state = static_cast<State *>(pointer);
    bool started = false;
    error.clear();
    try {
        ready(pointer);
        if (state->pending_root >= 0) throw std::runtime_error("frame append is already pending");
        if (!tokens || token_count <= 0 || visual_count <= 0)
            throw std::runtime_error("frame requires nonempty text and visual rows");
        preflight(*state, int64_t(token_count) + visual_count);
        state->pending_root = state->position;
        state->pending_visual_count = visual_count;
        state->last_handoff = {};
        state->embedding_input_pointer = 0;
        started = true;
        state->last_handoff.text_ingest_started_s = now();
        ingest(*state, tokens, nullptr, token_count, 0, false);
        state->last_handoff.text_ingest_finished_s = now();
        return state->position;
    } catch (const std::exception & exception) {
        error = exception.what();
        if (started) {
            if (!restore_root(*state, state->pending_root))
                error += "; failed to remove frame label; reset required";
            state->pending_root = -1;
            state->pending_visual_count = 0;
            state->last_handoff = {};
        }
        return -1;
    }
}

int fv_cancel_append(void * pointer) {
    error.clear();
    if (!pointer) { error = "language context is null"; return -1; }
    auto & state = *static_cast<State *>(pointer);
    if (state.pending_root < 0) return state.position;
    const bool restored = restore_root(state, state.pending_root);
    state.pending_root = -1;
    state.pending_visual_count = 0;
    state.last_handoff = {};
    state.embedding_input_pointer = 0;
    if (!restored) { error = "failed to cancel frame; reset required"; return -1; }
    return state.position;
}

int fv_append(void * pointer, const int32_t * tokens, int token_count,
              const float * features, int visual_count, int width) {
    return append(pointer, tokens, token_count, features, nullptr, visual_count, width, false);
}

int fv_append_surface(void * pointer, const int32_t * tokens, int token_count,
                      ANESurfaceLeaseHandle * lease, int visual_count, int width) {
    return append(pointer, tokens, token_count, nullptr, lease, visual_count, width, true);
}

uint64_t fv_last_embedding_pointer(void * pointer) {
    return pointer ? static_cast<State *>(pointer)->embedding_input_pointer : 0;
}

int fv_last_handoff(void * pointer, fv_handoff_info * info) {
    if (!pointer || !info) { error = "handoff context and output are required"; return -1; }
    *info = static_cast<State *>(pointer)->last_handoff;
    return 0;
}

int fv_vocab_size(void * pointer) {
    if (!pointer) { error = "language context is null"; return -1; }
    return llama_vocab_n_tokens(llama_model_get_vocab(static_cast<State *>(pointer)->model));
}

// times retains the old ABI: first-token-ready, last-token-ready, native-finished.
// A null logits buffer disables capture. EOG, when enabled, is included in output.
int fv_query_ex(void * pointer, const int32_t * tokens, int token_count,
                int budget, int32_t * output, double * times,
                float * first_logits, int logits_capacity, int stop_at_eog,
                fv_query_info * info) {
    auto * state = static_cast<State *>(pointer);
    const int root = state ? state->position : -1;
    const double unset = std::numeric_limits<double>::quiet_NaN();
    if (times) std::fill_n(times, 3, unset);
    if (info) *info = {unset, 0, root, state && !state->needs_reset, 0};
    bool started = false;
    error.clear();
    try {
        ready(pointer);
        if (state->pending_root >= 0) throw std::runtime_error("frame append is still pending");
        if (!tokens || token_count <= 0 || budget <= 0 || !output || !times || !info)
            throw std::runtime_error("query requires nonempty tokens, positive budget and output buffers");
        const auto * vocab = llama_model_get_vocab(state->model);
        const int vocab_size = llama_vocab_n_tokens(vocab);
        if ((first_logits && logits_capacity < vocab_size) || (!first_logits && logits_capacity != 0))
            throw std::runtime_error("first logits buffer is smaller than vocabulary or missing");
        if (stop_at_eog != 0 && stop_at_eog != 1)
            throw std::runtime_error("stop_at_eog must be 0 or 1");
        // Reserve the full requested budget, even though the final output is not ingested.
        preflight(*state, int64_t(token_count) + budget);
        started = true;
        ingest(*state, tokens, nullptr, token_count, 0);
        info->prefill_finished_s = now();
        if (first_logits) {
            const float * logits = llama_get_logits_ith(state->context, -1);
            if (!logits) throw std::runtime_error("first-step logits are unavailable");
            std::copy_n(logits, vocab_size, first_logits);
            info->logits_captured = 1;
        }
        for (int index = 0; index < budget; ++index) {
            const llama_token token = llama_sampler_sample(state->sampler, state->context, -1);
            const double stamp = now();
            if (index == 0) times[0] = stamp;
            times[1] = stamp;
            output[index] = token;
            info->actual_count = index + 1;
            // llama_sampler_sample already accepts the selected CPU token.
            if (stop_at_eog && llama_vocab_is_eog(vocab, token)) break;
            // Fixed output budget, including EOS, with no discarded final step.
            if (index + 1 < budget) ingest(*state, &token, nullptr, 1, 0);
        }
        started = false;
        info->cleanup_ok = restore_root(*state, root);
        if (!info->cleanup_ok)
            throw std::runtime_error("failed to remove private answer suffix; reset required");
        times[2] = now();
        return root;
    } catch (const std::exception & exception) {
        error = exception.what();
        if (started) {
            info->cleanup_ok = restore_root(*state, root);
            if (!info->cleanup_ok) error += "; failed to remove private answer suffix; reset required";
        }
        if (times) times[2] = now();
        return -1;
    }
}

int fv_query(void * pointer, const int32_t * tokens, int token_count,
             int budget, int32_t * output, double * times) {
    fv_query_info info{};
    return fv_query_ex(pointer, tokens, token_count, budget, output, times, nullptr, 0, 0, &info);
}
}
