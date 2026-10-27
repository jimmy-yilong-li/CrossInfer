// Model-free dependencies for testing the real mac_fastvlm_ggml.cpp C ABI.
#include "ggml-backend.h"
#include "llama.h"
#include "ane_bridge.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>

struct llama_vocab {};
struct SamplerState { int step = 0; llama_sampler * child = nullptr; };
struct llama_model { llama_vocab vocab; };
struct llama_memory_i { int end = 0; };
struct llama_context {
    uint32_t capacity;
    uint32_t batch;
    llama_memory_i memory;
    float logits[4]{};
    bool pending = false;
};
struct ANESurfaceLeaseHandle {
    const void * base;
    size_t bytes;
    int fail_before;
    int fail_after;
};

namespace {
struct TestState {
    int model_limit = 32768;
    int fail_decode = 0;
    int fail_cleanup = 0;
    int fail_context = 0;
    int fail_sampler = 0;
    int null_logits = 0;
    int round_context = 0;
    int model_loads = 0;
    int live_models = 0;
    int live_contexts = 0;
    int live_samplers = 0;
    int decode_calls = 0;
    int cleanup_calls = 0;
    int logits_reads = 0;
    int sample_calls = 0;
    int surface_calls = 0;
    int chain_inits = 0;
    int chain_adds = 0;
    int accept_calls = 0;
    llama_context * context = nullptr;
} test;

int * field(const char * name) {
#define FIELD(key) if (std::strcmp(name, #key) == 0) return &test.key
    FIELD(model_limit); FIELD(fail_decode); FIELD(fail_cleanup);
    FIELD(fail_context); FIELD(fail_sampler); FIELD(null_logits); FIELD(round_context);
    FIELD(model_loads); FIELD(live_models); FIELD(live_contexts); FIELD(live_samplers);
    FIELD(decode_calls); FIELD(cleanup_calls); FIELD(logits_reads); FIELD(sample_calls);
    FIELD(surface_calls); FIELD(chain_inits); FIELD(chain_adds); FIELD(accept_calls);
#undef FIELD
    assert(false && "unknown test field");
    return nullptr;
}
} // namespace

extern "C" {
void fv_test_reset() {
    assert(!test.live_models && !test.live_contexts && !test.live_samplers);
    test = TestState{};
}
void fv_test_set(const char * name, int value) { *field(name) = value; }
int fv_test_get(const char * name) {
    if (std::strcmp(name, "kv_end") == 0) return test.context->memory.end;
    if (std::strcmp(name, "capacity") == 0) return test.context->capacity;
    if (std::strcmp(name, "pending") == 0) return test.context->pending;
    return *field(name);
}

void ggml_backend_load_all_from_path(const char *) {}
void llama_backend_init() {}
llama_model_params llama_model_default_params() { return {}; }
llama_context_params llama_context_default_params() { return {}; }
llama_model * llama_model_load_from_file(const char *, llama_model_params) {
    ++test.model_loads;
    ++test.live_models;
    return new llama_model;
}
void llama_model_free(llama_model * model) { --test.live_models; delete model; }
int32_t llama_model_n_ctx_train(const llama_model *) { return test.model_limit; }
int32_t llama_model_n_embd(const llama_model *) { return 3; }
const llama_vocab * llama_model_get_vocab(const llama_model * model) { return &model->vocab; }
int32_t llama_vocab_n_tokens(const llama_vocab *) { return 4; }
bool llama_vocab_is_eog(const llama_vocab *, llama_token token) { return token == 2; }
llama_context * llama_init_from_model(llama_model *, llama_context_params params) {
    if (test.fail_context) return nullptr;
    ++test.live_contexts;
    test.context = new llama_context{params.n_ctx + uint32_t(test.round_context), params.n_batch, {}};
    return test.context;
}
void llama_free(llama_context * context) {
    assert(!context->pending);
    --test.live_contexts;
    delete context;
    test.context = nullptr;
}
uint32_t llama_n_ctx(const llama_context * context) { return context->capacity; }
uint32_t llama_n_batch(const llama_context * context) { return context->batch; }
llama_memory_t llama_get_memory(const llama_context * context) {
    return const_cast<llama_memory_i *>(&context->memory);
}
void llama_memory_clear(llama_memory_t memory, bool) { memory->end = 0; }
bool llama_memory_seq_rm(llama_memory_t memory, llama_seq_id seq, llama_pos p0, llama_pos p1) {
    assert(!test.context->pending && seq == 0 && p1 == -1);
    ++test.cleanup_calls;
    if (test.fail_cleanup) return false;
    memory->end = p0;
    return true;
}
int32_t llama_decode(llama_context * context, llama_batch batch) {
    ++test.decode_calls;
    assert(batch.n_tokens > 0 && (batch.token || batch.embd));
    assert(batch.pos[0] == context->memory.end);
    context->pending = true;
    // A failed decode may still dirty KV, so rollback must include this batch.
    context->memory.end += batch.n_tokens;
    const float raw[] = {0.25f, 1.5f, -2.0f, 3.75f};
    std::copy_n(raw, 4, context->logits);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return test.decode_calls == test.fail_decode ? -1 : 0;
}
void llama_synchronize(llama_context * context) { context->pending = false; }
float * llama_get_logits_ith(llama_context * context, int32_t index) {
    assert(!context->pending && index == -1);
    ++test.logits_reads;
    return test.null_logits ? nullptr : context->logits;
}
llama_sampler * llama_sampler_init_greedy() {
    if (test.fail_sampler) return nullptr;
    ++test.live_samplers;
    return new llama_sampler{nullptr, new SamplerState};
}
llama_sampler_chain_params llama_sampler_chain_default_params() { return {}; }
llama_sampler * llama_sampler_chain_init(llama_sampler_chain_params) {
    if (test.fail_sampler) return nullptr;
    ++test.chain_inits;
    ++test.live_samplers;
    return new llama_sampler{nullptr, new SamplerState};
}
void llama_sampler_chain_add(llama_sampler * chain, llama_sampler * sampler) {
    ++test.chain_adds;
    static_cast<SamplerState *>(chain->ctx)->child = sampler;
}
void llama_sampler_free(llama_sampler * sampler) {
    auto * state = static_cast<SamplerState *>(sampler->ctx);
    if (state->child) llama_sampler_free(state->child);
    --test.live_samplers;
    delete state;
    delete sampler;
}
void llama_sampler_reset(llama_sampler * sampler) { static_cast<SamplerState *>(sampler->ctx)->step = 0; }
void llama_sampler_accept(llama_sampler *, llama_token) { ++test.accept_calls; }
llama_token llama_sampler_sample(llama_sampler * sampler, llama_context * context, int32_t index) {
    assert(!context->pending && index == -1);
    ++test.sample_calls;
    // Sampling deliberately overwrites logits: capture must happen first, once.
    std::fill_n(context->logits, 4, -999.0f);
    const int step = static_cast<SamplerState *>(sampler->ctx)->step++;
    const llama_token token = step == 0 ? 1 : step == 1 ? 2 : 3;
    llama_sampler_accept(sampler, token);
    return token;
}
int ane_bridge_consume_surface_lease(ANESurfaceLeaseHandle * lease,
        ane_bridge_output_consumer_fn callback, void * context, char * error, size_t capacity) {
    ++test.surface_calls;
    if (lease->fail_before) {
        std::snprintf(error, capacity, "surface lock failed");
        return 1;
    }
    const int status = callback(lease->base, lease->bytes, context, error, capacity);
    assert(!test.context->pending && "borrowed pointer escaped before synchronization");
    if (lease->fail_after) {
        std::snprintf(error, capacity, "surface unlock failed");
        return 1;
    }
    return status;
}
}
