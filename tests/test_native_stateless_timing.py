"""Behavioral timing tests without models, native runtime libraries, or accelerators.

Compile the production stateless source region and surface glue against a queued-
work mock and deterministic clock. Only the clock alias and Objective-C pool
markers are substituted; generation, ingestion, tracing, and callback bodies are
not reimplemented. This is not a full native build or a hardware timing test.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest


MOCKS = r"""
#include "edgeflow_llm/edgeflow_llm.h"
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "qwen25_prefill.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

struct MockClock {
    using time_point = std::chrono::steady_clock::time_point;
    inline static time_point current{};
    static time_point now() { return current; }
    static void advance(int ms) { current += std::chrono::milliseconds(ms); }
};
enum { LLM_ARCH_QWEN2VL };
struct llama_model {
    int arch = LLM_ARCH_QWEN2VL;
    ggml_tensor *tok_embd = nullptr;
};
struct llama_vocab {};
struct llama_context {
    int pending_ms = 0;
    int generated_decodes = 0;
    int eog_after = 10;
    int syncs = 0;
    uint32_t batch_capacity = 2048;
    int decode_calls = 0;
    int fail_decode_call = 0;
    int throw_decode_call = 0;
    std::array<float, 8> logits{};
};
llama_context *mapped_context = nullptr;
int pending_on_unmap = -1;

// These definitions make accidental model/backend initialization fail closed.
void llama_backend_init() { throw std::logic_error("backend init forbidden"); }
llama_model_params llama_model_default_params() { return {}; }
llama_context_params llama_context_default_params() { return {}; }
llama_model *llama_model_load_from_file(const char *, llama_model_params) {
    throw std::logic_error("model loading forbidden");
}
llama_context *llama_init_from_model(llama_model *, llama_context_params) {
    throw std::logic_error("native context init forbidden");
}
void llama_model_free(llama_model *model) { delete model; }
void llama_free(llama_context *ctx) { delete ctx; }
uint32_t llama_n_ctx(const llama_context *) { return 4096; }
uint32_t llama_n_batch(const llama_context *ctx) { return ctx->batch_capacity; }
int32_t llama_model_n_embd_inp(const llama_model *) { return 4096; }
const llama_vocab *llama_model_get_vocab(const llama_model *) {
    static llama_vocab vocab;
    return &vocab;
}
int32_t llama_vocab_n_tokens(const llama_vocab *) { return 8; }
bool llama_vocab_is_eog(const llama_vocab *, llama_token token) { return token == 0; }
const char *llama_model_chat_template(const llama_model *, const char *) {
    return "mock";
}
int32_t llama_chat_apply_template(const char *, const llama_chat_message *messages,
                                size_t count, bool, char *out, int32_t capacity) {
    const std::string rendered = "P" + std::string(messages[count - 1].content) + "S";
    if (out && capacity >= static_cast<int32_t>(rendered.size())) {
        std::memcpy(out, rendered.data(), rendered.size());
    }
    return static_cast<int32_t>(rendered.size());
}
int32_t llama_tokenize(const llama_vocab *, const char *, int32_t text_len,
                      llama_token *out, int32_t capacity, bool, bool) {
    if (text_len == 0) { return 0; }
    if (capacity == 0) { return -1; }
    out[0] = 7;
    return 1;
}
int32_t llama_token_to_piece(const llama_vocab *, llama_token token, char *out,
                            int32_t capacity, int32_t, bool) {
    if (capacity == 0) { return -1; }
    out[0] = static_cast<char>('A' + token - 1);
    return 1;
}
llama_memory_t llama_get_memory(const llama_context *) { return nullptr; }
void llama_memory_clear(llama_memory_t, bool) { MockClock::advance(7); }
void llama_synchronize(llama_context *ctx) {
    ++ctx->syncs;
    MockClock::advance(ctx->pending_ms);
    ctx->pending_ms = 0;
}
int32_t llama_decode(llama_context *ctx, llama_batch batch) {
    ++ctx->decode_calls;
    MockClock::advance(1); // Host submission, not execution completion.
    const bool prefill = batch.embd != nullptr || batch.token[0] == 7;
    ctx->pending_ms += prefill ? 100 : 20;
    if (!prefill) { ++ctx->generated_decodes; }
    if (ctx->decode_calls == ctx->throw_decode_call) { throw 42; }
    if (ctx->decode_calls == ctx->fail_decode_call) { return -2; }
    return 0;
}
float *llama_get_logits_ith(llama_context *ctx, int32_t) {
    llama_synchronize(ctx); // Matches the vendored public logits-read contract.
    MockClock::advance(3); // First-decision work must also count for EOG.
    ctx->logits.fill(-1.0f);
    const int token = ctx->generated_decodes >= ctx->eog_after
        ? 0 : ctx->generated_decodes + 1;
    ctx->logits.at(token) = 1.0f;
    return ctx->logits.data();
}
void ggml_fp16_to_fp32_row(const ggml_fp16_t *, float *, int64_t) {
    throw std::logic_error("FP16 conversion is outside this timing test");
}
size_t ggml_row_size(ggml_type, int64_t) {
    throw std::logic_error("text embedding extraction is outside this timing test");
}
const ggml_type_traits *ggml_get_type_traits(ggml_type) {
    throw std::logic_error("text embedding extraction is outside this timing test");
}
void ggml_backend_tensor_get(const ggml_tensor *, void *, size_t, size_t) {
    throw std::logic_error("text embedding extraction is outside this timing test");
}

// Mock the surface transport only; the production callback and dispatch run.
struct ANEKernelHandle { const void *data; size_t size; };
struct ANESurfaceLeaseHandle { const void *data; size_t size; };
using ane_bridge_output_consumer_fn = int (*)(const void *, size_t, void *, char *, size_t);
int ane_bridge_consume_output_surface(ANEKernelHandle *surface, int,
    ane_bridge_output_consumer_fn consume, void *ctx, char *error, size_t cap) {
    const int rc = consume(surface->data, surface->size, ctx, error, cap);
    pending_on_unmap = mapped_context->pending_ms;
    return rc;
}
int ane_bridge_consume_surface_lease(ANESurfaceLeaseHandle *surface,
    ane_bridge_output_consumer_fn consume, void *ctx, char *error, size_t cap) {
    const int rc = consume(surface->data, surface->size, ctx, error, cap);
    pending_on_unmap = mapped_context->pending_ms;
    return rc;
}
"""


MAIN = r"""
int main(int argc, char **argv) {
    if (argc != 3) { return 2; }
    const std::string path = argv[1];
    const std::string stop = argv[2];
    edgeflow_llm_ctx ctx;
    ctx.model.reset(new llama_model);
    ctx.context.reset(new llama_context);
    ctx.context->eog_after = (stop == "immediate-eog" || stop == "fixed-eog")
        ? 0 : stop == "early-eog" ? 2 : 10;
    std::vector<long long> deliveries;
    if (stop == "fixed-eog") {
        const int configured = edgeflow_llm_set_generation_options(&ctx, 1,
            [](int token, const char *piece, uint64_t size, void *data) {
                if (token == 0 || size != 1 || piece[0] != 'A') { return 1; }
                static_cast<std::vector<long long> *>(data)->push_back(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        MockClock::now().time_since_epoch()).count());
                return 0;
            }, &deliveries);
        if (configured != 0) { return 3; }
    }
    const bool text_only = path == "v1-text" || path == "v2-text";
    const bool llava = path.find("llava15") != std::string::npos;
    const bool fastvlm = path.find("fastvlm") != std::string::npos;
    const bool request_error = stop.find("-error") != std::string::npos;
    mapped_context = ctx.context.get();
    if (stop == "capacity-error") {
        // A legal prefix fits, but the visual batch exceeds n_batch.
        ctx.context->batch_capacity = 2;
    } else if (stop == "ingest-status-error") {
        ctx.context->fail_decode_call = text_only ? 1 : 2;
    } else if (stop == "ingest-exception-error") {
        ctx.context->throw_decode_call = text_only ? 1 : 2;
    } else if (stop == "generation-status-error") {
        ctx.context->fail_decode_call = (text_only ? 1 : 3) + 1;
    } else if (stop == "discarded-tail-status") {
        // The third emitted token is returned, but must never be decoded.
        ctx.context->fail_decode_call = (text_only ? 1 : 3) + 3;
    }
    std::vector<float> visual(576 * 4096, 0.0f);
    edgeflow_llm_request req{};
    req.prompt = stop == "null-prompt-error" ? nullptr : "question";
    req.visual_embd = text_only ? nullptr : visual.data();
    req.num_visual_tokens = llava ? 576 : 4;
    req.hidden_size = 4096;
    req.image_grid_thw[0] = 1;
    req.image_grid_thw[1] = req.image_grid_thw[2] = llava ? 24 : 4;
    req.max_new_tokens = stop == "max-tokens-error" ? 0
        : stop == "single-token-limit" ? 1 : 3;
    edgeflow_llm_request_v2 req2{};
    req2.prompt = req.prompt;
    req2.visual_embd = req.visual_embd;
    req2.num_visual_tokens = req.num_visual_tokens;
    req2.hidden_size = req.hidden_size;
    std::copy(req.image_grid_thw, req.image_grid_thw + 3, req2.image_grid_thw);
    req2.max_new_tokens = req.max_new_tokens;
    req2.model_family = llava ? EDGEFLOW_MODEL_FAMILY_LLAVA15
        : fastvlm ? EDGEFLOW_MODEL_FAMILY_FASTVLM : EDGEFLOW_MODEL_FAMILY_QWEN25VL;
    req2.visual_dtype = EDGEFLOW_DTYPE_FP32;
    ANEKernelHandle kernel{visual.data(), visual.size() * sizeof(float)};
    ANESurfaceLeaseHandle lease{visual.data(), visual.size() * sizeof(float)};
    edgeflow_llm_result result{};
    int rc;
    if (path == "surface-kernel") {
        req.visual_embd = nullptr;
        rc = edgeflow_llm_generate_from_ane_surface(&ctx, &req, &kernel, 0, &result);
    } else if (path == "surface-packed") {
        req.visual_embd = nullptr;
        rc = edgeflow_llm_generate_from_surface_lease_packed(&ctx, &req, &lease, 2, 2, &result);
    } else if (path == "surface-lease") {
        req.visual_embd = nullptr;
        rc = edgeflow_llm_generate_from_surface_lease(&ctx, &req, &lease, &result);
    } else if (path.find("surface-v2") == 0) {
        req2.visual_embd = nullptr;
        rc = edgeflow_llm_generate_from_surface_lease_v2(&ctx, &req2, &lease, &result);
    } else if (path.find("v2") == 0) {
        rc = edgeflow_llm_generate_v2(
            &ctx, stop == "null-request-error" ? nullptr : &req2, &result);
    } else {
        rc = edgeflow_llm_generate(
            &ctx, stop == "null-request-error" ? nullptr : &req, &result);
    }
    std::cout << "rc=" << rc << " prompt=" << result.prompt_eval_ms
        << " eval=" << result.eval_ms << " first=" << result.first_token_ms
        << " pending=" << ctx.context->pending_ms
        << " count=" << result.eval_count << " decodes=" << ctx.context->generated_decodes
        << " syncs=" << ctx.context->syncs
        << " calls=" << ctx.context->decode_calls << " unmap=" << pending_on_unmap
        << " total=" << std::chrono::duration_cast<std::chrono::milliseconds>(
            MockClock::now().time_since_epoch()).count()
        << " text=" << (result.text && result.text[0] ? result.text : "-");
    if (stop == "fixed-eog" && !deliveries.empty()) {
        std::cout << " cb_count=" << deliveries.size() << " cb_first=" << deliveries.front()
            << " cb_last=" << deliveries.back();
    }
    std::cout << '\n';
    edgeflow_llm_free_result(&result);
    if (request_error) {
        // Reuse the same context through the real ABI after removing the fault.
        ctx.context->batch_capacity = 2048;
        ctx.context->fail_decode_call = ctx.context->throw_decode_call = 0;
        ctx.context->generated_decodes = 0;
        req.prompt = req2.prompt = "next question";
        req.visual_embd = nullptr;
        req2.visual_embd = nullptr;
        req.max_new_tokens = req2.max_new_tokens = 1;
        const auto next_start = MockClock::now();
        const bool v2 = path.find("v2") == 0 || path.find("surface-v2") == 0;
        const int next_rc = v2 ? edgeflow_llm_generate_v2(&ctx, &req2, &result)
                               : edgeflow_llm_generate(&ctx, &req, &result);
        std::cout << "rc=" << next_rc << " prompt=" << result.prompt_eval_ms
            << " eval=" << result.eval_ms << " first=" << result.first_token_ms
            << " pending=" << ctx.context->pending_ms << " count=" << result.eval_count
            << " total=" << std::chrono::duration_cast<std::chrono::milliseconds>(
                MockClock::now() - next_start).count() << '\n';
        edgeflow_llm_free_result(&result);
        return next_rc;
    }
    return rc;
}
"""


@pytest.fixture(scope="module")
def timing_binary(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++17 compiler is required for the native timing test")
    runtime = Path(__file__).resolve().parents[1] / "tools/edgeflow_llm_runtime"
    source = (runtime / "src/edgeflow_llm.cpp").read_text()
    # Exclude initialization ABI exports and the unrelated stateful implementation.
    stateless = source[
        source.index("namespace {") : source.index(
            "namespace {\n\nthread_local std::string g_session_init_error;"
        )
    ]
    clock_alias = "using Clock = std::chrono::steady_clock;"
    assert stateless.count(clock_alias) == 1, "production clock seam changed"
    stateless = stateless.replace(clock_alias, "using Clock = MockClock;", 1)
    glue = (runtime / "src/edgeflow_llm_ane_glue.mm").read_text()
    # Pool lifetime is not part of this test; no Objective-C/ANE runtime is linked.
    glue = glue[glue.index("namespace {") :].replace("@autoreleasepool", "")
    glue = glue.split('static int append_session_from_surface_lease(', 1)[0]
    directory = tmp_path_factory.mktemp("native-stateless-timing")
    translation_unit = directory / "timing.cpp"
    translation_unit.write_text(MOCKS + stateless + glue + MAIN)
    executable = directory / "timing"
    compiled = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O0",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Wno-c++20-extensions",
            "-I", str(runtime / "include"),
            "-I", str(runtime / "src"),
            "-I", str(runtime / "llama_src/include"),
            "-I", str(runtime / "llama_src/ggml/include"),
            "-isystem", str(runtime / "llama_src/ggml/src"),
            str(translation_unit),
            "-o", str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return executable


@pytest.mark.parametrize("path", ["surface-lease", "surface-packed"])
def test_fixed_output_masks_eog_and_delivers_before_next_decode(timing_binary, path):
    run = subprocess.run([str(timing_binary), path, "fixed-eog"],
                         capture_output=True, text=True, timeout=10, check=False)
    assert run.returncode == 0, run.stdout + run.stderr
    result = dict(item.split("=", 1) for item in run.stdout.split())
    assert result["text"] == "AAA"
    assert int(result["count"]) == int(result["cb_count"]) == 3
    assert int(result["decodes"]) == 2
    assert int(result["cb_first"]) == 7 + float(result["prompt"]) + 3
    assert int(result["cb_last"]) - int(result["cb_first"]) == 48
    assert int(result["pending"]) == 0


@pytest.mark.parametrize("trace", [False, True], ids=["trace-off", "trace-on"])
@pytest.mark.parametrize(
    "stop", ["token-limit", "single-token-limit", "immediate-eog", "early-eog",
             "discarded-tail-status"]
)
@pytest.mark.parametrize(
    "path",
    [
        "v1-text", "v1-qwen", "v2-text", "v2-qwen", "v2-fastvlm", "v2-llava15",
        "surface-kernel", "surface-lease", "surface-packed", "surface-v2-qwen",
        "surface-v2-fastvlm", "surface-v2-llava15",
    ],
)
def test_native_stateless_completion_timing(timing_binary, path, stop, trace):
    env = dict(os.environ, EDGEFLOW_LLM_TRACE_DECODE="1" if trace else "0")
    run = subprocess.run(
        [str(timing_binary), path, stop],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    result = dict(item.split("=", 1) for item in run.stdout.split())
    emitted = {
        "token-limit": 3, "single-token-limit": 1, "immediate-eog": 0,
        "early-eog": 2, "discarded-tail-status": 3,
    }[stop]
    stopped_on_eog = stop.endswith("eog")
    decisions = emitted + stopped_on_eog
    continuations = emitted if stopped_on_eog else emitted - 1
    prefill_batches = 1 if path.endswith("text") or path == "surface-packed" else 3
    prompt_ms = 101 * prefill_batches
    eval_ms = 21 * continuations + 3 * decisions
    assert int(result["decodes"]) == continuations, result
    assert int(result["calls"]) == prefill_batches + continuations, result
    assert float(result["prompt"]) == prompt_ms, result
    assert float(result["eval"]) == eval_ms, result
    assert float(result["first"]) == prompt_ms + 3, result
    assert int(result["pending"]) == 0, result
    assert int(result["count"]) == emitted, result
    assert int(result["unmap"]) == (0 if path.startswith("surface") else -1), result
    assert result["text"] == ("ABC"[:emitted] or "-"), result
    # KV clearing is outside the existing reported timing boundary.
    assert int(result["total"]) == 7 + prompt_ms + eval_ms, result
    assert int(result["syncs"]) == 2 + decisions + (
        prefill_batches + continuations if trace else 0
    ), result
    records = re.findall(r"trace phase=(\w+).*?elapsed_ms=([\d.]+)", run.stderr)
    assert len(records) == (prefill_batches + continuations if trace else 0), run.stderr
    for phase, duration in records:
        assert float(duration) == (21 if phase == "decode_step" else 101), run.stderr


@pytest.mark.parametrize("trace", [False, True], ids=["trace-off", "trace-on"])
@pytest.mark.parametrize(
    "path,failure",
    [
        (path, failure)
        for path in (
            "v1-text", "v1-qwen", "v2-text", "v2-qwen", "v2-fastvlm", "v2-llava15",
            "surface-kernel", "surface-lease", "surface-v2-qwen",
            "surface-v2-fastvlm", "surface-v2-llava15",
        )
        for failure in (
            "capacity-error", "ingest-status-error", "ingest-exception-error",
            "generation-status-error",
        )
        if failure != "capacity-error" or not path.endswith("text")
    ],
)
def test_stateless_errors_drain_before_return_and_next_request(
    timing_binary, path, failure, trace,
):
    env = dict(os.environ, EDGEFLOW_LLM_TRACE_DECODE="1" if trace else "0")
    run = subprocess.run(
        [str(timing_binary), path, failure], env=env,
        capture_output=True, text=True, timeout=10,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    failed, following = [
        dict(item.split("=", 1) for item in line.split())
        for line in run.stdout.splitlines()
    ]
    assert int(failed["rc"]) == 7, failed
    assert int(failed["pending"]) == 0, failed
    assert int(failed["syncs"]) > 0, failed
    assert int(failed["unmap"]) == (0 if path.startswith("surface") else -1), failed
    assert failed["text"] == "-", failed
    for field in ("prompt", "eval", "first", "count"):
        assert float(failed[field]) == 0, failed
    prefill_batches = 1 if path.endswith("text") else 3
    if failure == "capacity-error":
        completed_ms, calls = 101, 1
    elif failure.startswith("ingest-"):
        calls = 1 if path.endswith("text") else 2
        completed_ms = 101 * calls
    else:
        calls = prefill_batches + 1
        completed_ms = 101 * prefill_batches + 24
    assert int(failed["calls"]) == calls, failed
    assert int(failed["total"]) == 7 + completed_ms, failed
    assert following == {
        "rc": "0", "prompt": "101", "eval": "3", "first": "104",
        "pending": "0", "count": "1", "total": "111",
    }, following


@pytest.mark.parametrize("path", ["v1-text", "v2-text"])
@pytest.mark.parametrize("failure,rc", [
    ("null-request-error", 1), ("null-prompt-error", 3), ("max-tokens-error", 4),
])
def test_stateless_preflight_errors_keep_existing_guards(timing_binary, path, failure, rc):
    run = subprocess.run(
        [str(timing_binary), path, failure],
        env=dict(os.environ, EDGEFLOW_LLM_TRACE_DECODE="0"),
        capture_output=True, text=True, timeout=10,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    failed, following = [
        dict(item.split("=", 1) for item in line.split())
        for line in run.stdout.splitlines()
    ]
    assert int(failed["rc"]) == rc, failed
    for field in ("pending", "syncs", "calls", "prompt", "eval", "first", "count", "total"):
        assert float(failed[field]) == 0, failed
    assert int(following["rc"]) == 0, following
