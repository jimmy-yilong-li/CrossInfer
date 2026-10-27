// tabm_decoder_worker.cpp: in-process MNN Decoder Adapter worker.
//
// A persistent worker that loads the MNN Llm ONCE and serves requests over a
// line protocol on stdin/stdout. Replaces the black-box `llm_demo` text pipe:
// exact token ids + exact per-phase timing (prefill_us/decode_us deltas from
// LlmContext), and an `embeds` mode that is the REAL cross-process consumer of
// the TABM B-layer shm slot (attach by descriptor -> one memcpy into MNN's
// input_embeds writeMap -> generate).
//
// Protocol (single in-flight; MNN pollutes stdout with logs, so consumers must
// PREFIX-FILTER lines: only "READY", "RES ", "VINFO ", "FATAL " are protocol
// lines; retained mode also uses "CTXCAP " and "CTXTIME "):
//   worker -> "READY <load_seconds>"
//   REQ <id> text   <max_tokens> <b64prompt>
//   REQ <id> embeds <max_tokens> <shm_name> <offset> <n_vis> <hidden>
//   REQ <id> vlm    <max_tokens> <shm_name> <offset> <n_vis> <hidden> <b64prompt>
//   TOK <id> <b64text>                   (prefix-registry identity op)
//   CTX <id> reset   <stream> <version> <b64_fixed_prefix>
//   CTX <id> append  <stream> <expected_version> <shm> <offset> <n_vis> <hidden>
//   CTX <id> query   <stream> <version> <max_tokens> <b64_private_suffix> <logits_stem|->
//   CTX <id> release <stream> <version>
// CTX requires EDGEFLOW_TABM_RETAINED=1 AND actual config precision="high",
// or explicit CPU/low fixed-scale or OpenCL/low quant-GEMM + native capability.
// Explicit EDGEFLOW_TABM_RETAINED_SEGMENTED_REPLAY=1 instead declares the
// same-backend per-frame reference for stock CPU/OpenCL low + greedy, chunk=0.
// Startup emits "CTXCAP per_frame_v1" before READY only in that mode.
// It replies with a bounded RES <id> CTX ... shape (adapter_protocol.py).
// Prefix + first frame are ingested together; each append advances version.
// Query completes first logits, decodes, then eraseHistory(root_end,0).
// Legacy low admission is unchanged; full-logit fidelity stays <=0.005.
// With EDGEFLOW_TABM_TRACE=1, every CTX emits before RES:
//   CTXTIME <id> <op> <request_start_s> <prefill_start_s> <prefill_end_s>
//           <first_token_ready_s> <finish_s>
// CLOCK_MONOTONIC seconds; -1 denotes only a missing phase. First token means
// the completed native step, not client-visible streaming text.
//   EXIT
//   worker -> "VINFO <id> nbytes=<n> crc32=<8-hex> crc_us=<us>" before RES
//             for successful visual-buffer attachment in embeds/vlm mode
//   worker -> "RES <id> <ok:0|1> <n_prompt> <n_decode> <prefill_us> <decode_us> <copied_bytes> <ids_csv|-> <err_b64|->"
//   worker -> "TINFO <id> n_pre=... n_vis=... n_post=... reset_us=...
//              prepare_us=... stage_us=... generate_us=... native_wall_us=..."
//             before successful VLM RES only when EDGEFLOW_TABM_TRACE=1.
//   worker -> "RES <id> TOK <token_count> <token_ids_hash>"   (TOK reply only; hash = FNV-1a
//                                                              64-bit over each token id's 4
//                                                              little-endian bytes, lowercase hex)
//
// TOK op: tokenizes `b64text` with the SAME
// tokenizer_encode() call the `text` REQ path uses (no chat template
// applied — TOK hashes the raw prefix's OWN tokens, not a templated
// variant), exactly once per prefix, so the Python PrefixRegistry can key on
// tokenized identity instead of raw text. A malformed TOK line gets this
// file's existing per-request error convention (respond() + ReqResult, the
// generic RES shape below) — NOT a process FATAL.
//
// vlm mode: the prompt must contain the literal placeholder "<image>".
// The worker applies the chat template, SPLITS the templated string on that
// placeholder (the tokenizer never needs to know an <image> token id),
// embeds the pre/post text segments via the model's own Llm::embedding(),
// and splices [pre | projected-vision-from-shm | post] into ONE input_embeds.
// The shm region must hold ALREADY-PROJECTED embeddings (LLM hidden dim, fp32).
//
// Copy accounting (honest): `embeds` mode performs exactly ONE payload copy here
// (shm slot -> MNN writeMap), reported as copied_bytes. With the producer's
// write into the slot (copy 1, counted by TABM), the cross-process B+C-host
// chain is TWO copies total. This is single-copy-per-hop, NOT zero-copy.
// `vlm` mode reports copied_bytes = ALL bytes staged into the input VARP
// (pre_text + vision + post_text); the vision payload portion is n_vis*hidden*4.
//
// Build: native/build.sh tabm_decoder_worker.cpp (mirrors llm_demo link:
// -fno-rtti -fno-exceptions; libllm + MNN_CL + MNN_Express + MNN).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <cstdint>
#include <string>
#include <sstream>
#include <vector>
#include <memory>
#include <time.h>
#include <iostream>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "llm/llm.hpp"
#ifdef CROSSINFER_QWEN_POSITIONS
#include "qwen_mnn_positions.hpp"
#endif
#include "first_token_clock.hpp"

using namespace MNN::Transformer;
using namespace MNN::Express;

static double now_s() {
    timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return static_cast<double>(value.tv_sec) + static_cast<double>(value.tv_nsec) / 1e9;
}

static long long elapsed_us(double start, double end) {
    return (long long)((end - start) * 1e6);
}

#ifdef CROSSINFER_RETAINED_KV
#include <stdexcept>
static double now_ms() { return now_s() * 1000.0; }
static void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
#include "mnn_handoff_state.hpp"
#include "handoff_snapshot_file.hpp"
struct RetainedTransfer {
    bool donor = false;
    std::string model_key;
    HandoffCapture capture;
    std::unique_ptr<HandoffImporter> importer;
    WorkerKVSnapshot imported;
    size_t completed_import_layers = 0;
    void clear() {
        capture.enabled = false;
        capture.snapshot = HandoffSnapshot();
        capture.seen.clear();
        if (importer) importer->snapshot = nullptr;
        imported = WorkerKVSnapshot();
        completed_import_layers = 0;
    }
    void complete_import() {
        if (importer && importer->snapshot) {
            importer->complete();
            completed_import_layers = importer->seen.size();
            imported = WorkerKVSnapshot();
        }
    }
};
#endif

// ---- base64 (decode only; prompt arrives b64 so the line protocol stays trivial) ----
static std::string b64_decode(const std::string& in) {
    static int T[256];
    static bool init = false;
    if (!init) {
        for (int i = 0; i < 256; i++) T[i] = -1;
        const char* tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; i++) T[(unsigned char)tbl[i]] = i;
        init = true;
    }
    std::string out;
    int val = 0, bits = -8;
    for (unsigned char c : in) {
        if (T[c] == -1) { if (c == '=') break; continue; }
        val = (val << 6) + T[c];
        bits += 6;
        if (bits >= 0) { out.push_back((char)((val >> bits) & 0xFF)); bits -= 8; }
    }
    return out;
}
static std::string b64_encode(const std::string& in) {
    const char* tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    int val = 0, bits = -6;
    for (unsigned char c : in) {
        val = (val << 8) + c;
        bits += 8;
        while (bits >= 0) { out.push_back(tbl[(val >> bits) & 0x3F]); bits -= 6; }
    }
    if (bits > -6) out.push_back(tbl[((val << 8) >> (bits + 8)) & 0x3F]);
    while (out.size() % 4) out.push_back('=');
    return out;
}

static long long to_ll(const std::string& s, bool* ok) {
    char* end = nullptr;
    errno = 0;
    long long v = strtoll(s.c_str(), &end, 10);
    *ok = (end && *end == '\0' && errno == 0 && !s.empty());
    return v;
}

struct ReqResult {
    bool ok = false;
    int n_prompt = 0;
    std::vector<int> ids;
    long long prefill_us = 0, decode_us = 0, copied_bytes = 0;
    long long visual_nbytes = 0, visual_crc_us = 0;
    uint32_t visual_crc32 = 0;
    std::string err;
    struct {
        bool enabled = false;
        long long n_pre = 0, n_vis = 0, n_post = 0;
        long long reset_us = 0, prepare_us = 0, stage_us = 0;
        long long generate_us = 0, native_wall_us = 0;
        long long first_token_mono_us = 0, native_ttft_us = 0;
    } trace;
};

static uint32_t crc32_bytes(const void* raw, size_t size) {
    static uint32_t table[256];
    static bool initialized = false;
    if (!initialized) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t value = i;
            for (int bit = 0; bit < 8; ++bit) {
                value = (value >> 1) ^ (0xedb88320U & (0U - (value & 1U)));
            }
            table[i] = value;
        }
        initialized = true;
    }
    const unsigned char* data = static_cast<const unsigned char*>(raw);
    uint32_t crc = 0xffffffffU;
    for (size_t i = 0; i < size; ++i) {
        crc = table[(crc ^ data[i]) & 0xffU] ^ (crc >> 8);
    }
    return crc ^ 0xffffffffU;
}

static void respond(long long id, const ReqResult& r) {
    std::string csv;
    if (r.ids.empty()) {
        csv = "-";
    } else {
        std::ostringstream os;
        for (size_t i = 0; i < r.ids.size(); i++) { if (i) os << ","; os << r.ids[i]; }
        csv = os.str();
    }
    if (r.visual_nbytes > 0) {
        printf("\nVINFO %lld nbytes=%lld crc32=%08x crc_us=%lld\n",
               id, r.visual_nbytes, r.visual_crc32, r.visual_crc_us);
    }
    if (r.trace.enabled) {
        printf("\nTINFO %lld n_pre=%lld n_vis=%lld n_post=%lld reset_us=%lld "
               "prepare_us=%lld stage_us=%lld generate_us=%lld native_wall_us=%lld",
               id, r.trace.n_pre, r.trace.n_vis, r.trace.n_post,
               r.trace.reset_us, r.trace.prepare_us, r.trace.stage_us,
               r.trace.generate_us, r.trace.native_wall_us);
        if (r.trace.first_token_mono_us > 0) {
            printf(" first_token_mono_us=%lld native_ttft_us=%lld",
                   r.trace.first_token_mono_us, r.trace.native_ttft_us);
        }
        printf("\n");
    }
    // Leading \n: MNN logs to stdout and may not end with a newline; starting
    // the protocol token on a fresh line keeps the consumer's prefix filter
    // sound even if a log line was left unterminated.
    printf("\nRES %lld %d %d %zu %lld %lld %lld %s %s\n",
           id, r.ok ? 1 : 0, r.n_prompt, r.ids.size(),
           r.prefill_us, r.decode_us, r.copied_bytes,
           csv.c_str(), r.err.empty() ? "-" : b64_encode(r.err).c_str());
    fflush(stdout);
}

#include "retained_context.hpp"

int main(int argc, const char* argv[]) {
    if (argc < 2) { printf("\nFATAL usage: %s config.json\n", argv[0]); return 2; }
#ifdef CROSSINFER_QWEN_POSITIONS
    if (argc != 2) {
        printf("\nFATAL usage: %s config.json\n", argv[0]); return 2;
    }
#endif
    const char* trace_flag = getenv("EDGEFLOW_TABM_TRACE");
    const bool trace_enabled = trace_flag && strcmp(trace_flag, "1") == 0;

    // Outlive llm: its context holds the borrowed stream pointer.
    FirstTokenClock first_token_clock;
    std::ostream first_token_stream(&first_token_clock);
#ifdef CROSSINFER_QWEN_POSITIONS
    std::unique_ptr<Llm> llm;
    try {
        llm.reset(new edgeflow_qcs6490::QwenMnnLlm(std::make_shared<LlmConfig>(std::string(argv[1]))));
    } catch (const std::exception& error) {
        printf("\nFATAL %s\n", error.what()); fflush(stdout); return 2;
    }
#else
    std::unique_ptr<Llm> llm(Llm::createLLM(argv[1]));
#endif
    RetainedRoot retained;
    const char* retained_flag = getenv("EDGEFLOW_TABM_RETAINED");
    retained.enabled = retained_flag && strcmp(retained_flag, "1") == 0;
    const char* streams_flag = getenv("EDGEFLOW_TABM_RETAINED_STREAMS");
    const bool shared_streams = streams_flag && strcmp(streams_flag, "2") == 0;
    if ((streams_flag && strcmp(streams_flag, "1") && !shared_streams) ||
        (shared_streams && !retained.enabled)) {
        printf("\nFATAL EDGEFLOW_TABM_RETAINED_STREAMS requires 1 or retained opt-in with 2\n");
        fflush(stdout); return 2;
    }
    std::string shared_model_name;
    if (shared_streams && (!llm || !retained_model_name(llm->dump_config(), shared_model_name))) {
        printf("\nFATAL shared streams require an unescaped relative original llm_model filename\n");
        fflush(stdout); return 2;
    }
    if (shared_streams && (retained_config_literal(llm->dump_config(), "is_visual", "true") ||
                           retained_config_literal(llm->dump_config(), "is_audio", "true") ||
                           retained_config_literal(llm->dump_config(), "has_talker", "true"))) {
        printf("\nFATAL shared streams require plain Llm projected-input configuration, not Omni\n");
        fflush(stdout); return 2;
    }
    const bool segmented_replay = retained_segmented_replay();
#ifdef CROSSINFER_QWEN_POSITIONS
    if (shared_streams) {
        printf("\nFATAL shared streams do not clone Qwen positional subclasses\n"); fflush(stdout); return 2;
    }
    if (!retained.enabled || !segmented_replay) {
        printf("\nFATAL Qwen positions target requires retained context and same-backend per_frame_v1 replay\n");
        fflush(stdout); return 2;
    }
#endif
    if (segmented_replay && !retained.enabled) {
        printf("\nFATAL segmented replay requires retained context\n"); fflush(stdout); return 2;
    }
    if (retained.enabled) {
        if (!llm.get() || !retained_precision_allowed(*llm)) {
            printf("\nFATAL retained context requires actual config precision=high or capability-validated CPU low fixed-scale / OpenCL low quant-GEMM or explicit per_frame_v1 replay (CPU/OpenCL low, greedy, chunk=0, empty chunk_limits); modes/policies must not conflict (gate <=0.005)\n");
            fflush(stdout); return 2;
        }
        if (!llm->set_config("{\"reuse_kv\":true}") || !llm->reuse_kv()) {
            printf("\nFATAL retained reuse_kv configuration failed\n"); fflush(stdout); return 2;
        }
        const char* directory = getenv("EDGEFLOW_TABM_RETAINED_DIAGNOSTICS_DIR");
        if (directory && directory[0]) {
            if (directory[0] != '/' || strlen(directory) > 4096) {
                printf("\nFATAL retained diagnostics require an absolute experiment directory\n");
                fflush(stdout); return 2;
            }
            retained.diagnostic_dir = open(directory, O_RDONLY | O_DIRECTORY | O_NOFOLLOW);
            if (retained.diagnostic_dir < 0) {
                printf("\nFATAL cannot open retained experiment directory\n"); fflush(stdout); return 2;
            }
        }
    }
#ifdef CROSSINFER_RETAINED_KV
    const char* transfer_flag = getenv("EDGEFLOW_TABM_RETAINED_KV_TRANSFER");
    const bool transfer_enabled = transfer_flag && strcmp(transfer_flag, "1") == 0;
    if (transfer_enabled) {
        const char* key = getenv("EDGEFLOW_TABM_KV_MODEL_KEY");
        if (!retained.enabled || !crossinferKVTransferEnabled() || !key || !retained_token(key, 128)) {
            printf("\nFATAL KV transfer needs explicit retained opt-in and model key\n"); fflush(stdout); return 2;
        }
        retained.transfer.reset(new RetainedTransfer());
        retained.transfer->model_key = key;
        retained.transfer->donor = retained_config_string(llm->dump_config(), "backend_type", "opencl");
        if (!retained.transfer->donor && !retained_config_string(llm->dump_config(), "backend_type", "cpu")) {
            printf("\nFATAL KV transfer supports only OpenCL donor or CPU receiver\n"); fflush(stdout); return 2;
        }
        llm->set_config("{\"enable_debug\":true}");
    }
#endif
    // Ensure the tmp_path dir EXISTS before load: with
    // kvcache_mmap=true a missing tmp_path dir => null mmap in
    // CPUKVCacheManager::onAlloc => SIGSEGV; load() never validates it.
    // EEXIST is fine; any other failure surfaces on first disk-KV use.
    mkdir("tmp", 0755);
    llm->set_config("{\"tmp_path\":\"tmp\"}");
    double tl = now_s();
    if (!llm->load()) { printf("\nFATAL load failed\n"); fflush(stdout); return 1; }
    // Destruction order keeps the base model alive until its shared child exits.
    std::unique_ptr<Llm> shared_llm;
    RetainedRoot shared_root;
    if (shared_streams) {
        shared_llm.reset(llm->create_lora(shared_model_name));
        if (!shared_llm || !shared_llm->reuse_kv() || !retained_precision_allowed(*shared_llm)) {
            printf("\nFATAL shared-weight Llm creation failed\n"); fflush(stdout); return 1;
        }
        shared_root.enabled = true;
        if (retained.diagnostic_dir >= 0) {
            shared_root.diagnostic_dir = dup(retained.diagnostic_dir);
            if (shared_root.diagnostic_dir < 0) {
                printf("\nFATAL cannot duplicate retained diagnostics directory\n"); fflush(stdout); return 2;
            }
        }
#ifdef CROSSINFER_RETAINED_KV
        if (retained.transfer) {
            shared_root.transfer.reset(new RetainedTransfer());
            shared_root.transfer->donor = retained.transfer->donor;
            shared_root.transfer->model_key = retained.transfer->model_key;
        }
#endif
    }
    Llm* retained_models[] = {llm.get(), shared_llm.get()};
    RetainedRoot* retained_roots[] = {&retained, &shared_root};
    const int retained_stream_count = shared_streams ? 2 : 1;
#ifdef CROSSINFER_QWEN_POSITIONS
    printf("\nCTXCAP qwen_image_sequence_v1\n");
#endif
#ifdef CROSSINFER_RETAINED_KV
    if (retained.transfer) {
        try {
            for (int i = 0; i < retained_stream_count; ++i) {
                auto& transfer = *retained_roots[i]->transfer;
                if (transfer.donor) transfer.capture.attach(*retained_models[i]);
                else {
                    transfer.importer.reset(new HandoffImporter());
                    transfer.importer->attach(*retained_models[i]);
#ifdef CROSSINFER_QWEN_POSITIONS
                    transfer.importer->history = [](void* value, int positions) {
                        auto* qwen = static_cast<edgeflow_qcs6490::QwenMnnLlm*>(static_cast<Llm*>(value));
                        return qwen->install_physical_history(positions) ? 1 : 0;
                    };
#endif
                }
            }
        } catch (const std::exception& error) {
            printf("\nFATAL %s\n", error.what()); fflush(stdout); return 2;
        }
        printf("\nCTXCAP kv_transfer_v1\n");
        printf("\nCTXCAP visual_prefix_v1\n");
    }
#endif
    if (segmented_replay) printf("\nCTXCAP per_frame_v1\n");
    if (shared_streams) printf("\nCTXCAP shared_weights_2_v1\n");
    printf("\nREADY %.2f\n", now_s() - tl);
    fflush(stdout);

    std::string line;
    bool oversized = false;
    while (retained.enabled ? retained_getline(line, oversized) : static_cast<bool>(std::getline(std::cin, line))) {
        if (line == "EXIT") break;
        std::istringstream is(line);
        std::string tag;
        is >> tag;
        if (tag == "CTX") {
            const double request_started = now_s();
            std::vector<std::string> fields;
            fields.push_back(tag);
            std::string field;
            while (is >> field) fields.push_back(field);
            bool ok_id = false;
            long long id = fields.size() > 1 ? to_ll(fields[1], &ok_id) : -1;
            RetainedReply reply;
            reply.request_started = request_started;
            RetainedRoot unbound;
            RetainedRoot* root = shared_streams ? &unbound : &retained;
            Llm* model = llm.get();
            std::string op = fields.size() > 2 ? fields[2] : "reset";
            if (!ok_id || fields.size() < 5 || oversized) {
                reply.result.err = "bad or oversized CTX request";
            } else {
                if (shared_streams) {
                    int slot = -1;
                    for (int i = 0; i < retained_stream_count; ++i) {
                        if (retained_roots[i]->active && retained_roots[i]->stream == fields[3]) slot = i;
                    }
                    const bool creates = op == "reset" || op == "import" || op == "cache_import";
                    if (slot < 0 && creates) {
                        // release clears identity/KV, not the persistent Llm or its shared weights.
                        for (int i = 0; i < retained_stream_count; ++i) {
                            if (!retained_roots[i]->active) { slot = i; break; }
                        }
                    }
                    if (slot >= 0) {
                        root = retained_roots[slot];
                        model = retained_models[slot];
                    } else {
                        reply.result.err = creates ? "retained stream capacity is 2; release a root first" :
                            "unknown retained stream; reset or import required";
                    }
                }
            }
            reply.root_before = root->end;
            if (reply.result.err.empty()) {
#if defined(CROSSINFER_RETAINED_KV) || defined(CROSSINFER_QWEN_POSITIONS)
                try { retained_run(*model, *root, fields, reply); }
                catch (const std::exception& error) {
                    root->valid = false;
                    reply.result.err = error.what();
                }
#else
                retained_run(*model, *root, fields, reply);
#endif
            }
            reply.finish_s = now_s();
            reply.request_wall_us = elapsed_us(request_started, reply.finish_s);
            retained_respond(ok_id ? id : -1, op, *root, reply, *model, trace_enabled, root != &unbound);
            continue;
        }
        if (oversized) {
            ReqResult r; r.err = "request exceeds retained worker line bound";
            respond(-1, r); continue;
        }
        if (tag == "TOK") {
            // TOK <id> <b64text> -> RES <id> TOK <token_count> <token_ids_hash>
            // (see header comment above). Reuses tokenizer_encode() exactly as
            // the `text` REQ path does. A malformed TOK line (unparsable id)
            // gets the file's existing per-request error convention below —
            // NOT a process FATAL.
            std::string s_id, b64;
            is >> s_id >> b64;
            bool ok_id = false;
            long long id = to_ll(s_id, &ok_id);
            if (!ok_id) {
                ReqResult r;
                r.err = "bad TOK header";
                respond(-1, r);
                continue;
            }
            std::string text = b64_decode(b64);
            std::vector<int> ids = llm->tokenizer_encode(text);
            unsigned long long h = 14695981039346656037ULL;  // FNV-1a 64 offset basis
            for (size_t i = 0; i < ids.size(); i++) {
                unsigned int tid = (unsigned int)ids[i];
                for (int b = 0; b < 4; b++) {
                    h ^= (unsigned char)((tid >> (8 * b)) & 0xFF);
                    h *= 1099511628211ULL;                   // FNV-1a 64 prime
                }
            }
            char hex[17];
            snprintf(hex, sizeof(hex), "%016llx", h);
            // Leading \n: same rationale as respond() below (MNN stdout log
            // pollution may leave a prior line unterminated).
            printf("\nRES %lld TOK %zu %s\n", id, ids.size(), hex);
            fflush(stdout);
            continue;
        }
        std::string mode;
        std::string s_id, s_max;
        is >> s_id >> mode >> s_max;
        if (tag != "REQ") continue;                 // ignore non-protocol lines
        bool ok1 = false, ok2 = false;
        long long id = to_ll(s_id, &ok1);
        long long max_tokens = to_ll(s_max, &ok2);
        ReqResult r;
        if (!ok1 || !ok2 || max_tokens <= 0 || max_tokens > (1 << 20)) {
            r.err = "bad REQ header";
            respond(ok1 ? id : -1, r);
            continue;
        }
#ifdef CROSSINFER_QWEN_POSITIONS
        r.err = "Qwen positions target accepts retained CTX only; image packets must use CTX append";
        respond(id, r); continue;
#endif
        if (retained.active || shared_root.active) {
            r.err = "retained root owns worker; CTX release required before ordinary REQ";
            respond(id, r); continue;
        }

        // fresh, independent request
        const bool trace_vlm = trace_enabled && mode == "vlm";
        const double trace_start = trace_vlm ? now_s() : 0.0;
        const long long first_token_start_us = trace_vlm ? first_token_monotonic_us() : 0;
        first_token_clock.reset();
        llm->reset();
        llm->generate_init(trace_vlm ? &first_token_stream : nullptr);
        const double reset_done = trace_vlm ? now_s() : 0.0;
        auto* ctx = llm->getContext();
        long long pre0 = ctx->prefill_us, dec0 = ctx->decode_us;

        if (mode == "text") {
            std::string b64;
            is >> b64;
            std::string prompt = b64_decode(b64);
            if (prompt.empty()) {
                r.err = "empty prompt";
                respond(id, r);
                continue;
            }
            std::string templated = llm->apply_chat_template(prompt);
            std::vector<int> in_ids = llm->tokenizer_encode(
                templated.empty() ? prompt : templated);
            r.n_prompt = (int)in_ids.size();
            r.ids = llm->generate(in_ids, (int)max_tokens);
            r.ok = !r.ids.empty();
            if (!r.ok) r.err = "generate returned no tokens";
        } else if (mode == "embeds") {
            std::string shm_name, s_off, s_nvis, s_hidden;
            is >> shm_name >> s_off >> s_nvis >> s_hidden;
            bool o1 = false, o2 = false, o3 = false;
            long long off = to_ll(s_off, &o1);
            long long n_vis = to_ll(s_nvis, &o2);
            long long hidden = to_ll(s_hidden, &o3);
            if (!o1 || !o2 || !o3 || off < 0 || n_vis <= 0 || hidden <= 0) {
                r.err = "bad embeds descriptor fields";
                respond(id, r);
                continue;
            }
            // Process-boundary input feeding a raw memcpy: bound the magnitudes
            // so the (int) casts below cannot truncate and n_vis*hidden*4 cannot
            // overflow. 2^20 is far beyond any real embedding dims.
            if (n_vis > (1 << 20) || hidden > (1 << 20)) {
                r.err = "descriptor dims too large";
                respond(id, r);
                continue;
            }
            if (shm_name.empty()) { r.err = "empty shm name"; respond(id, r); continue; }
            std::string posix = shm_name[0] == '/' ? shm_name : "/" + shm_name;
            long long nbytes = n_vis * hidden * 4;
            int fd = shm_open(posix.c_str(), O_RDONLY, 0);
            if (fd < 0) { r.err = "shm_open failed: " + posix; respond(id, r); continue; }
            struct stat st;
            if (fstat(fd, &st) != 0 || (long long)st.st_size < off + nbytes) {
                close(fd);
                r.err = "shm too small for descriptor region";
                respond(id, r);
                continue;
            }
            void* base = mmap(nullptr, (size_t)(off + nbytes), PROT_READ, MAP_SHARED, fd, 0);
            close(fd);
            if (base == MAP_FAILED) { r.err = "mmap failed"; respond(id, r); continue; }
            const float* src = (const float*)((const char*)base + off);

            double crc_start = now_s();
            r.visual_nbytes = nbytes;
            r.visual_crc32 = crc32_bytes(src, (size_t)nbytes);
            r.visual_crc_us = (long long)((now_s() - crc_start) * 1e6);

            VARP embeds = _Input({(int)n_vis, 1, (int)hidden}, NCHW);
            float* dst = embeds->writeMap<float>();
            if (!dst) {
                munmap(base, (size_t)(off + nbytes));
                r.err = "writeMap null";
                respond(id, r);
                continue;
            }
            memcpy(dst, src, (size_t)nbytes);       // the ONE adapter-side payload copy
            munmap(base, (size_t)(off + nbytes));
            r.copied_bytes = nbytes;
            r.n_prompt = (int)n_vis;
            r.ids = llm->generate(embeds, (int)max_tokens);
            r.ok = !r.ids.empty();
            if (!r.ok) r.err = "generate returned no tokens";
        } else if (mode == "vlm") {
            std::string shm_name, s_off, s_nvis, s_hidden, b64;
            is >> shm_name >> s_off >> s_nvis >> s_hidden >> b64;
            bool o1 = false, o2 = false, o3 = false;
            long long off = to_ll(s_off, &o1);
            long long n_vis = to_ll(s_nvis, &o2);
            long long hidden = to_ll(s_hidden, &o3);
            if (!o1 || !o2 || !o3 || off < 0 || n_vis <= 0 || hidden <= 0 ||
                n_vis > (1 << 20) || hidden > (1 << 20) || shm_name.empty()) {
                r.err = "bad vlm descriptor fields";
                respond(id, r);
                continue;
            }
            std::string prompt = b64_decode(b64);
            const std::string kPlaceholder = "<image>";
            if (prompt.find(kPlaceholder) == std::string::npos) {
                r.err = "vlm prompt lacks <image> placeholder";
                respond(id, r);
                continue;
            }
            std::string templated = llm->apply_chat_template(prompt);
            if (templated.empty()) templated = prompt;
            size_t pos = templated.find(kPlaceholder);
            if (pos == std::string::npos) {
                r.err = "chat template dropped the <image> placeholder";
                respond(id, r);
                continue;
            }
            // Single-image contract: exactly ONE placeholder (a second one would
            // silently become literal text in post_ids).
            if (templated.find(kPlaceholder, pos + kPlaceholder.size()) != std::string::npos) {
                r.err = "multiple <image> placeholders (single-image contract)";
                respond(id, r);
                continue;
            }
            // Split AROUND the placeholder; the tokenizer never sees "<image>".
            std::vector<int> pre_ids  = llm->tokenizer_encode(templated.substr(0, pos));
            std::vector<int> post_ids = llm->tokenizer_encode(templated.substr(pos + kPlaceholder.size()));
            VARP pre_emb = nullptr, post_emb = nullptr;
            if (!pre_ids.empty())  pre_emb  = llm->embedding(pre_ids);
            if (!post_ids.empty()) post_emb = llm->embedding(post_ids);
            if (!pre_emb.get() && !post_emb.get()) {
                r.err = "both text segments empty — cannot validate hidden";
                respond(id, r);
                continue;
            }
            // embedding() returns {seq,1,hidden}; guard the shape assumptions
            // before dereferencing dims (process-boundary-adjacent).
            if ((pre_emb.get()  && (!pre_emb->getInfo()  || pre_emb->getInfo()->dim.size()  < 3)) ||
                (post_emb.get() && (!post_emb->getInfo() || post_emb->getInfo()->dim.size() < 3))) {
                r.err = "embedding() returned unexpected shape";
                respond(id, r);
                continue;
            }
            if ((pre_emb.get()  && pre_emb->getInfo()->dim[2]  != (int)hidden) ||
                (post_emb.get() && post_emb->getInfo()->dim[2] != (int)hidden)) {
                r.err = "descriptor hidden != model hidden";
                respond(id, r);
                continue;
            }
            const double stage_start = trace_vlm ? now_s() : 0.0;
            std::string posix = shm_name[0] == '/' ? shm_name : "/" + shm_name;
            long long vbytes = n_vis * hidden * 4;
            int fd = shm_open(posix.c_str(), O_RDONLY, 0);
            if (fd < 0) { r.err = "shm_open failed: " + posix; respond(id, r); continue; }
            struct stat st;
            if (fstat(fd, &st) != 0 || (long long)st.st_size < off + vbytes) {
                close(fd);
                r.err = "shm too small for descriptor region";
                respond(id, r);
                continue;
            }
            void* base = mmap(nullptr, (size_t)(off + vbytes), PROT_READ, MAP_SHARED, fd, 0);
            close(fd);
            if (base == MAP_FAILED) { r.err = "mmap failed"; respond(id, r); continue; }
            const float* vsrc = (const float*)((const char*)base + off);

            double crc_start = now_s();
            r.visual_nbytes = vbytes;
            r.visual_crc32 = crc32_bytes(vsrc, (size_t)vbytes);
            r.visual_crc_us = (long long)((now_s() - crc_start) * 1e6);

            long long n_pre  = pre_emb.get()  ? pre_emb->getInfo()->dim[0]  : 0;
            long long n_post = post_emb.get() ? post_emb->getInfo()->dim[0] : 0;
            long long total = n_pre + n_vis + n_post;
            VARP embeds = _Input({(int)total, 1, (int)hidden}, NCHW);
            float* dst = embeds->writeMap<float>();
            if (!dst) {
                munmap(base, (size_t)(off + vbytes));
                r.err = "writeMap null";
                respond(id, r);
                continue;
            }
            long long hb = hidden;  // floats per token
            if (pre_emb.get()) {
                const float* p = pre_emb->readMap<float>();
                memcpy(dst, p, (size_t)(n_pre * hb * 4));
            }
            memcpy(dst + n_pre * hb, vsrc, (size_t)vbytes);
            if (post_emb.get()) {
                const float* p = post_emb->readMap<float>();
                memcpy(dst + (n_pre + n_vis) * hb, p, (size_t)(n_post * hb * 4));
            }
            munmap(base, (size_t)(off + vbytes));
            const double stage_done = trace_vlm ? now_s() : 0.0;
            r.copied_bytes = total * hb * 4;   // ALL staged bytes; vision payload = vbytes of it
            r.n_prompt = (int)total;
            const double generate_start = trace_vlm ? now_s() : 0.0;
            r.ids = llm->generate(embeds, (int)max_tokens);
            const double generate_done = trace_vlm ? now_s() : 0.0;
            r.ok = !r.ids.empty();
            if (!r.ok) r.err = "generate returned no tokens";
            if (trace_vlm && r.ok) {
                r.trace.enabled = true;
                r.trace.n_pre = n_pre;
                r.trace.n_vis = n_vis;
                r.trace.n_post = n_post;
                r.trace.reset_us = elapsed_us(trace_start, reset_done);
                r.trace.prepare_us = elapsed_us(reset_done, stage_start);
                r.trace.stage_us = elapsed_us(stage_start, stage_done);
                r.trace.generate_us = elapsed_us(generate_start, generate_done);
                r.trace.first_token_mono_us = first_token_clock.first_us;
                r.trace.native_ttft_us = first_token_clock.first_us > 0
                    ? first_token_clock.first_us - first_token_start_us : 0;
            }
        } else if (mode == "prefix") {
            // Cross-request prefix-cache request.
            // Wire: REQ <id> prefix <max> <name> <b64prefix>
            // <b64suffix>. Both segments are tokenized SEPARATELY and
            // UN-TEMPLATED: identity discipline matches the TOK op.
            // The prefix KV must be the KV of exactly the
            // ids its registry identity hashes. Requires kvcache_mmap=true
            // in the worker's config (setPrefixCacheFile itself rejects
            // configs without it) + the tmp dir created at startup (a
            // missing tmp_path dir causes a null-mmap segfault in
            // CPUKVCacheManager::onAlloc).
            std::string name, b64p, b64s;
            is >> name >> b64p >> b64s;
            std::string prefix_text = b64_decode(b64p);
            std::string suffix_text = b64_decode(b64s);
            if (name.empty() || prefix_text.empty() || suffix_text.empty()) {
                r.err = "bad prefix fields (need name, b64prefix, b64suffix)";
                respond(id, r);
                continue;
            }
            // Name is a FILENAME STEM handed to
            // MNNFilePathConcat under prefix_cache_path — reject path shapes
            // ('/' or "..") so a caller bypassing the host-side gate cannot
            // escape the cache directory.
            if (name.find('/') != std::string::npos ||
                name.find("..") != std::string::npos) {
                r.err = "prefix cache name must be a filename stem (no '/', no '..')";
                respond(id, r);
                continue;
            }
            // Tokenize + validate BEFORE arming: setPrefixCacheFile puts the
            // engine into a 2-call armed state (mCallIndex), and an error
            // `continue` after arming would leak that state into the NEXT
            // request (whether reset() clears mPrefixCacheMode is not a
            // documented contract — do not depend on it).
            std::vector<int> pre_ids = llm->tokenizer_encode(prefix_text);
            std::vector<int> suf_ids = llm->tokenizer_encode(suffix_text);
            if (pre_ids.empty() || suf_ids.empty()) {
                r.err = "prefix/suffix tokenized to empty";
                respond(id, r);
                continue;
            }
            bool hit = llm->setPrefixCacheFile(name, 0);
            long long c1p0 = ctx->prefill_us;
            llm->generate(pre_ids, 0);   // call 1: PendingWrite (miss) / passExecute skip (hit)
            long long c1_us = ctx->prefill_us - c1p0;
            long long c2p0 = ctx->prefill_us;
            r.ids = llm->generate(suf_ids, (int)max_tokens);   // call 2: continue / PendingRead
            long long c2_us = ctx->prefill_us - c2p0;
            r.n_prompt = (int)(pre_ids.size() + suf_ids.size());
            r.ok = !r.ids.empty();
            if (!r.ok) r.err = "generate returned no tokens";
            // PFXINFO: NON-"RES " info line — the generic decode parser's
            // prefix filter discards it as junk (backward-harmless); only the
            // adapter's prefix-aware scan consumes it. Leading '\n' = same
            // unterminated-line discipline as respond().
            printf("\nPFXINFO %lld hit=%d c1_prefill_us=%lld c2_prefill_us=%lld n_prefix=%zu n_suffix=%zu\n",
                   id, hit ? 1 : 0, c1_us, c2_us, pre_ids.size(), suf_ids.size());
            fflush(stdout);
        } else {
            r.err = "unknown mode: " + mode;
            respond(id, r);
            continue;
        }

        r.prefill_us = ctx->prefill_us - pre0;
        r.decode_us  = ctx->decode_us  - dec0;
        if (r.trace.enabled) r.trace.native_wall_us = elapsed_us(trace_start, now_s());
        respond(id, r);
    }
    return 0;
}
