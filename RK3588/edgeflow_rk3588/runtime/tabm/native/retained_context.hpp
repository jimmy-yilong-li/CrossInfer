// Serial, opt-in research root. No historical forks or scheduler.
// Included after the worker's protocol/timing helpers; C++11, no exceptions.
#include <algorithm>
#include <cmath>
#include <climits>
#include <map>
#include <dlfcn.h>

static const long long kRetainedPositions = 8192;
static const long long kRetainedBytes = 64LL * 1024 * 1024;
static const size_t kRetainedLineBytes = 65536;

static bool retained_token(const std::string& s, size_t maximum) {
    if (s.empty() || s.size() > maximum) return false;
    for (char c : s) {
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '_' || c == '-')) return false;
    }
    return true;
}

// dump_config() is already valid JSON. Inspect only its top-level settings,
// not a nested backend field or the same text inside a chat template.
static size_t retained_config_value(const std::string& config, const std::string& key) {
    int depth = 0;
    for (size_t i = 0; i < config.size(); ++i) {
        char c = config[i];
        if (c == '{' || c == '[') ++depth;
        else if (c == '}' || c == ']') --depth;
        else if (c == '"') {
            size_t start = i++;
            while (i < config.size() && config[i] != '"') {
                if (config[i] == '\\') ++i;
                ++i;
            }
            if (depth != 1 || config.substr(start, i - start + 1) != "\"" + key + "\"") continue;
            size_t next = config.find_first_not_of(" \t\r\n", i + 1);
            if (next == std::string::npos || config[next] != ':') continue;
            next = config.find_first_not_of(" \t\r\n", next + 1);
            return next;
        }
    }
    return std::string::npos;
}

static bool retained_config_literal(const std::string& config, const std::string& key, const std::string& value) {
    size_t next = retained_config_value(config, key);
    if (next == std::string::npos) return false;
    if (value == "[]") {
        if (config[next] != '[') return false;
        next = config.find_first_not_of(" \t\r\n", next + 1);
        if (next == std::string::npos || config[next] != ']') return false;
        ++next;
    } else {
        if (config.compare(next, value.size(), value) != 0) return false;
        next += value.size();
    }
    next = config.find_first_not_of(" \t\r\n", next);
    return next != std::string::npos && (config[next] == ',' || config[next] == '}');
}

static bool retained_config_string(const std::string& config, const std::string& key, const std::string& value) {
    return retained_config_literal(config, key, "\"" + value + "\"");
}

static bool retained_model_name(const std::string& config, std::string& name) {
    const size_t start = retained_config_value(config, "llm_model");
    name = "llm.mnn";
    if (start != std::string::npos) {
        if (config[start] != '"') return false;
        const size_t end = config.find('"', start + 1);
        if (end == std::string::npos) return false;
        name = config.substr(start + 1, end - start - 1);
    }
    // create_lora interpolates this into JSON; LlmConfig prepends base_dir.
    if (name.empty() || name.size() > 4096 || name[0] == '/') return false;
    for (unsigned char c : name) {
        if (c < 32 || c == '"' || c == '\\') return false;
    }
    return start == std::string::npos || retained_config_string(config, "llm_model", name);
}

static bool retained_segmented_replay() {
    const char* flag = getenv("EDGEFLOW_TABM_RETAINED_SEGMENTED_REPLAY");
    return flag && strcmp(flag, "1") == 0;
}

static bool retained_precision_allowed(Llm& llm) {
#ifdef CROSSINFER_QWEN_POSITIONS
    if (!retained_segmented_replay()) return false;
#endif
    const std::string config = llm.dump_config();
    // VARP ingestion does not provide a canonical token history to penalty
    // samplers. Admit only history-independent greedy sampling in this slice.
    if (!retained_config_string(config, "sampler_type", "greedy")) return false;
    typedef int (*Capability)();
    const char* quant_flag = getenv("EDGEFLOW_TABM_RETAINED_OPENCL_QUANT_GEMM");
    const char* flag = getenv("EDGEFLOW_TABM_RETAINED_CPU_FIXED_SCALE");
    if (retained_segmented_replay()) {
        if ((flag && strcmp(flag, "1") == 0) || (quant_flag && strcmp(quant_flag, "1") == 0) ||
            getenv("MNN_EDGEFLOW_FIXED_QUERY_SCALE") || getenv("MNN_CROSSINFER_PREFILL_LINEAR_POLICY")) return false;
        // A declared same-backend, per-frame replay reference, NOT a promise
        // of stock-low equality to a monolithic full-input prefill.
        return (retained_config_string(config, "backend_type", "cpu") ||
                retained_config_string(config, "backend_type", "opencl")) &&
               retained_config_string(config, "precision", "low") &&
               retained_config_literal(config, "chunk", "0") &&
               retained_config_literal(config, "chunk_limits", "[]");
    }
    if (quant_flag && strcmp(quant_flag, "1") == 0) {
        const char* policy = getenv("MNN_CROSSINFER_PREFILL_LINEAR_POLICY");
        if (!policy || strcmp(policy, "quant_gemm") ||
            !retained_config_string(config, "backend_type", "opencl") ||
            !retained_config_string(config, "precision", "low")) return false;
        Capability capability = reinterpret_cast<Capability>(
            dlsym(RTLD_DEFAULT, "mnn_crossinfer_opencl_prefill_quant_gemm_v1"));
        return capability && capability() == 1;
    }
    if (retained_config_string(config, "precision", "high")) return true;
    const char* scale = getenv("MNN_EDGEFLOW_FIXED_QUERY_SCALE");
    if (!flag || strcmp(flag, "1") || !scale || strcmp(scale, "1") ||
        !retained_config_string(config, "backend_type", "cpu") ||
        !retained_config_string(config, "precision", "low")) return false;
    Capability capability = reinterpret_cast<Capability>(
        dlsym(RTLD_DEFAULT, "mnn_crossinfer_cpu_fixed_query_scale_v1"));
    return capability && capability() == 1;
}

static bool retained_text(const std::string& encoded, std::string& text) {
    if (encoded.empty() || encoded.size() > 43692 || encoded.size() % 4) return false;
    // The existing permissive decoder is deliberately not used permissively
    // at this endpoint: a canonical round trip rejects junk/padding errors.
    text = b64_decode(encoded);
    return !text.empty() && text.size() <= 32768 && text.find('\0') == std::string::npos &&
           b64_encode(text) == encoded;
}

struct RetainedRoot {
    bool enabled = false, active = false, valid = false;
    std::string stream;
    long long version = 0;
    size_t end = 0, physical = 0;
    int hidden = 0;
    std::vector<int> prefix;
    std::map<long long, size_t> version_ends;
    size_t requests = 0, diagnostic_bytes = 0;
    int diagnostic_dir = -1;
#ifdef CROSSINFER_QWEN_POSITIONS
    edgeflow_qcs6490::QwenPositions::State qwen_positions = {0, 0};
    std::map<long long, int64_t> version_cursors;
#endif
#ifdef CROSSINFER_RETAINED_KV
    std::unique_ptr<RetainedTransfer> transfer;
#endif
    ~RetainedRoot() { if (diagnostic_dir >= 0) close(diagnostic_dir); }
};

struct RetainedReply {
    ReqResult result;
    double request_started = 0.0;
    double prefill_start_s = -1.0, prefill_end_s = -1.0;
    double first_token_ready_s = -1.0, finish_s = -1.0;
    size_t root_before = 0, first_logits_count = 0;
    size_t exported_bytes = 0;
    bool rollback_verified = false;
    long long reset_us = 0, prefill_wall_us = 0, decode_wall_us = 0;
    long long ttft_us = -1, request_wall_us = 0;
    std::string first_logits_file;
};

struct RetainedMapping {
    void* base = MAP_FAILED;
    size_t length = 0;
    const float* data = nullptr;
    ~RetainedMapping() { if (base != MAP_FAILED) munmap(base, length); }
    bool attach(const std::string& name, long long offset, long long bytes, std::string& err) {
        std::string posix = name[0] == '/' ? name : "/" + name;
        int fd = shm_open(posix.c_str(), O_RDONLY, 0);
        if (fd < 0) { err = "retained shm_open failed"; return false; }
        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size < offset || bytes > st.st_size - offset) {
            close(fd); err = "retained shm too small for descriptor region"; return false;
        }
        long page = sysconf(_SC_PAGESIZE);
        if (page <= 0) { close(fd); err = "invalid mmap page size"; return false; }
        long long aligned = offset - offset % page;
        length = static_cast<size_t>(offset - aligned + bytes);
        base = mmap(nullptr, length, PROT_READ, MAP_SHARED, fd, static_cast<off_t>(aligned));
        close(fd);
        if (base == MAP_FAILED) { err = "retained mmap failed"; return false; }
        data = reinterpret_cast<const float*>(static_cast<const char*>(base) + offset - aligned);
        return true;
    }
};

struct RetainedDump {
    int dir = -1, fd = -1;
    std::string name;
    bool committed = false;
    ~RetainedDump() {
        if (fd >= 0) close(fd);
        if (!committed && !name.empty()) unlinkat(dir, name.c_str(), 0);
    }
    bool open_file(int directory, const std::string& stem, std::string& err) {
        if (directory < 0) { err = "retained diagnostics directory is not configured"; return false; }
        dir = directory;
        std::string candidate = stem + ".f32";
        fd = openat(dir, candidate.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
        if (fd < 0) { err = "cannot create retained logits file (no overwrite/symlinks)"; return false; }
        name = candidate;
        return true;
    }
    bool write_floats(const float* values, size_t count, std::string& err) {
        const char* p = reinterpret_cast<const char*>(values);
        size_t remaining = count * sizeof(float);
        while (remaining) {
            ssize_t n = write(fd, p, remaining);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) { err = "retained logits write failed"; return false; }
            p += n; remaining -= static_cast<size_t>(n);
        }
        int old_fd = fd;
        fd = -1;
        if (close(old_fd) != 0) { err = "retained logits close failed"; return false; }
        return true;
    }
};

static bool retained_logits(Llm& llm, const float*& values, size_t& count, std::string& err) {
    auto outputs = llm.getOutputs();
    if (outputs.empty() || !outputs[0].get() || !outputs[0]->getInfo() ||
        outputs[0]->getInfo()->size <= 0 || outputs[0]->getInfo()->size > 1000000) {
        err = "missing or unbounded retained logits"; return false;
    }
    count = static_cast<size_t>(outputs[0]->getInfo()->size);
    values = outputs[0]->readMap<float>(); // Complete deferred device work.
    if (!values) { err = "retained logits readMap failed"; return false; }
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i])) { err = "nonfinite retained logits"; return false; }
    }
    return true;
}

static VARP retained_embedding(Llm& llm, const std::vector<int>& ids, int hidden, std::string& err) {
    VARP emb = llm.embedding(ids);
    if (!emb.get() || !emb->getInfo() || emb->getInfo()->dim.size() != 3 ||
        emb->getInfo()->dim[0] != static_cast<int>(ids.size()) ||
        emb->getInfo()->dim[1] != 1 || emb->getInfo()->dim[2] != hidden) {
        err = "retained descriptor hidden != model hidden or invalid embedding shape";
        return nullptr;
    }
    return emb;
}

static void retained_run(Llm& llm, RetainedRoot& root, const std::vector<std::string>& fields,
                         RetainedReply& reply) {
    ReqResult& r = reply.result;
#ifdef CROSSINFER_QWEN_POSITIONS
    auto& qwen = static_cast<edgeflow_qcs6490::QwenMnnLlm&>(llm);
#endif
    const std::string& op = fields[2];
    bool integer = false;
    long long version = to_ll(fields[4], &integer);
    if (!root.enabled) { r.err = "retained context is not enabled"; return; }
    if (root.requests >= 1024) { r.err = "retained request bound exhausted; recreate worker"; return; }
    ++root.requests; // Rejections also consume the finite process-lifetime budget.
    if (!retained_token(fields[3], 128) || fields[3] == "-" || !integer || version < 0) {
        r.err = "invalid retained stream/version"; return;
    }
    if (!retained_precision_allowed(llm) || !llm.reuse_kv()) {
        r.err = "retained context requires actual high or capability-validated CPU low fixed-scale / OpenCL low quant-GEMM or explicit per_frame_v1 replay (CPU/OpenCL low, chunk=0, empty chunk_limits), and reuse_kv=true"; return;
    }
    // Ownership precedes text embedding, shm_open, generate_init and generation.
    if (op != "reset" && op != "import" && op != "cache_import" &&
        (!root.active || fields[3] != root.stream || version != root.version)) {
        r.err = "wrong retained stream/version (only current root exists)"; return;
    }
#ifdef CROSSINFER_RETAINED_KV
    if (op == "export" || op == "import" || op == "cache_import") {
        const bool cached = op == "cache_import";
        const bool historical = op == "export" && fields.size() == 9;
        if (!root.transfer || (!historical && fields.size() != (cached ? 10 : 7))) {
            r.err = "KV transfer is not enabled"; return;
        }
        if (fields[6].size() != 64 || fields[6].find_first_not_of("0123456789abcdef") != std::string::npos) {
            r.err = "invalid KV snapshot content digest"; return;
        }
        const std::string path = b64_decode(fields[5]);
        if (path.empty() || path[0] != '/' || path.find('\0') != std::string::npos ||
            path.size() > 4096 || b64_encode(path) != fields[5]) { r.err = "invalid KV snapshot path"; return; }
        auto& transfer = *root.transfer;
        if (op == "export") {
            if (!transfer.donor || !root.valid || !root.end ||
                transfer.capture.snapshot.positions != root.end) { r.err = "no completed donor KV root"; return; }
            WorkerKVSnapshot file;
            file.stream = root.stream; file.version = root.version;
            size_t positions = root.end;
            if (historical) {
                bool version_ok = false, positions_ok = false;
                const auto target = to_ll(fields[7], &version_ok);
                const auto end = to_ll(fields[8], &positions_ok);
                const auto boundary = root.version_ends.find(target);
                if (!version_ok || !positions_ok || target <= 0 || target > root.version ||
                    end <= 0 || boundary == root.version_ends.end() ||
                    boundary->second != static_cast<size_t>(end)) {
                    r.err = "KV export must end at a completed version boundary"; return;
                }
                file.version = target;
                positions = static_cast<size_t>(end);
            }
            file.hidden = root.hidden; file.prefix = root.prefix; file.model_key = transfer.model_key;
            file.content_digest = fields[6];
#ifdef CROSSINFER_QWEN_POSITIONS
            file.qwen_next_position = historical ? root.version_cursors.at(file.version)
                                                : root.qwen_positions.next_position;
# endif
            write_worker_kv_snapshot(path, file, transfer.capture.snapshot, positions);
            reply.exported_bytes = transfer.capture.snapshot.bytes() / root.end * positions;
        } else {
            if (transfer.donor) { r.err = "KV import requires CPU receiver"; return; }
            auto file = read_worker_kv_snapshot(path);
            const std::string& source_stream = fields[cached ? 7 : 3];
            const std::string& source_digest = fields[cached ? 9 : 6];
            bool source_version_ok = !cached;
            const long long source_version = cached ? to_ll(fields[8], &source_version_ok) : version;
            if (!source_version_ok || source_version != version ||
                file.model_key != transfer.model_key || file.content_digest != source_digest ||
                file.stream != source_stream || file.version != source_version ||
                (root.active && root.prefix != file.prefix)) { r.err = "KV model/stream/version/prefix mismatch"; return; }
            transfer.clear();
            transfer.imported = std::move(file);
            transfer.importer->begin(llm, transfer.imported.state);
            root.stream = fields[3]; root.version = version;
            root.end = transfer.imported.state.positions; root.physical = llm.getCurrentHistory();
            root.prefix = transfer.imported.prefix; root.hidden = transfer.imported.hidden;
            root.version_ends.clear();
            root.version_ends[root.version] = root.end;
#ifdef CROSSINFER_QWEN_POSITIONS
            root.qwen_positions = {static_cast<int64_t>(root.end), transfer.imported.qwen_next_position};
            qwen.restore_positions(root.qwen_positions);
            root.version_cursors.clear();
            root.version_cursors[root.version] = root.qwen_positions.next_position;
# endif
            root.active = root.valid = true;
        }
        r.ok = true;
        return;
    }
#endif
    if (op == "reset" || op == "release") {
        std::vector<int> prefix;
        if (op == "reset") {
            std::string text;
            if (fields.size() != 6 || !retained_text(fields[5], text)) {
                r.err = "invalid retained fixed prefix"; return;
            }
            prefix = llm.tokenizer_encode(text);
            if (prefix.empty() || prefix.size() > kRetainedPositions) {
                r.err = "retained prefix token count exceeds bounds"; return;
            }
#ifdef CROSSINFER_QWEN_POSITIONS
            if (std::find(prefix.begin(), prefix.end(), static_cast<int>(edgeflow_qcs6490::QwenMnnLlm::kVisionStartId)) != prefix.end()) {
                r.err = "Qwen fixed prefix must exclude vision_start; marker belongs to the 228-row packet"; return;
            }
#endif
        } else if (fields.size() != 5) { r.err = "invalid retained release fields"; return; }
        double started = now_s();
        llm.reset();
        llm.generate_init();
#ifdef CROSSINFER_RETAINED_KV
        if (root.transfer) root.transfer->clear();
#endif
        reply.reset_us = elapsed_us(started, now_s());
        root.active = op == "reset";
        root.valid = llm.getContext()->all_seq_len == 0 && llm.getContext()->gen_seq_len == 0 &&
                     llm.getContext()->output_tokens.empty();
        root.stream = root.active ? fields[3] : "";
        root.version = root.active ? version : 0;
        root.prefix.swap(prefix);
        root.end = 0;
        root.version_ends.clear();
#ifdef CROSSINFER_QWEN_POSITIONS
        root.qwen_positions = qwen.snapshot_positions();
        root.version_cursors.clear();
#endif
        root.physical = llm.getCurrentHistory(); // reset removal is also deferred.
        root.hidden = 0;
        r.ok = root.valid;
        if (!r.ok) r.err = "retained reset failed";
        return;
    }
    if (op != "append" && op != "query") { r.err = "unknown retained operation"; return; }
    if (!root.valid || llm.getContext()->all_seq_len != static_cast<int>(root.end) ||
        llm.getCurrentHistory() != root.physical || root.physical < root.end) {
        root.valid = false; r.err = "retained root is invalid; reset required"; return;
    }
#ifdef CROSSINFER_QWEN_POSITIONS
    const auto qwen_state = qwen.snapshot_positions();
    if (qwen_state.physical_tokens != static_cast<int64_t>(root.end) ||
        qwen_state.next_position != root.qwen_positions.next_position) {
        root.valid = false; r.err = "Qwen retained semantic root is invalid; reset required"; return;
    }
#endif
    // Borrowed input must outlive the completed forward/logit read, not just setup.
    RetainedMapping mapped;
    VARP input = nullptr;
    long long max_tokens = 0;
    RetainedDump dump;
    if (op == "append") {
        bool a = false, b = false, c = false;
        long long offset = fields.size() == 9 ? to_ll(fields[6], &a) : -1;
        long long n_vis = fields.size() == 9 ? to_ll(fields[7], &b) : -1;
        long long hidden = fields.size() == 9 ? to_ll(fields[8], &c) : -1;
        std::string name = fields.size() == 9 ? fields[5] : "";
        std::string stem = !name.empty() && name[0] == '/' ? name.substr(1) : name;
        if (!a || !b || !c || offset < 0 || offset % 4 || n_vis <= 0 || n_vis > kRetainedPositions ||
            hidden <= 0 || hidden > 16384 || stem.empty() || stem.size() > 240 ||
            stem == "." || stem == ".." ||
            stem.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-") != std::string::npos ||
            version == LLONG_MAX) {
            r.err = "invalid retained shm descriptor/version"; return;
        }
#ifdef CROSSINFER_QWEN_POSITIONS
        if (n_vis != edgeflow_qcs6490::QwenMnnLlm::kFrameRows) {
            r.err = "Qwen retained append requires 228 rows: vision_start + 225 features + vision_end + newline"; return;
        }
#endif
        long long bytes = n_vis * hidden * sizeof(float);
        size_t n_pre = root.end == 0 ? root.prefix.size() : 0;
        long long total = static_cast<long long>(n_pre) + n_vis;
        if (bytes > kRetainedBytes || offset > LLONG_MAX - bytes ||
            root.end + total > kRetainedPositions || total * hidden * sizeof(float) > kRetainedBytes ||
            (root.hidden && hidden != root.hidden)) {
            r.err = "retained append exceeds root/byte bound or hidden mismatch"; return;
        }
        // Validate model geometry even on the first frame, before attaching SHM.
        VARP pre = nullptr;
        if (n_pre) {
            pre = retained_embedding(llm, root.prefix, static_cast<int>(hidden), r.err);
            if (!pre.get()) return;
        }
        if (!mapped.attach(name, offset, bytes, r.err)) return;
        double crc_started = now_s();
        r.visual_nbytes = bytes;
        r.visual_crc32 = crc32_bytes(mapped.data, static_cast<size_t>(bytes));
        r.visual_crc_us = elapsed_us(crc_started, now_s());
        const char* borrow = getenv("EDGEFLOW_TABM_RETAINED_BORROW_INPUT");
        if (!n_pre && borrow && strcmp(borrow, "1") == 0) {
            Variable::Info info;
            info.dim = {static_cast<int>(total), 1, static_cast<int>(hidden)};
            info.order = NCHW;
            info.type = halide_type_of<float>();
            input = Variable::create(Expr::create(std::move(info), mapped.data, VARP::INPUT, Expr::REF));
            if (!input.get() || input->readMap<float>() != mapped.data) {
                r.err = "retained borrowed input binding failed"; return;
            }
            r.copied_bytes = 0;
        } else {
            // The first frame includes the text prefix in the same prefill partition.
            input = _Input({static_cast<int>(total), 1, static_cast<int>(hidden)}, NCHW);
            float* target = input.get() ? input->writeMap<float>() : nullptr;
            const float* text = n_pre ? pre->readMap<float>() : nullptr;
            if (!target || (n_pre && !text)) { r.err = "retained input/text map failed"; return; }
            if (n_pre) memcpy(target, text, n_pre * hidden * sizeof(float));
            memcpy(target + n_pre * hidden, mapped.data, static_cast<size_t>(bytes));
            r.copied_bytes = total * hidden * sizeof(float);
        }
        r.n_prompt = static_cast<int>(total);
    } else {
        std::string text;
        max_tokens = fields.size() == 8 ? to_ll(fields[5], &integer) : 0;
        if (fields.size() != 8 || !integer || max_tokens <= 0 || max_tokens > 128 ||
            !retained_text(fields[6], text) ||
            (fields[7] != "-" && !retained_token(fields[7], 64)) || !root.hidden) {
            r.err = "invalid retained query or no ingested frame"; return;
        }
        auto ids = llm.tokenizer_encode(text);
        if (ids.empty() || root.end + ids.size() + max_tokens > kRetainedPositions ||
            ids.size() * root.hidden * sizeof(float) > kRetainedBytes) {
            r.err = "retained query exceeds 8192 positions or byte bound"; return;
        }
        input = retained_embedding(llm, ids, root.hidden, r.err);
        if (!input.get()) return;
        r.n_prompt = static_cast<int>(ids.size());
        if (fields[7] != "-" && !dump.open_file(root.diagnostic_dir, fields[7], r.err)) return;
    }
    const float* staged = input->readMap<float>();
    if (!staged) { r.err = "retained staged input readMap failed"; return; }
    for (int i = 0; i < input->getInfo()->size; ++i) {
        if (!std::isfinite(staged[i])) { r.err = "nonfinite retained input"; return; }
    }
#ifdef CROSSINFER_QWEN_POSITIONS
    if (op == "append") {
        const size_t n_pre = root.end == 0 ? root.prefix.size() : 0;
        qwen.plan_frame(n_pre, r.n_prompt - static_cast<long long>(n_pre));
    }
#endif
#ifdef CROSSINFER_RETAINED_KV
    if (op == "append" && root.transfer && root.transfer->donor) {
        root.transfer->capture.seen.clear();
        root.transfer->capture.enabled = true;
    }
#endif
    double started = now_s();
    llm.generate_init(); // Clears response/sampler state, NOT the retained KV.
    reply.reset_us = elapsed_us(started, now_s());
    started = now_s();
    reply.prefill_start_s = started;
#ifdef CROSSINFER_RETAINED_KV
    if (root.transfer && root.transfer->importer && root.transfer->importer->snapshot) {
        // The ABI counter spans streams; scope it to this serialized consuming prefill.
        auto& importer = *root.transfer->importer;
        importer.count_before = importer.count();
    }
#endif
    llm.generate(input, 0);
    const float* values = nullptr;
    size_t count = 0;
    bool complete = retained_logits(llm, values, count, r.err);
#ifdef CROSSINFER_RETAINED_KV
    if (complete && root.transfer) {
        if (op == "append" && root.transfer->donor) root.transfer->capture.complete(r.n_prompt);
        root.transfer->complete_import();
    }
#endif
    const double prefill_finished = now_s();
    if (complete) reply.prefill_end_s = prefill_finished;
    reply.prefill_wall_us = elapsed_us(started, prefill_finished);
    const auto* ctx = llm.getContext();
    bool prefill_ok = complete && ctx->status == LlmStatus::RUNNING && ctx->gen_seq_len == 0 &&
                      ctx->output_tokens.empty() && llm.getCurrentHistory() == root.end + r.n_prompt &&
                      ctx->all_seq_len == static_cast<int>(root.end) + r.n_prompt;
#ifdef CROSSINFER_QWEN_POSITIONS
    prefill_ok = prefill_ok && qwen.snapshot_positions().physical_tokens == static_cast<int64_t>(root.end) + r.n_prompt;
#endif
    if (!prefill_ok && r.err.empty()) r.err = "retained completed prefill history/status mismatch";
    if (prefill_ok) reply.first_logits_count = count;
    if (op == "append") {
        if (prefill_ok) {
            root.end += r.n_prompt;
            root.physical = llm.getCurrentHistory();
            root.hidden = input->getInfo()->dim[2];
#ifdef CROSSINFER_QWEN_POSITIONS
            root.qwen_positions = qwen.snapshot_positions();
#endif
            ++root.version;
            root.version_ends[root.version] = root.end;
#ifdef CROSSINFER_QWEN_POSITIONS
            root.version_cursors[root.version] = root.qwen_positions.next_position;
#endif
            r.ok = true;
        } else root.valid = false;
    } else {
        bool diagnostic_ok = true;
        if (prefill_ok && dump.fd >= 0) {
            if (count * sizeof(float) > kRetainedBytes - root.diagnostic_bytes) {
                r.err = "retained diagnostics byte budget exhausted"; diagnostic_ok = false;
            } else diagnostic_ok = dump.write_floats(values, count, r.err);
        }
        started = now_s();
        bool decode_ok = false;
        if (prefill_ok && diagnostic_ok) {
            FirstTokenClock first_clock;
            std::ostream first_stream(&first_clock);
            auto* mutable_ctx = const_cast<LlmContext*>(ctx);
            std::ostream* saved_stream = mutable_ctx->os;
            // Redirect output only; generate_init would reset the retained state.
            mutable_ctx->os = &first_stream;
            llm.generate(1);
            mutable_ctx->os = saved_stream;
            bool first_complete = retained_logits(llm, values, count, r.err);
            if (first_complete && !ctx->output_tokens.empty() && first_clock.first_us > 0) {
                reply.first_token_ready_s = std::max(reply.prefill_end_s, first_clock.first_us / 1e6);
                reply.ttft_us = elapsed_us(reply.request_started, reply.first_token_ready_s);
            }
            if (first_complete && max_tokens > 1) llm.generate(static_cast<int>(max_tokens - 1));
            bool final_complete = first_complete && retained_logits(llm, values, count, r.err);
            decode_ok = final_complete && (ctx->status == LlmStatus::RUNNING ||
                        ctx->status == LlmStatus::NORMAL_FINISHED || ctx->status == LlmStatus::MAX_TOKENS_FINISHED) &&
                        !ctx->output_tokens.empty() && ctx->output_tokens.size() <= static_cast<size_t>(max_tokens) &&
                        llm.getCurrentHistory() == static_cast<size_t>(ctx->all_seq_len) &&
                        ctx->all_seq_len <= kRetainedPositions;
            if (!decode_ok && r.err.empty()) r.err = "retained decode completion/status mismatch";
            r.ids = ctx->output_tokens;
        }
        reply.decode_wall_us = elapsed_us(started, now_s());
        size_t physical = llm.getCurrentHistory();
        if (physical > root.end) llm.eraseHistory(root.end, 0);
#ifdef CROSSINFER_QWEN_POSITIONS
        qwen.restore_positions(root.qwen_positions);
#endif
        reply.rollback_verified = llm.getContext()->all_seq_len == static_cast<int>(root.end) &&
                                  llm.getCurrentHistory() == physical && physical >= root.end;
        root.physical = physical;
        root.valid = prefill_ok && (decode_ok || !diagnostic_ok) && reply.rollback_verified;
        r.ok = root.valid && decode_ok && diagnostic_ok;
        if (!reply.rollback_verified) r.err = "retained query rollback failed; reset required";
        if (r.ok && !dump.name.empty()) {
            dump.committed = true;
            root.diagnostic_bytes += reply.first_logits_count * sizeof(float);
            reply.first_logits_file = dump.name;
        }
    }
    r.prefill_us = ctx->prefill_us;
    r.decode_us = ctx->decode_us;
}

static void retained_respond(long long id, const std::string& op, const RetainedRoot& root,
                             const RetainedReply& reply, Llm& llm, bool trace_enabled,
                             bool has_context = true) {
    const ReqResult& r = reply.result;
    if (r.visual_nbytes) {
        printf("\nVINFO %lld nbytes=%lld crc32=%08x crc_us=%lld\n",
               id, r.visual_nbytes, r.visual_crc32, r.visual_crc_us);
    }
    std::ostringstream csv;
    for (size_t i = 0; i < r.ids.size(); ++i) { if (i) csv << ','; csv << r.ids[i]; }
    size_t physical = has_context ? llm.getCurrentHistory() : 0;
    size_t pending = physical >= root.end ? physical - root.end : 0;
    // Ordinary REQ history is not an inactive retained root's logical position.
    int logical = root.active ? llm.getContext()->all_seq_len : 0;
    if (trace_enabled) {
        printf("\nCTXTIME %lld %s %.9f %.9f %.9f %.9f %.9f\n", id, op.c_str(),
               reply.request_started, reply.prefill_start_s, reply.prefill_end_s,
               reply.first_token_ready_s, reply.finish_s);
    }
#ifdef CROSSINFER_RETAINED_KV
    if (root.transfer) {
        printf("\nCTXKV %lld imported_layers=%zu capture_us=%lld snapshot_bytes=%zu\n", id,
            root.transfer->completed_import_layers,
            static_cast<long long>(root.transfer->capture.snapshot.capture_ms * 1000),
            op == "export" && r.ok ? reply.exported_bytes :
            root.transfer->donor ? root.transfer->capture.snapshot.bytes() : root.transfer->imported.state.bytes());
    }
#endif
    printf("\nRES %lld CTX %d %s %s %lld %d %d %zu %d %zu %zu %zu %d %d "
           "%lld %lld %lld %s %zu %s %s %zu %lld %lld %lld %lld %lld\n",
           id, r.ok ? 1 : 0, op.c_str(), root.stream.empty() ? "-" : root.stream.c_str(),
           root.version, root.active ? 1 : 0, root.valid ? 1 : 0,
           root.end, logical, physical, pending, root.prefix.size(),
           reply.rollback_verified ? 1 : 0, r.n_prompt, r.prefill_us, r.decode_us, r.copied_bytes,
           r.ids.empty() ? "-" : csv.str().c_str(), reply.first_logits_count,
           reply.first_logits_file.empty() ? "-" : reply.first_logits_file.c_str(),
           r.err.empty() ? "-" : b64_encode(r.err).c_str(), reply.root_before, reply.reset_us,
           reply.prefill_wall_us, reply.decode_wall_us, reply.ttft_us, reply.request_wall_us);
    fflush(stdout);
}

static bool retained_getline(std::string& line, bool& oversized) {
    line.clear(); oversized = false;
    char c;
    bool any = false;
    bool in_tag = false, classified = false, context_line = false;
    std::string tag;
    while (std::cin.get(c)) {
        any = true;
        if (c == '\n') break;
        if (!classified) {
            if (std::isspace(static_cast<unsigned char>(c))) {
                if (in_tag) { classified = true; context_line = tag == "CTX"; }
            } else {
                in_tag = true;
                tag.push_back(c);
                if (tag.size() > 3) classified = true;
            }
            if (context_line && line.size() > kRetainedLineBytes) {
                line.resize(kRetainedLineBytes);
                oversized = true;
            }
        }
        if (!context_line || line.size() < kRetainedLineBytes) line.push_back(c);
        else oversized = true;
    }
    return any;
}
