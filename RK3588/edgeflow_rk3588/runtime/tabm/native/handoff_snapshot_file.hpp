#ifndef CROSSINFER_HANDOFF_SNAPSHOT_FILE_HPP
#define CROSSINFER_HANDOFF_SNAPSHOT_FILE_HPP

#include <cstdio>
#include <fcntl.h>
#include <unistd.h>

// Experiment-local, same-machine format; arrays are canonical fp32 KV, not logits.
struct WorkerKVSnapshot {
  HandoffSnapshot state;
  std::string stream, model_key, content_digest;
  int64_t version = 0;
  int32_t hidden = 0;
  std::vector<int> prefix;
#ifdef CROSSINFER_QWEN_POSITIONS
  int64_t qwen_next_position = 0;
#endif
};

static void validate_worker_kv_snapshot(const WorkerKVSnapshot& file,
                                      const HandoffSnapshot& source, size_t positions) {
  require(!file.stream.empty() && file.stream.size() <= 128 &&
          !file.model_key.empty() && file.model_key.size() <= 128 && file.content_digest.size() == 64 &&
          file.content_digest.find_first_not_of("0123456789abcdef") == std::string::npos &&
          file.version > 0 && file.hidden > 0 && file.hidden <= 16384 &&
          !file.prefix.empty() && file.prefix.size() < positions && positions <= source.positions &&
          source.positions <= 8192 && !source.layers.empty() &&
          source.layers.size() <= 128 && source.bytes() <= 256 * 1024 * 1024,
          "invalid KV snapshot metadata");
#ifdef CROSSINFER_QWEN_POSITIONS
  require(file.qwen_next_position > 0 &&
          static_cast<uint64_t>(file.qwen_next_position) <= positions,
          "invalid Qwen KV semantic cursor");
#endif
  for (const auto& item : source.layers) {
    const auto& layer = item.second;
    CrossInferKVLayerV1 view = {1, sizeof(CrossInferKVLayerV1), layer.tokens,
      layer.heads, layer.dimension, layer.key.size(), layer.key.data(), layer.value.data()};
    require(!item.first.empty() && item.first.size() <= 1024 &&
            layer.tokens == source.positions && layer.key.size() == layer.value.size() &&
            crossinferValidKVLayer(view), "invalid KV snapshot layer");
  }
}

static void validate_worker_kv_snapshot(const WorkerKVSnapshot& file) {
  validate_worker_kv_snapshot(file, file.state, file.state.positions);
}

template<class T> static void kv_write(FILE* file, const T& value) {
  require(std::fwrite(&value, sizeof(T), 1, file) == 1, "KV snapshot write failed");
}
template<class T> static T kv_read(FILE* file) {
  T value;
  require(std::fread(&value, sizeof(T), 1, file) == 1, "truncated KV snapshot");
  return value;
}
static void kv_write_string(FILE* file, const std::string& value) {
  kv_write(file, static_cast<uint32_t>(value.size()));
  require(std::fwrite(value.data(), 1, value.size(), file) == value.size(), "KV string write failed");
}
static std::string kv_read_string(FILE* file, size_t limit) {
  const auto size = kv_read<uint32_t>(file);
  require(size > 0 && size <= limit, "invalid KV snapshot string length");
  std::string value(size, '\0');
  require(std::fread(&value[0], 1, size, file) == size, "truncated KV snapshot string");
  require(value.find('\0') == std::string::npos, "invalid KV snapshot string");
  return value;
}

static void write_worker_kv_snapshot(const std::string& path, const WorkerKVSnapshot& state,
                                    const HandoffSnapshot& source, size_t positions) {
  validate_worker_kv_snapshot(state, source, positions);
  const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
  require(fd >= 0, "cannot create immutable KV snapshot");
  FILE* raw = fdopen(fd, "wb");
  if (!raw) { close(fd); unlink(path.c_str()); require(false, "cannot open KV snapshot stream"); }
  std::unique_ptr<FILE, decltype(&std::fclose)> file(raw, &std::fclose);
  try {
#ifdef CROSSINFER_QWEN_POSITIONS
    kv_write(file.get(), uint64_t{0x32564b5f4b525843}); // Qwen-only v2, never read as legacy v1.
#else
    kv_write(file.get(), uint64_t{0x31564b5f4b525843});
#endif
    kv_write_string(file.get(), state.stream);
    kv_write_string(file.get(), state.model_key);
    kv_write_string(file.get(), state.content_digest);
    kv_write(file.get(), state.version);
    kv_write(file.get(), state.hidden);
    kv_write(file.get(), static_cast<uint32_t>(positions));
#ifdef CROSSINFER_QWEN_POSITIONS
    kv_write(file.get(), state.qwen_next_position);
#endif
    kv_write(file.get(), static_cast<uint32_t>(state.prefix.size()));
    for (const int token : state.prefix) kv_write(file.get(), static_cast<int32_t>(token));
    kv_write(file.get(), static_cast<uint32_t>(source.layers.size()));
    for (const auto& item : source.layers) {
      kv_write_string(file.get(), item.first);
      const auto& layer = item.second;
      kv_write(file.get(), static_cast<uint32_t>(positions));
      kv_write(file.get(), layer.heads); kv_write(file.get(), layer.dimension);
      const auto count = positions * layer.heads * layer.dimension;
      require(std::fwrite(layer.key.data(), sizeof(float), count, file.get()) == count &&
              std::fwrite(layer.value.data(), sizeof(float), count, file.get()) == count,
              "KV array write failed");
    }
    require(std::fflush(file.get()) == 0, "KV snapshot flush failed");
  } catch (...) { file.reset(); unlink(path.c_str()); throw; }
}

static void write_worker_kv_snapshot(const std::string& path, const WorkerKVSnapshot& state) {
  write_worker_kv_snapshot(path, state, state.state, state.state.positions);
}

static WorkerKVSnapshot read_worker_kv_snapshot(const std::string& path) {
  std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(path.c_str(), "rb"), &std::fclose);
  require(file != nullptr, "cannot open KV snapshot");
#ifdef CROSSINFER_QWEN_POSITIONS
  require(kv_read<uint64_t>(file.get()) == 0x32564b5f4b525843, "invalid Qwen v2 KV snapshot magic");
#else
  require(kv_read<uint64_t>(file.get()) == 0x31564b5f4b525843, "invalid KV snapshot magic");
#endif
  WorkerKVSnapshot state;
  state.stream = kv_read_string(file.get(), 128);
  state.model_key = kv_read_string(file.get(), 128);
  state.content_digest = kv_read_string(file.get(), 64);
  state.version = kv_read<int64_t>(file.get());
  state.hidden = kv_read<int32_t>(file.get());
  state.state.positions = kv_read<uint32_t>(file.get());
  require(state.state.positions > 0 && state.state.positions <= 8192, "invalid KV positions");
#ifdef CROSSINFER_QWEN_POSITIONS
  state.qwen_next_position = kv_read<int64_t>(file.get());
#endif
  const auto prefix = kv_read<uint32_t>(file.get());
  require(prefix > 0 && prefix < state.state.positions, "invalid KV prefix length");
  for (uint32_t i = 0; i < prefix; ++i) state.prefix.push_back(kv_read<int32_t>(file.get()));
  const auto layers = kv_read<uint32_t>(file.get());
  require(layers > 0 && layers <= 128, "invalid KV layer count");
  for (uint32_t i = 0; i < layers; ++i) {
    const auto name = kv_read_string(file.get(), 1024);
    require(state.state.layers.count(name) == 0, "duplicate KV layer");
    HandoffLayer layer;
    layer.tokens = kv_read<uint32_t>(file.get());
    layer.heads = kv_read<uint32_t>(file.get());
    layer.dimension = kv_read<uint32_t>(file.get());
    const size_t count = static_cast<size_t>(layer.tokens) * layer.heads * layer.dimension;
    require(layer.tokens == state.state.positions && layer.heads > 0 && layer.heads <= 128 &&
            layer.dimension > 0 && layer.dimension <= 512 && count <= 8 * 1024 * 1024 &&
            state.state.bytes() + count * sizeof(float) * 2 <= 256 * 1024 * 1024,
            "KV layer exceeds bounds");
    layer.key.resize(count); layer.value.resize(count);
    require(std::fread(layer.key.data(), sizeof(float), count, file.get()) == count &&
            std::fread(layer.value.data(), sizeof(float), count, file.get()) == count,
            "truncated KV array");
    state.state.layers.emplace(name, std::move(layer));
  }
  require(std::fgetc(file.get()) == EOF && !std::ferror(file.get()), "unexpected KV trailing data");
  validate_worker_kv_snapshot(state);
  return state;
}
#endif
