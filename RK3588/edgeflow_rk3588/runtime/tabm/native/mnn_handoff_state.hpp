#ifndef CROSSINFER_MNN_HANDOFF_STATE_HPP
#define CROSSINFER_MNN_HANDOFF_STATE_HPP

// Include after Llm, require() and now_ms(); shared by probe and opt-in worker.
#include "mnn_kv_transfer.hpp"
#include <MNN/Tensor.hpp>
#include <dlfcn.h>
#include <map>
#include <set>

struct HandoffLayer {
  uint32_t tokens = 0, heads = 0, dimension = 0;
  std::vector<float> key, value;
};

struct HandoffSnapshot {
  size_t positions = 0;
  std::map<std::string, HandoffLayer> layers;
  double capture_ms = 0;
  size_t bytes() const {
    size_t result = 0;
    for (const auto &item : layers)
      result +=
          (item.second.key.size() + item.second.value.size()) * sizeof(float);
    return result;
  }
};

static size_t bounded_kv_count(const MNN::Tensor *tensor) {
  require(tensor && tensor->dimensions() == 4 && tensor->length(0) == 1 &&
              tensor->getType() == halide_type_of<float>() &&
              tensor->length(1) > 0 && tensor->length(1) <= 8192 &&
              tensor->length(2) > 0 && tensor->length(2) <= 128 &&
              tensor->length(3) > 0 && tensor->length(3) <= 512,
          "unsupported attention tensor contract");
  const size_t count = static_cast<size_t>(tensor->length(1)) *
                       tensor->length(2) * tensor->length(3);
  require(count <= 8 * 1024 * 1024, "K/V shape/byte bound exceeded");
  return count;
}

static HandoffLayer
capture_attention(const std::vector<MNN::Tensor *> &inputs) {
  require(inputs.size() >= 3, "attention has no K/V inputs");
  HandoffLayer layer;
  std::vector<float> *arrays[] = {&layer.key, &layer.value};
  for (int i = 0; i != 2; ++i) {
    MNN::Tensor *device = inputs[i + 1];
    const size_t expected_count = bounded_kv_count(device);
    // getDimensionType() reports CAFFE for packed C4 too. An explicit planar
    // destination makes the backend unpack rather than preserving that format.
    std::unique_ptr<MNN::Tensor> host(
        new MNN::Tensor(device, MNN::Tensor::CAFFE));
    require(host && host->host<float>() &&
                host->getDimensionType() == MNN::Tensor::CAFFE &&
                device->copyToHostTensor(host.get()),
            "canonical K/V host conversion failed");
    const uint32_t n = host->length(1), h = host->length(2),
                   d = host->length(3);
    if (i == 0) {
      layer.tokens = n;
      layer.heads = h;
      layer.dimension = d;
    } else {
      require(n == layer.tokens && h == layer.heads && d == layer.dimension,
              "K/V shapes differ");
    }
    const size_t count = static_cast<size_t>(n) * h * d;
    require(count == expected_count && host->elementSize() == count,
            "K/V shape/byte bound exceeded");
    arrays[i]->assign(host->host<float>(), host->host<float>() + count);
  }
  CrossInferKVLayerV1 view = {1,
                              sizeof(CrossInferKVLayerV1),
                              layer.tokens,
                              layer.heads,
                              layer.dimension,
                              layer.key.size(),
                              layer.key.data(),
                              layer.value.data()};
  require(crossinferValidKVLayer(view), "invalid/nonfinite captured K/V");
  return layer;
}

class HandoffCapture {
public:
  HandoffSnapshot snapshot;
  bool enabled = false;
  std::set<std::string> seen;
  void attach(Llm &llm) {
    llm.setDebugCallback(
        [this](const std::vector<MNN::Tensor *> &inputs,
               const MNN::OperatorInfo *op) {
          if (!enabled || op->type() != "Attention")
            return true;
          const double started = now_ms();
          const std::string name = op->name();
          require(!name.empty() && seen.insert(name).second,
                  "duplicate/unnamed attention layer");
          require(inputs.size() >= 3 &&
                      snapshot.bytes() + (bounded_kv_count(inputs[1]) +
                                          bounded_kv_count(inputs[2])) *
                                             sizeof(float) <=
                          256 * 1024 * 1024 &&
                      snapshot.positions + inputs[1]->length(1) <= 8192,
                  "snapshot bound exceeded before allocation");
          HandoffLayer added = capture_attention(inputs);
          auto &layer = snapshot.layers[name];
          require(layer.tokens == snapshot.positions &&
                      (!layer.tokens || (layer.heads == added.heads &&
                                         layer.dimension == added.dimension)),
                  "inconsistent capture history/layer shape");
          layer.tokens += added.tokens;
          layer.heads = added.heads;
          layer.dimension = added.dimension;
          layer.key.insert(layer.key.end(), added.key.begin(), added.key.end());
          layer.value.insert(layer.value.end(), added.value.begin(),
                             added.value.end());
          snapshot.capture_ms += now_ms() - started;
          return true;
        },
        [](const std::vector<MNN::Tensor *> &, const MNN::OperatorInfo *) {
          return true;
        });
  }
  void complete(size_t added) {
    require(!seen.empty() && seen.size() == snapshot.layers.size(),
            "not all attention layers were captured");
    snapshot.positions += added;
    require(snapshot.positions <= 8192 && snapshot.bytes() <= 256 * 1024 * 1024,
            "snapshot bounds exceeded");
    for (const auto &item : snapshot.layers)
      require(item.second.tokens == snapshot.positions,
              "incomplete layer snapshot");
    enabled = false;
  }
};

template <typename T> static T handoff_symbol(const char *name) {
  void *symbol = dlsym(RTLD_DEFAULT, name);
  require(symbol != nullptr,
          std::string("missing isolated MNN capability: ") + name);
  return reinterpret_cast<T>(symbol);
}

static void validate_handoff_prefill_policy(bool quant_gemm,
                                            const std::string &backend,
                                            const std::string &precision) {
  const char *policy = std::getenv("MNN_CROSSINFER_PREFILL_LINEAR_POLICY");
  if (quant_gemm) {
    require(backend == "opencl" && precision == "low",
            "quant-GEMM source config must be OpenCL low precision");
    const auto capability = handoff_symbol<int (*)()>(
        "mnn_crossinfer_opencl_prefill_quant_gemm_v1");
    require(policy && std::strcmp(policy, "quant_gemm") == 0 &&
                capability() == 1,
            "missing/live-disabled quant-GEMM source capability");
  } else {
    require(!policy, "unexpected inherited source numerical policy");
  }
}

struct HandoffImporter {
  using Stage = int (*)(const CrossInferKVLayerV1 *);
  using History = int (*)(void *, int);
  using Count = size_t (*)();
  Stage stage = handoff_symbol<Stage>("mnn_crossinfer_cpu_stage_kv_v1");
  History history = handoff_symbol<History>("mnn_crossinfer_import_history_v1");
  Count count = handoff_symbol<Count>("mnn_crossinfer_cpu_imported_layers_v1");
  const HandoffSnapshot *snapshot = nullptr;
  std::set<std::string> seen;
  size_t count_before = 0;
  double stage_ms = 0;
  void attach(Llm &llm) {
    const auto size = handoff_symbol<Count>("mnn_crossinfer_kv_layer_size_v1");
    require(size() == sizeof(CrossInferKVLayerV1), "KV transfer ABI mismatch");
    llm.setDebugCallback(
        [this](const std::vector<MNN::Tensor *> &,
               const MNN::OperatorInfo *op) {
          if (!snapshot || op->type() != "Attention")
            return true;
          const auto found = snapshot->layers.find(op->name());
          require(found != snapshot->layers.end() &&
                      seen.insert(op->name()).second,
                  "recipient attention layer mismatch");
          const auto &layer = found->second;
          CrossInferKVLayerV1 view = {1,
                                      sizeof(CrossInferKVLayerV1),
                                      layer.tokens,
                                      layer.heads,
                                      layer.dimension,
                                      layer.key.size(),
                                      layer.key.data(),
                                      layer.value.data()};
          const double started = now_ms();
          require(stage(&view) == 1, "CPU rejected imported K/V layer");
          stage_ms += now_ms() - started;
          return true;
        },
        [](const std::vector<MNN::Tensor *> &, const MNN::OperatorInfo *) {
          return true;
        });
  }
  void begin(Llm &llm, const HandoffSnapshot &state) {
    require(!snapshot && !state.layers.empty() && state.positions > 0 &&
                state.positions <= 8192,
            "invalid import lifecycle");
    for (const auto &item : state.layers) {
      const auto &layer = item.second;
      CrossInferKVLayerV1 view = {1,
                                  sizeof(CrossInferKVLayerV1),
                                  layer.tokens,
                                  layer.heads,
                                  layer.dimension,
                                  layer.key.size(),
                                  layer.key.data(),
                                  layer.value.data()};
      require(!item.first.empty() && layer.tokens == state.positions &&
                  layer.key.size() == layer.value.size() &&
                  crossinferValidKVLayer(view),
              "invalid snapshot before destination reset");
    }
    llm.reset();
    llm.generate_init();
    seen.clear();
    stage_ms = 0;
    count_before = count();
    require(history(&llm, static_cast<int>(state.positions)) == 1,
            "recipient rejected logical history");
    snapshot = &state;
  }
  void complete() {
    require(snapshot && seen.size() == snapshot->layers.size() &&
                count() - count_before == seen.size(),
            "missing/duplicate actual CPU KV imports");
    snapshot = nullptr;
  }
};

#endif
