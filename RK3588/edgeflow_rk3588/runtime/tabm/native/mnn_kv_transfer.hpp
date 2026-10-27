#ifndef CROSSINFER_MNN_KV_TRANSFER_HPP
#define CROSSINFER_MNN_KV_TRANSFER_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// Borrowed canonical [position, KV head, head dimension] arrays. The caller
// owns them until the synchronous attention import has completed.
struct CrossInferKVLayerV1 {
  uint32_t abi;
  uint32_t size;
  uint32_t tokens;
  uint32_t heads;
  uint32_t dimension;
  size_t count;
  const float *key;
  const float *value;
};

inline bool crossinferKVTransferEnabled() {
  const char *flag = std::getenv("CROSSINFER_MNN_KV_TRANSFER");
  return flag && std::strcmp(flag, "1") == 0;
}

inline bool crossinferValidKVLayer(const CrossInferKVLayerV1 &layer) {
  if (layer.abi != 1 || layer.size != sizeof(layer) || !layer.tokens ||
      layer.tokens > 8192 || !layer.heads || layer.heads > 128 ||
      !layer.dimension || layer.dimension > 512 || !layer.key || !layer.value)
    return false;
  const size_t count =
      static_cast<size_t>(layer.tokens) * layer.heads * layer.dimension;
  if (count > 8 * 1024 * 1024 || layer.count != count)
    return false;
  for (size_t i = 0; i < count; ++i)
    if (!std::isfinite(layer.key[i]) || !std::isfinite(layer.value[i]))
      return false;
  return true;
}

#endif
