#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace edgeflow {

// Each sequence owns its last logits, including when another sequence runs.
// Reuse storage and the greedy decision until a new decode replaces them.
class SessionLogits {
public:
  void assign(const float *values, const size_t count) {
    if (values == nullptr || count == 0) {
      throw std::runtime_error("session logits are unavailable");
    }
    greedy_token_ = -1;
    masked_token_ = -1;
    values_.assign(values, values + count);
  }

  void clear() {
    greedy_token_ = -1;
    masked_token_ = -1;
    values_.clear();
  }

  const std::vector<float> &values() const { return values_; }

  int32_t argmax_excluding(const std::vector<int32_t> &excluded_sorted) {
    if (masked_token_ >= 0 && masked_exclusions_ == excluded_sorted) {
      return masked_token_;
    }
    int32_t best = -1;
    for (size_t i = 0; i < values_.size(); ++i) {
      if (!std::isfinite(values_[i])) {
        throw std::runtime_error("session logits contain a non-finite value");
      }
      if (!std::binary_search(excluded_sorted.begin(), excluded_sorted.end(),
                              static_cast<int32_t>(i)) &&
          (best < 0 || values_[i] > values_[best])) {
        best = static_cast<int32_t>(i);
      }
    }
    if (best < 0) throw std::runtime_error("no eligible session logits");
    masked_exclusions_ = excluded_sorted;
    masked_token_ = best;
    return best;
  }

  int32_t argmax() {
    if (values_.empty()) {
      throw std::runtime_error("session logits are empty");
    }
    if (greedy_token_ >= 0) {
      return greedy_token_;
    }
    size_t best = 0;
    size_t i = 0;
#if defined(__aarch64__) && defined(__ARM_NEON)
    if (values_.size() >= 4) {
      float32x4_t maxima = vdupq_n_f32(-std::numeric_limits<float>::infinity());
      uint32x4_t indices = vdupq_n_u32(UINT32_MAX);
      uint32x4_t valid = vdupq_n_u32(UINT32_MAX);
      const float32x4_t finite_limit = vdupq_n_f32(std::numeric_limits<float>::max());
      uint32x4_t positions = {0, 1, 2, 3};
      for (; i + 4 <= values_.size(); i += 4) {
        const float32x4_t values = vld1q_f32(values_.data() + i);
        valid = vandq_u32(valid, vcleq_f32(vabsq_f32(values), finite_limit));
        const uint32x4_t greater = vcgtq_f32(values, maxima);
        indices = vbslq_u32(greater, positions, indices);
        maxima = vbslq_f32(greater, values, maxima);
        positions = vaddq_u32(positions, vdupq_n_u32(4));
      }
      if (vminvq_u32(valid) == 0) {
        throw std::runtime_error("session logits contain a non-finite value");
      }
      // Equal maxima retain the earliest vocabulary index, as in the scalar path.
      const float maximum = vmaxvq_f32(maxima);
      best = vminvq_u32(vbslq_u32(vceqq_f32(maxima, vdupq_n_f32(maximum)),
                                  indices, vdupq_n_u32(UINT32_MAX)));
    }
#endif
    for (; i < values_.size(); ++i) {
      if (!std::isfinite(values_[i])) {
        throw std::runtime_error("session logits contain a non-finite value");
      }
      if (values_[i] > values_[best]) {
        best = i;
      }
    }
    greedy_token_ = static_cast<int32_t>(best);
    return greedy_token_;
  }

private:
  std::vector<float> values_;
  int32_t greedy_token_ = -1;
  int32_t masked_token_ = -1;
  std::vector<int32_t> masked_exclusions_;
};

} // namespace edgeflow
