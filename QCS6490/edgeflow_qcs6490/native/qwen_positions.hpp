#ifndef EDGEFLOW_QCS6490_QWEN_POSITIONS_HPP
#define EDGEFLOW_QCS6490_QWEN_POSITIONS_HPP

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace edgeflow_qcs6490 {

// Image-sequence M-RoPE only: each frame is an image, not native temporal video.
class QwenPositions {
public:
    // Axes are temporal, height, width. Concatenate returned segments per axis.
    using Positions = std::array<std::vector<std::int32_t>, 3>;

    struct State {
        std::int64_t physical_tokens;
        std::int64_t next_position;
    };

    std::int64_t physical_tokens() const { return physical_tokens_; }
    std::int64_t next_position() const { return next_position_; }

    State snapshot() const { return {physical_tokens_, next_position_}; }

    // Restore counters for a retained/transferred root, not KV or output buffers.
    void restore(State state) {
        if (state.physical_tokens < 0 || state.next_position < 0) {
            throw std::invalid_argument("Qwen position counters must be nonnegative");
        }
        if (state.next_position > position_limit()) {
            throw std::overflow_error("Qwen semantic position overflow");
        }
        physical_tokens_ = state.physical_tokens;
        next_position_ = state.next_position;
    }

    Positions append_text(std::int64_t token_count) {
        if (token_count < 0) {
            throw std::invalid_argument("Qwen text token count must be nonnegative");
        }
        check_position_growth(token_count);
        check_physical_growth(token_count);
        Positions positions = make_positions(token_count);
        for (std::int64_t i = 0; i < token_count; ++i) {
            const std::int32_t position = static_cast<std::int32_t>(next_position_ + i);
            for (auto& axis : positions) {
                axis.push_back(position);
            }
        }
        physical_tokens_ += token_count;
        next_position_ += token_count;
        return positions;
    }

    Positions append_image(std::int64_t merged_h, std::int64_t merged_w) {
        if (merged_h <= 0 || merged_w <= 0) {
            throw std::invalid_argument("Qwen merged image dimensions must be positive");
        }
        const std::int64_t extent = merged_h > merged_w ? merged_h : merged_w;
        check_position_growth(extent);
        // The position bound limits both dimensions to 2^31, so h*w fits int64.
        const std::int64_t token_count = merged_h * merged_w;
        check_physical_growth(token_count);
        Positions positions = make_positions(token_count);
        for (std::int64_t row = 0; row < merged_h; ++row) {
            for (std::int64_t col = 0; col < merged_w; ++col) {
                positions[0].push_back(static_cast<std::int32_t>(next_position_));
                positions[1].push_back(static_cast<std::int32_t>(next_position_ + row));
                positions[2].push_back(static_cast<std::int32_t>(next_position_ + col));
            }
        }
        // Commit only after allocation/filling succeeds; physical != semantic.
        physical_tokens_ += token_count;
        next_position_ += extent;
        return positions;
    }

private:
    // One past the last int32 coordinate is a valid exhausted-state sentinel.
    static std::int64_t position_limit() {
        return static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) + 1;
    }

    void check_position_growth(std::int64_t increment) const {
        if (increment > position_limit() - next_position_) {
            throw std::overflow_error("Qwen semantic position overflow");
        }
    }

    void check_physical_growth(std::int64_t increment) const {
        if (increment > std::numeric_limits<std::int64_t>::max() - physical_tokens_) {
            throw std::overflow_error("Qwen physical token count overflow");
        }
    }

    static Positions make_positions(std::int64_t token_count) {
        Positions positions;
        for (auto& axis : positions) {
            if (static_cast<std::uint64_t>(token_count) > axis.max_size()) {
                throw std::overflow_error("Qwen position vector size overflow");
            }
            axis.reserve(static_cast<std::vector<std::int32_t>::size_type>(token_count));
        }
        return positions;
    }

    std::int64_t physical_tokens_ = 0;
    std::int64_t next_position_ = 0;
};

}  // namespace edgeflow_qcs6490

#endif
