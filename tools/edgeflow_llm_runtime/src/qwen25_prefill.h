#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

struct Qwen25PackedLayout {
    int32_t tokens;
    int32_t next_position;
    std::vector<int32_t> positions;
};

inline Qwen25PackedLayout qwen25_packed_layout(int prefix, int height,
                                              int width, int suffix) {
    const int64_t visual = int64_t(height) * width;
    const int64_t count = int64_t(prefix) + visual + suffix;
    const int64_t end = int64_t(prefix) + std::max(height, width) + suffix;
    if (prefix <= 0 || suffix <= 0 || height <= 0 || width <= 0 ||
        count > std::numeric_limits<int32_t>::max() ||
        end > std::numeric_limits<int32_t>::max()) {
        throw std::invalid_argument("invalid packed Qwen prompt dimensions");
    }
    Qwen25PackedLayout result{int32_t(count), int32_t(end),
                              std::vector<int32_t>(size_t(count) * 4, 0)};
    for (int i = 0; i < count; ++i) {
        int t, y, x;
        if (i < prefix) {
            t = y = x = i;
        } else if (i < prefix + visual) {
            t = prefix;
            y = prefix + (i - prefix) / width;
            x = prefix + (i - prefix) % width;
        } else {
            t = y = x = prefix + std::max(height, width) + int(i - prefix - visual);
        }
        result.positions[i] = t;
        result.positions[count + i] = y;
        result.positions[2 * count + i] = x;
    }
    return result;
}
