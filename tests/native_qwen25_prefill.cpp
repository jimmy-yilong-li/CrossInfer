#include "qwen25_prefill.h"

#include <cassert>
#include <stdexcept>

int main() {
    const auto layout = qwen25_packed_layout(4, 14, 14, 11);
    assert(layout.tokens == 211 && layout.next_position == 29);
    for (int i = 0; i < 4; ++i) {
        assert(layout.positions[i] == i);
        assert(layout.positions[211 + i] == i);
        assert(layout.positions[422 + i] == i);
    }
    for (int i = 0; i < 196; ++i) {
        assert(layout.positions[4 + i] == 4);
        assert(layout.positions[211 + 4 + i] == 4 + i / 14);
        assert(layout.positions[422 + 4 + i] == 4 + i % 14);
    }
    for (int i = 0; i < 11; ++i) {
        for (int axis = 0; axis < 3; ++axis) {
            assert(layout.positions[axis * 211 + 200 + i] == 18 + i);
        }
    }
    const auto wide = qwen25_packed_layout(7, 3, 5, 2);
    assert(wide.tokens == 24 && wide.next_position == 14);
    for (const auto & value : {layout, wide}) {
        assert(value.positions.size() == size_t(4 * value.tokens));
        for (int i = 0; i < value.tokens; ++i) {
            assert(value.positions[3 * value.tokens + i] == 0);
        }
    }
    try {
        qwen25_packed_layout(0, 14, 14, 11);
        return 1;
    } catch (const std::invalid_argument &) {}
}
