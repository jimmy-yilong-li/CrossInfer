#pragma once

#include "llama.h"

#include <array>
#include <vector>

// Own only batch metadata. The caller owns token/embedding storage and must
// retain it until llama_decode + llama_synchronize have completed.
struct BorrowedBatch {
    std::vector<llama_pos> positions;
    std::vector<int32_t> sequence_counts;
    std::array<llama_seq_id, 1> sequence{0};
    std::vector<llama_seq_id *> sequence_ids;
    std::vector<int8_t> logits;
    llama_batch value{};

    BorrowedBatch(const int32_t * tokens, const float * features, int count,
                  int start, bool logits_last)
        : positions(count), sequence_counts(count, 1),
          sequence_ids(count, sequence.data()), logits(count, 0) {
        for (int i = 0; i < count; ++i) positions[i] = start + i;
        if (logits_last) logits.back() = 1;
        value = {count, const_cast<int32_t *>(tokens), const_cast<float *>(features),
                 positions.data(), sequence_counts.data(), sequence_ids.data(), logits.data()};
    }
};
