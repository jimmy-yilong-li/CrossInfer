#ifndef CROSSINFER_QWEN_MNN_POSITIONS_HPP
#define CROSSINFER_QWEN_MNN_POSITIONS_HPP

#include "qwen_positions.hpp"
#include "llm/llm.hpp"
#include "llmconfig.hpp" // Use the header from the linked libllm's engine/src.
#include "core/KVMeta.hpp"
#include <MNN/expr/ExprCreator.hpp>

namespace edgeflow_qcs6490 {

// Language-only Llm: never use createLLM/Omni, which loads a visual model.
// Same-backend segmented image reference, not native video temporal positions.
class QwenMnnLlm : public MNN::Transformer::Llm {
public:
    enum { kFrameRows = 228, kVisionStartId = 151652, kVisionEndId = 151653, kNewlineId = 198 };

    explicit QwenMnnLlm(std::shared_ptr<MNN::Transformer::LlmConfig> config)
        : MNN::Transformer::Llm(config) {
        if (!config->is_mrope()) throw std::invalid_argument("Qwen sidecar requires is_mrope=true");
    }

    QwenPositions::State snapshot_positions() const { return positions_.snapshot(); }

    // The runtime's v1 history setter deliberately rejects M-RoPE. This target
    // separately persists/restores its semantic cursor, so install physical KV
    // history here without changing the generic library or its text-only gate.
    bool install_physical_history(int positions) {
        using MNN::Transformer::LlmStatus;
        if (positions <= 0 || positions > 8192 || mConfig->backend_type() != "cpu" || !mConfig->reuse_kv() ||
            mContext->status != LlmStatus::RUNNING || mContext->all_seq_len != 0 || mContext->gen_seq_len != 0 ||
            !mContext->history_tokens.empty() || !mContext->output_tokens.empty() ||
            mMeta->previous != mMeta->remove || mMeta->n_reserve != 0 || mMeta->file_flag != MNN::KVMeta::NoChange) return false;
        mMeta->previous = static_cast<size_t>(positions);
        mMeta->remove = mMeta->add = 0; mMeta->reserve = nullptr; mMeta->n_reserve = 0;
        mContext->all_seq_len = positions; mContext->current_token = -1;
        return true;
    }

    void restore_positions(QwenPositions::State state) {
        positions_.restore(state);
        planned_ = QwenPositions::Positions();
    }

    void reset() override {
        MNN::Transformer::Llm::reset();
        restore_positions({0, 0});
    }

    // SHM already contains [vision_start | 225 true features | vision_end | newline].
    // Prefix is separate text, included only in the first native prefill.
    void plan_frame(std::int64_t prefix_tokens, std::int64_t payload_rows) {
        if (payload_rows != kFrameRows || prefix_tokens < 0 ||
            (positions_.physical_tokens() != 0 && prefix_tokens != 0) || !planned_[0].empty()) {
            throw std::invalid_argument("Qwen append requires one 228-row image packet and first-frame-only prefix");
        }
        QwenPositions next = positions_;
        QwenPositions::Positions packet = next.append_text(prefix_tokens);
        append(packet, next.append_text(1)); // vision_start is a text coordinate.
        append(packet, next.append_image(15, 15));
        append(packet, next.append_text(2)); // vision_end, newline
        planned_after_ = next.snapshot();
        planned_.swap(packet);
    }

    MNN::Express::VARP gen_position_ids(int seq_len) override {
        if (seq_len <= 0 || positions_.physical_tokens() != mContext->all_seq_len) {
            throw std::runtime_error("Qwen logical KV root and semantic cursor disagree");
        }
        const bool image_packet = !planned_[0].empty();
        if (image_packet && planned_[0].size() != static_cast<size_t>(seq_len)) {
            throw std::runtime_error("Qwen planned packet length differs from native prefill (requires chunk=0)");
        }
        QwenPositions next = positions_;
        QwenPositions::Positions text;
        if (!image_packet) text = next.append_text(seq_len); // Query and future decode use semantic base.
        const auto& coordinates = image_packet ? planned_ : text;
        auto result = MNN::Express::_Input({3, seq_len}, MNN::Express::NCHW, halide_type_of<int>());
        int* target = result.get() ? result->writeMap<int>() : nullptr;
        if (!target) throw std::runtime_error("Qwen position map failed");
        for (int axis = 0; axis < 3; ++axis) {
            for (int i = 0; i < seq_len; ++i) target[axis * seq_len + i] = coordinates[axis][i];
        }
        positions_.restore(image_packet ? planned_after_ : next.snapshot());
        planned_ = QwenPositions::Positions();
        return result;
    }

private:
    static void append(QwenPositions::Positions& packet, const QwenPositions::Positions& segment) {
        for (int axis = 0; axis < 3; ++axis) {
            packet[axis].insert(packet[axis].end(), segment[axis].begin(), segment[axis].end());
        }
    }

    QwenPositions positions_;
    QwenPositions::Positions planned_;
    QwenPositions::State planned_after_ = {0, 0};
};

} // namespace edgeflow_qcs6490

#endif
