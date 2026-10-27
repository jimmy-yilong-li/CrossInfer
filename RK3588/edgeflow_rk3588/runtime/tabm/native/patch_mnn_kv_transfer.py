#!/usr/bin/env python3
"""Create isolated MNN source copies with the canonical-KV import capability.

No installed file is modified. The new nonvirtual method changes no class
layout; the CPU cache packing path is entered only with an explicitly staged
layer and CROSSINFER_MNN_KV_TRANSFER=1. This is not a serving-default switch.
"""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path

_CPU_BRIDGE = r"""
#include "mnn_kv_transfer.hpp"

namespace {
thread_local CrossInferKVLayerV1 crossinferPendingKV = {};
thread_local bool crossinferHasPendingKV = false;
thread_local size_t crossinferImportedLayers = 0;
}

extern "C" __attribute__((visibility("default"))) size_t mnn_crossinfer_kv_layer_size_v1() {
    return sizeof(CrossInferKVLayerV1);
}
extern "C" __attribute__((visibility("default"))) int mnn_crossinfer_cpu_stage_kv_v1(const CrossInferKVLayerV1* layer) {
    if (!crossinferKVTransferEnabled() || crossinferHasPendingKV || !layer || !crossinferValidKVLayer(*layer)) return 0;
    crossinferPendingKV = *layer;
    crossinferHasPendingKV = true;
    return 1;
}
extern "C" __attribute__((visibility("default"))) size_t mnn_crossinfer_cpu_imported_layers_v1() {
    return crossinferImportedLayers;
}
static bool crossinferTakeKVLayer(CrossInferKVLayerV1& layer) {
    if (!crossinferHasPendingKV) return false;
    layer = crossinferPendingKV;
    crossinferPendingKV = {};
    crossinferHasPendingKV = false;
    return true;
}
"""

_CPU_IMPORT = r"""        CrossInferKVLayerV1 layer = {};
        if (crossinferTakeKVLayer(layer)) {
            if (!crossinferKVTransferEnabled() || !mKVCache || !mMeta ||
                mMeta->previous != layer.tokens || mMeta->remove != 0 ||
                mMeta->n_reserve != 0 || mBytes < 2 || mBytes > 4 ||
                mKeyQuantMode != KVQuantMode::None || mValueQuantMode != KVQuantMode::None ||
                layer.heads != static_cast<uint32_t>(mKvNumHead) ||
                layer.dimension != static_cast<uint32_t>(mHeadDim) ||
                layer.tokens + seqLen > 8192) return NOT_SUPPORT;
            std::unique_ptr<Tensor> importedKey(Tensor::create<float>({1, static_cast<int>(layer.tokens), mKvNumHead, mHeadDim}));
            std::unique_ptr<Tensor> importedValue(Tensor::create<float>({1, static_cast<int>(layer.tokens), mKvNumHead, mHeadDim}));
            if (!importedKey || !importedValue || !importedKey->host<float>() || !importedValue->host<float>()) return OUT_OF_MEMORY;
            if (mBytes == 2) {
                auto k = importedKey->host<FLOAT16_T>();
                auto v = importedValue->host<FLOAT16_T>();
                for (size_t i = 0; i < layer.count; ++i) {
                    k[i] = static_cast<FLOAT16_T>(layer.key[i]);
                    v[i] = static_cast<FLOAT16_T>(layer.value[i]);
                    if (!std::isfinite(static_cast<float>(k[i])) || !std::isfinite(static_cast<float>(v[i]))) return NOT_SUPPORT;
                }
            } else {
                ::memcpy(importedKey->host<float>(), layer.key, layer.count * sizeof(float));
                ::memcpy(importedValue->host<float>(), layer.value, layer.count * sizeof(float));
            }
            mKVCacheManager->onClear();
            mKVCacheManager->onAlloc(mMeta, static_cast<int>(layer.tokens) + seqLen);
            mKVCacheManager->onUpdateKV(importedKey.get(), importedValue.get(), static_cast<int>(layer.tokens));
            if (mKVCacheManager->kvLength() != layer.tokens) return NOT_SUPPORT;
            ++crossinferImportedLayers;
        }
"""

_HISTORY = r"""
bool Llm::crossinferSetImportedHistoryV1(int positions) {
    const char* flag = std::getenv("CROSSINFER_MNN_KV_TRANSFER");
    if (!flag || std::strcmp(flag, "1") != 0 || positions <= 0 || positions > 8192 ||
        mConfig->backend_type() != "cpu" || mConfig->is_mrope() || !mConfig->reuse_kv() ||
        mContext->status != LlmStatus::RUNNING || mContext->all_seq_len != 0 ||
        mContext->gen_seq_len != 0 || !mContext->history_tokens.empty() ||
        !mContext->output_tokens.empty() || mMeta->previous != mMeta->remove ||
        mMeta->n_reserve != 0 || mMeta->file_flag != KVMeta::NoChange) return false;
    mMeta->previous = static_cast<size_t>(positions);
    mMeta->remove = mMeta->add = 0;
    mMeta->reserve = nullptr;
    mMeta->n_reserve = 0;
    mContext->all_seq_len = positions;
    mContext->current_token = -1;
    return true;
}

extern "C" __attribute__((visibility("default"))) int mnn_crossinfer_import_history_v1(void* llm, int positions) {
    return llm && static_cast<Llm*>(llm)->crossinferSetImportedHistoryV1(positions) ? 1 : 0;
}
"""


def _replace_once(source: str, anchor: str, replacement: str, *, marker: str) -> str:
    if marker in source or source.count(anchor) != 1:
        raise ValueError("already patched, conflicting, or unsupported MNN source")
    return source.replace(anchor, replacement, 1)


def patch_llm_header(source: str) -> str:
    anchor = "    size_t getCurrentHistory() const;\n"
    return _replace_once(
        source,
        anchor,
        anchor + "    bool crossinferSetImportedHistoryV1(int positions);\n",
        marker="crossinferSetImportedHistoryV1",
    )


def patch_llm_source(source: str) -> str:
    anchor = "size_t Llm::getCurrentHistory() const {\n    return mMeta->previous;\n}\n"
    return _replace_once(source, anchor, anchor + _HISTORY, marker="crossinferSetImportedHistoryV1")


def patch_cpu_source(source: str) -> str:
    include = '#include "CPUAttention.hpp"\n'
    anchor = "    if (!mIsKVShared) {\n        if (mKVCache && mMeta != nullptr) {\n"
    if source.count(include) != 1:
        raise ValueError("unsupported CPUAttention include")
    patched = _replace_once(
        source,
        anchor,
        "    if (!mIsKVShared) {\n"
        + _CPU_IMPORT
        + "        if (mKVCache && mMeta != nullptr) {\n",
        marker="mnn_crossinfer_cpu_stage_kv_v1",
    )
    return patched.replace(include, include + _CPU_BRIDGE, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("cpu", "llm", "header"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        before = args.source.read_text()
        after = {
            "cpu": patch_cpu_source,
            "llm": patch_llm_source,
            "header": patch_llm_header,
        }[args.component](before)
        with args.output.open("x") as output:
            output.write(after)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(
        "".join(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile=str(args.source),
                tofile=str(args.output),
            )
        ),
        end="",
    )


if __name__ == "__main__":
    main()
