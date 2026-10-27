#!/bin/bash
# Build native components against the board's MNN libraries; no model preparation.
# Usage: build.sh [--qwen-positions] [source.cpp [output]]
# --qwen-positions enables the decoder sidecar for user-supplied prepared models.
# Links libllm + MNN_CL + Express + MNN; the worker needs retained-context exceptions.
set -e
B="${MNN_BUILD:?set MNN_BUILD to the MNN build directory}"
MNN_SRC="${MNN_SRC:?set MNN_SRC to the MNN source directory}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN_POSITIONS=0
if [[ "${1:-}" == "--qwen-positions" ]]; then
  QWEN_POSITIONS=1
  shift
fi
SRC="${1:-tabm_decoder_worker.cpp}"
if [[ "$SRC" != /* ]]; then
  SRC="$SCRIPT_DIR/$SRC"
fi
OUT="${2:-/tmp/$(basename "${SRC%.cpp}")}"
FLAGS=(-fno-exceptions)
INCLUDES=(-I"$MNN_SRC/include" -I"$MNN_SRC/transformers/llm/engine/include")
if [[ "$(basename "$SRC")" == "tabm_decoder_worker.cpp" ]]; then
  FLAGS=(-fexceptions -DCROSSINFER_RETAINED_KV)
elif [[ "$(basename "$SRC")" == "qwen_visual_worker.cpp" ]]; then
  FLAGS=(-fexceptions)
fi
if [[ "$QWEN_POSITIONS" == 1 ]]; then
  if [[ "$(basename "$SRC")" != "tabm_decoder_worker.cpp" ]]; then
    echo "--qwen-positions requires tabm_decoder_worker.cpp" >&2
    exit 2
  fi
  QWEN_NATIVE="$(cd "$SCRIPT_DIR/../../../../../QCS6490/edgeflow_qcs6490/native" && pwd)"
  FLAGS+=(-DCROSSINFER_QWEN_POSITIONS)
  INCLUDES+=(-I"$QWEN_NATIVE" -I"$MNN_SRC/source" -I"$MNN_SRC/transformers/llm/engine/src")
fi
cmake --build "$B" --target llm MNN MNN_Express MNN_CL --parallel "${MNN_BUILD_JOBS:-2}"
c++ -std=c++11 -O3 -fno-rtti "${FLAGS[@]}" \
  "${INCLUDES[@]}" \
  "$SRC" -o "$OUT" \
  -Wl,-rpath,"$B:$B/source/backend/opencl:$B/express" \
  "$B/libllm.so" "$B/source/backend/opencl/libMNN_CL.so" "$B/express/libMNN_Express.so" "$B/libMNN.so" \
  -pthread -ldl
echo "built -> $OUT"
