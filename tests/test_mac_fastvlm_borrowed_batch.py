"""Prove the native embedding batch aliases its input, without a model."""

from pathlib import Path
import subprocess


def test_native_batch_uses_original_storage_and_preserves_positions(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "borrowed.cpp"
    source.write_text('''
#include "mac_fastvlm_borrowed_batch.h"
#include <cassert>
int main() {
    float storage[] = {1, 2, 3, 4, 5, 6};
    BorrowedBatch batch(nullptr, storage + 3, 1, 17, false);
    assert(batch.value.embd == storage + 3);
    assert(batch.value.token == nullptr);
    assert(batch.value.pos[0] == 17);
    assert(batch.value.n_seq_id[0] == 1);
    assert(batch.value.seq_id[0][0] == 0);
    storage[4] = 42;
    assert(batch.value.embd[1] == 42);
    int32_t tokens[] = {10, 20};
    BorrowedBatch text(tokens, nullptr, 2, 18, true);
    assert(text.value.token == tokens);
    assert(text.value.embd == nullptr);
    assert(text.value.pos[1] == 19);
    assert(text.value.logits[0] == 0 && text.value.logits[1] == 1);
}
''')
    binary = tmp_path / "borrowed"
    subprocess.run(["clang++", "-std=c++17", "-O2", str(source),
                    "-I", str(root / "tools"),
                    "-I", str(root / "tools/edgeflow_llm_runtime/llama_src/include"),
                    "-I", str(root / "tools/edgeflow_llm_runtime/llama_src/ggml/include"),
                    "-o", str(binary)], check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True)
