// Real session code, with storage calls replaced; no model or backend is loaded.
#define EDGEFLOW_LLM_RUNTIME_SOURCE_FINGERPRINT_SHA256 \
  "0000000000000000000000000000000000000000000000000000000000000000"
#include "../tools/edgeflow_llm_runtime/src/edgeflow_llm.cpp"

#include <iostream>

namespace {
bool hybrid = true;
bool recurrent = false;
int copies = 0;
llama_pos copied_position = -1;
std::vector<llama_pos> positions;
float *embedding_pointer = nullptr;
void check(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}
}

extern "C" bool llama_model_is_recurrent(const llama_model *) { return recurrent; }
extern "C" bool llama_model_is_hybrid(const llama_model *) { return hybrid; }
extern "C" void llama_model_free(llama_model *) {}
extern "C" void llama_free(llama_context *) {}
extern "C" llama_memory_t llama_get_memory(const llama_context *) { return nullptr; }
extern "C" void llama_memory_seq_cp(llama_memory_t, llama_seq_id, llama_seq_id,
                                    llama_pos, llama_pos end) {
  ++copies;
  copied_position = end < 0 ? 9 : end - 1;
}
extern "C" bool llama_memory_seq_rm(llama_memory_t, llama_seq_id, llama_pos,
                                    llama_pos) { return true; }
extern "C" llama_pos llama_memory_seq_pos_max(llama_memory_t, llama_seq_id) {
  return copied_position;
}
extern "C" uint32_t llama_n_batch(const llama_context *) { return 512; }
extern "C" int32_t llama_decode(llama_context *, llama_batch batch) {
  check(batch.token == nullptr, "embedding test received token IDs");
  embedding_pointer = batch.embd;
  positions.assign(batch.pos, batch.pos + 4 * batch.n_tokens);
  for (int i = 0; i < batch.n_tokens; ++i) {
    check(batch.n_seq_id[i] == 1 && batch.seq_id[i][0] == 0,
          "wrong sequence metadata");
  }
  return 0;
}

int main(int argc, char **argv) {
  try {
    check(argc == 2, "one case required");
    edgeflow_llm_session session;
    session.owner = std::this_thread::get_id();
    session.model.reset(reinterpret_cast<llama_model *>(1));
    session.context.reset(reinterpret_cast<llama_context *>(1));
    session.sequences.resize(3);
    session.sequences[0].active = true;
    session.sequences[0].next_position = 10;
    const std::string test = argv[1];
    if (test == "historical" || test == "recurrent") {
      if (test == "recurrent") { hybrid = false; recurrent = true; }
      const int rc = edgeflow_llm_session_fork_prefix(&session, 0, 1, 5);
      check(rc == EDGEFLOW_LLM_SESSION_INVALID_STATE,
            "hybrid historical fork must fail before copying current state");
      check(copies == 0 && !session.sequences[1].active,
            "rejected historical fork mutated storage");
    } else if (test == "current") {
      check(edgeflow_llm_session_fork_prefix(&session, 0, 1, 10) == 0,
            "full current root fork failed");
      check(copies == 1 && session.sequences[1].next_position == 10,
            "current root was not published");
    } else if (test == "ordinary") {
      hybrid = false;
      check(edgeflow_llm_session_fork_prefix(&session, 0, 1, 5) == 0,
            "nonrecurrent historical text fork regressed");
    } else if (test == "branch-source") {
      session.sequences[1].active = true;
      session.sequences[1].next_position = 10;
      check(edgeflow_llm_session_fork_prefix(&session, 1, 2, 10) ==
                EDGEFLOW_LLM_SESSION_INVALID_STATE,
            "hybrid branch-source fork must fail");
      check(copies == 0, "branch-source fork mutated storage");
    } else if (test == "positions") {
      float embeddings[144 * 8] = {};
      uint64_t borrowed = 0;
      decode_embedding_batch_2d_sequence(session.context.get(), embeddings, 144,
                                          8, 7, 12, 12, 0, &borrowed);
      check(embedding_pointer == embeddings && borrowed == uintptr_t(embeddings),
            "embedding input was copied");
      for (int i = 0; i < 144; ++i) {
        check(positions[i] == 7 && positions[144 + i] == 7 + i / 12 &&
                  positions[288 + i] == 7 + i % 12 && positions[432 + i] == 0,
              "incorrect four-plane image M-RoPE layout");
      }
    } else if (test == "capacity") {
      struct OldCache { uint64_t get_used_cells() const { return 123; } } old;
      struct NewCache {} newer;
      check(cache_used_cells(&old, 0) == 123, "exact occupancy capability lost");
      check(cache_used_cells(&newer, 0) == std::numeric_limits<uint64_t>::max(),
            "new upstream must fall back without get_used_cells");
      auto &root = session.sequences[0];
      root.next_position = 100;
      root.visual_boundaries.emplace(100, SessionSequenceState::VisualBoundary{88, 1200, 88, true});
      session.sequences[1] = root;
      session.sequences[1].next_position = 110;
      check(conservative_session_cells(&session) == 2410,
            "capacity confused M-RoPE extent with occupied cells");
      session.sequences[1] = SessionSequenceState{};
      check(conservative_session_cells(&session) == 1200,
            "whole branch removal did not release conservative admission budget");
    } else {
      throw std::runtime_error("unknown case");
    }
    std::cout << "PASS\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
