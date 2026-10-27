// Includes the real session functions; only llama storage operations are mocked.
// Build without models/backends using the command reported with this test.
#define EDGEFLOW_LLM_RUNTIME_SOURCE_FINGERPRINT_SHA256 \
  "0000000000000000000000000000000000000000000000000000000000000000"
#include "../src/edgeflow_llm.cpp"

#include <iostream>

namespace {

struct MockContext {
  std::map<llama_seq_id, std::vector<uint8_t>> bytes{{0, {1, 2, 3, 4}}};
  std::map<llama_seq_id, llama_pos> positions{{0, 10}};
  std::vector<char> operations;
  int export_failure = 0;
  int import_failure = 0;
};

MockContext &mock(llama_context *context) {
  return *reinterpret_cast<MockContext *>(context);
}

MockContext &mock(llama_memory_t memory) {
  return *reinterpret_cast<MockContext *>(memory);
}

void check(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

using Session = std::unique_ptr<edgeflow_llm_session,
                              decltype(&edgeflow_llm_session_free)>;
using Snapshot = std::unique_ptr<edgeflow_llm_root_snapshot,
                               decltype(&edgeflow_llm_root_snapshot_free)>;

Session make_session() {
  Session session(new edgeflow_llm_session, edgeflow_llm_session_free);
  session->context.reset(reinterpret_cast<llama_context *>(new MockContext));
  session->owner = std::this_thread::get_id();
  session->sequences.resize(4);
  auto &root = session->sequences.front();
  root.active = true;
  root.next_position = 12;
  root.visual_boundaries.emplace(
      4, SessionSequenceState::VisualBoundary{2, 6, 2, true});
  root.visual_boundaries.emplace(
      12, SessionSequenceState::VisualBoundary{10, 18, 10, true});
  const float logits[] = {1.0f, 4.0f};
  root.logits.assign(logits, 2);
  auto &signature = session->signature;
  signature.session_abi_version = EDGEFLOW_LLM_SESSION_ABI_VERSION;
  signature.n_ctx = 128;
  signature.n_batch = 32;
  signature.n_ubatch = 16;
  signature.n_seq_max = 4;
  signature.n_threads = 1;
  signature.kv_unified = 1;
  signature.model_family = EDGEFLOW_MODEL_FAMILY_QWEN25VL;
  signature.position_contract = EDGEFLOW_POSITION_CONTRACT_QWEN25VL_MROPE;
  signature.prompt_contract_version = 1;
  std::memset(signature.model_sha256, 'a', 64);
  std::memset(signature.runtime_sha256, 'b', 64);
  return session;
}

Snapshot export_root(edgeflow_llm_session *session) {
  edgeflow_llm_root_snapshot *raw = nullptr;
  check(edgeflow_llm_session_export_root(session, &raw) ==
            EDGEFLOW_LLM_SESSION_OK,
        "export failed");
  return Snapshot(raw, edgeflow_llm_root_snapshot_free);
}

void ownership_and_restore() {
  auto donor = make_session();
  donor->sequences[1].active = true;
  mock(donor->context.get()).bytes[1] = {9, 9};
  auto snapshot = export_root(donor.get());
  check(snapshot->root.logits.values().empty(), "snapshot retained logits");
  check(edgeflow_llm_root_snapshot_size(snapshot.get()) == 4,
        "wrong serialized size");
  check(mock(donor->context.get()).operations == std::vector<char>({'S', 'G'}),
        "export did not synchronize before serializing");
  // Simulate a subsequent donor append, including changed boundary metadata.
  donor->sequences[0].next_position = 20;
  donor->sequences[0].visual_boundaries.at(12).token_count = 99;
  mock(donor->context.get()).bytes[0].push_back(5);
  donor.reset();

  auto receiver = make_session();
  receiver->signature.requested_gpu_layers = 99;
  receiver->signature.observed_gpu_layers = 33;
  receiver->signature.backend_placement = EDGEFLOW_BACKEND_PLACEMENT_GPU_FULL;
  receiver->signature.n_threads = 8;
  receiver->signature.n_batch = 64;
  receiver->signature.n_ubatch = 32;
  receiver->signature.n_seq_max = 8;
  receiver->signature.runtime_source_fingerprint_sha256[0] = 'x';
  check(edgeflow_llm_session_import_root(receiver.get(), snapshot.get()) == 0,
        "compatible placement/thread/batch change was rejected");
  auto &context = mock(receiver->context.get());
  check(context.operations == std::vector<char>({'S', 'C', 'I', 'S'}),
        "import synchronization/clear ordering is wrong");
  check(context.bytes.at(0) == std::vector<uint8_t>({1, 2, 3, 4}),
        "snapshot bytes depended on donor lifetime or branches");
  edgeflow_llm_sequence_info info{};
  check(edgeflow_llm_session_get_sequence_info(receiver.get(), 0, &info) == 0,
        "sequence info failed");
  check(info.active && info.next_position == 12 && !info.has_logits,
        "import did not restore position with invalid logits");
  const auto &boundary = receiver->sequences[0].visual_boundaries.at(12);
  check(boundary.kv_position_max == 10 && boundary.token_count == 18 &&
            boundary.visual_start == 10 && boundary.forkable,
        "visual boundary metadata changed");
  check(receiver->signature.n_threads == 8,
        "import overwrote the receiver signature");
  snapshot.reset();
  check(edgeflow_llm_session_fork_prefix(receiver.get(), 0, 1, 4) ==
            EDGEFLOW_LLM_SESSION_INVALID_STATE,
        "import enabled historical multi-image forks");
  check(edgeflow_llm_session_fork_prefix(receiver.get(), 0, 1, 12) == 0,
        "restored latest-head visual fork failed");
  check(receiver->sequences[1].logits.values().empty(),
        "restored fork exposed donor logits");
}

void contracts_and_branches() {
  auto donor = make_session();
  auto snapshot = export_root(donor.get());
  auto receiver = make_session();
  const auto original = receiver->signature;
  using Mutator = void (*)(edgeflow_llm_session_signature &);
  const Mutator changes[] = {
      [](auto &s) { ++s.session_abi_version; },
      [](auto &s) { ++s.n_ctx; },
      [](auto &s) { s.kv_unified = 0; },
      [](auto &s) { s.model_family = EDGEFLOW_MODEL_FAMILY_FASTVLM; },
      [](auto &s) { s.kv_type_k = static_cast<edgeflow_kv_dtype>(1); },
      [](auto &s) { s.kv_type_v = static_cast<edgeflow_kv_dtype>(1); },
      [](auto &s) { s.position_contract = EDGEFLOW_POSITION_CONTRACT_TEXT_1D; },
      [](auto &s) { ++s.prompt_contract_version; },
      [](auto &s) { s.model_sha256[0] = 'c'; },
      [](auto &s) { s.runtime_sha256[0] = 'c'; },
  };
  for (const auto change : changes) {
    receiver->signature = original;
    change(receiver->signature);
    check(edgeflow_llm_session_import_root(receiver.get(), snapshot.get()) ==
              EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
          "incompatible signature was accepted");
    check(mock(receiver->context.get()).operations.empty() &&
              receiver->sequences[0].logits.values().size() == 2 &&
              !receiver->poisoned,
          "signature rejection mutated the receiver");
  }
  receiver->signature = original;
  receiver->sequences[1].active = true;
  check(edgeflow_llm_session_import_root(receiver.get(), snapshot.get()) ==
            EDGEFLOW_LLM_SESSION_INVALID_STATE,
        "import accepted a live branch");
  check(edgeflow_llm_session_reset_root(receiver.get()) ==
            EDGEFLOW_LLM_SESSION_INVALID_STATE,
        "reset accepted a live branch");
  check(mock(receiver->context.get()).operations.empty(),
        "live-branch rejection cleared or synchronized the receiver");
  check(edgeflow_llm_session_remove_sequence(receiver.get(), 1) == 0,
        "branch release failed");
  check(edgeflow_llm_session_import_root(receiver.get(), snapshot.get()) == 0,
        "import after branch release failed");
}

void reset_and_text() {
  auto session = make_session();
  const auto *context = session->context.get();
  const auto signature = session->signature;
  session->sequences[2] = session->sequences[0];
  session->sequences[2].active = false;
  mock(session->context.get()).bytes[2] = {9};
  check(edgeflow_llm_session_reset_root(session.get()) == 0, "reset failed");
  check(session->context.get() == context &&
            std::memcmp(&signature, &session->signature, sizeof(signature)) == 0,
        "reset replaced context or signature");
  check(mock(session->context.get()).bytes.empty(), "reset retained KV");
  for (size_t i = 0; i < session->sequences.size(); ++i) {
    const auto &state = session->sequences[i];
    check(state.active == (i == 0) && state.next_position == 0 &&
              state.visual_boundaries.empty() && state.logits.values().empty(),
          "reset retained sequence metadata");
  }
  auto empty_snapshot = export_root(session.get());
  check(edgeflow_llm_session_import_root(session.get(), empty_snapshot.get()) ==
            0,
        "empty root did not roundtrip");
  session->signature.position_contract = EDGEFLOW_POSITION_CONTRACT_TEXT_1D;
  session->sequences[0].next_position = 4;
  mock(session->context.get()).bytes[0] = {1, 2, 3, 4};
  mock(session->context.get()).positions[0] = 3;
  auto snapshot = export_root(session.get());
  check(edgeflow_llm_session_reset_root(session.get()) == 0, "text reset failed");
  check(edgeflow_llm_session_import_root(session.get(), snapshot.get()) == 0,
        "text root import failed");
  check(edgeflow_llm_session_fork_prefix(session.get(), 0, 1, 4) == 0,
        "text root fork failed");
}

void errors_and_guards() {
  auto donor = make_session();
  auto snapshot = export_root(donor.get());
  check(edgeflow_llm_root_snapshot_size(nullptr) == 0, "NULL size is not zero");
  edgeflow_llm_root_snapshot_free(nullptr);
  check(edgeflow_llm_session_export_root(donor.get(), nullptr) ==
            EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
        "NULL export output accepted");
  check(edgeflow_llm_session_import_root(donor.get(), nullptr) ==
            EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
        "NULL import accepted");
  check(edgeflow_llm_session_reset_root(nullptr) ==
            EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
        "NULL session accepted");
  for (int failure : {1, 2, 3}) {
    mock(donor->context.get()).export_failure = failure;
    edgeflow_llm_root_snapshot *raw = snapshot.get();
    check(edgeflow_llm_session_export_root(donor.get(), &raw) ==
              EDGEFLOW_LLM_SESSION_RUNTIME_ERROR &&
              raw == nullptr && !donor->poisoned,
          "export failure published a snapshot or poisoned the donor");
  }
  for (int failure : {1, 2, 3}) {
    auto receiver = make_session();
    mock(receiver->context.get()).import_failure = failure;
    check(edgeflow_llm_session_import_root(receiver.get(), snapshot.get()) ==
              EDGEFLOW_LLM_SESSION_RUNTIME_ERROR && receiver->poisoned,
          "partial import failure was not poisoned");
    check(edgeflow_llm_session_reset_root(receiver.get()) ==
              EDGEFLOW_LLM_SESSION_INVALID_STATE,
          "reset revived a poisoned session");
    check(edgeflow_llm_session_import_root(receiver.get(), snapshot.get()) ==
              EDGEFLOW_LLM_SESSION_INVALID_STATE,
          "import revived a poisoned session");
  }
  auto receiver = make_session();
  std::thread other([&] {
    edgeflow_llm_root_snapshot *raw = nullptr;
    int token = -1, eog = -1;
    check(edgeflow_llm_session_export_root(receiver.get(), &raw) ==
              EDGEFLOW_LLM_SESSION_WRONG_THREAD,
          "wrong-thread export accepted");
    check(edgeflow_llm_session_import_root(receiver.get(), snapshot.get()) ==
              EDGEFLOW_LLM_SESSION_WRONG_THREAD,
          "wrong-thread import accepted");
    check(edgeflow_llm_session_reset_root(receiver.get()) ==
              EDGEFLOW_LLM_SESSION_WRONG_THREAD,
          "wrong-thread reset accepted");
    check(edgeflow_llm_session_peek_token(receiver.get(), 0, &token, &eog) ==
              EDGEFLOW_LLM_SESSION_WRONG_THREAD,
          "wrong-thread peek accepted");
  });
  other.join();
  check(mock(receiver->context.get()).operations.empty(),
        "wrong-thread calls touched KV");
}

void peek_without_decode() {
  auto session = make_session();
  int token = -1, eog = -1;
  for (int i = 0; i < 2; ++i) {
    check(edgeflow_llm_session_peek_token(session.get(), 0, &token, &eog) == 0 &&
              token == 1 && eog == 0,
          "peek did not reuse the cached greedy token");
  }
  const float logits[] = {5.0f, 1.0f};
  session->sequences[0].logits.assign(logits, 2);
  check(edgeflow_llm_session_peek_token(session.get(), 0, &token, &eog) == 0 &&
            token == 0 && eog == 1,
        "peek missed updated logits or EOG");
  check(session->sequences[0].next_position == 12 &&
            mock(session->context.get()).operations.empty(),
        "peek decoded, synchronized, or advanced the sequence");
  check(edgeflow_llm_session_peek_token(session.get(), 1, &token, &eog) ==
            EDGEFLOW_LLM_SESSION_INVALID_STATE,
        "peek accepted an inactive branch");
  check(edgeflow_llm_session_peek_token(session.get(), -1, &token, &eog) ==
            EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
        "peek accepted an invalid sequence");
  check(edgeflow_llm_session_peek_token(session.get(), 0, nullptr, &eog) ==
            EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT,
        "peek accepted a NULL output");
  const float bad[] = {std::numeric_limits<float>::quiet_NaN(), 1.0f};
  session->sequences[0].logits.assign(bad, 2);
  check(edgeflow_llm_session_peek_token(session.get(), 0, &token, &eog) ==
            EDGEFLOW_LLM_SESSION_RUNTIME_ERROR && !session->poisoned,
        "read-only nonfinite peek poisoned the session or succeeded");
  auto snapshot = export_root(session.get());
  check(edgeflow_llm_session_import_root(session.get(), snapshot.get()) == 0,
        "peek test import failed");
  check(edgeflow_llm_session_peek_token(session.get(), 0, &token, &eog) ==
            EDGEFLOW_LLM_SESSION_INVALID_STATE,
        "peek accepted invalid imported-root logits");
}

} // namespace

extern "C" void llama_model_free(llama_model *) {}
extern "C" const llama_vocab *llama_model_get_vocab(const llama_model *) {
  return nullptr;
}
extern "C" bool llama_vocab_is_eog(const llama_vocab *, llama_token token) {
  return token == 0;
}
extern "C" void llama_free(llama_context *context) { delete &mock(context); }
extern "C" void llama_synchronize(llama_context *context) {
  mock(context).operations.push_back('S');
}
extern "C" llama_memory_t llama_get_memory(const llama_context *context) {
  return reinterpret_cast<llama_memory_t>(const_cast<llama_context *>(context));
}
extern "C" void llama_memory_clear(llama_memory_t memory, bool data) {
  check(data, "clear must erase KV data, not only metadata");
  auto &context = mock(memory);
  context.operations.push_back('C');
  context.bytes.clear();
  context.positions.clear();
}
extern "C" size_t llama_state_seq_get_size(llama_context *context,
                                          llama_seq_id sequence) {
  auto &state = mock(context);
  if (state.export_failure == 1) {
    return 0;
  }
  // An empty sequence still has a serialized header.
  return state.bytes.count(sequence) ? state.bytes.at(sequence).size() : 1;
}
extern "C" size_t llama_state_seq_get_data(llama_context *context, uint8_t *out,
                                          size_t size, llama_seq_id sequence) {
  auto &state = mock(context);
  state.operations.push_back('G');
  if (state.export_failure == 2) {
    return size - 1;
  }
  if (state.export_failure == 3) {
    throw std::runtime_error("mock export error");
  }
  if (state.bytes.count(sequence)) {
    std::copy(state.bytes.at(sequence).begin(), state.bytes.at(sequence).end(),
              out);
  } else {
    *out = 0;
  }
  return size;
}
extern "C" size_t llama_state_seq_set_data(llama_context *context,
                                          const uint8_t *bytes, size_t size,
                                          llama_seq_id sequence) {
  auto &state = mock(context);
  state.operations.push_back('I');
  state.bytes[sequence].assign(bytes, bytes + size);
  state.positions[sequence] = 10;
  if (state.import_failure == 1) {
    return 0;
  }
  if (state.import_failure == 2) {
    return size - 1;
  }
  if (state.import_failure == 3) {
    throw std::runtime_error("mock partial import error");
  }
  return size;
}
extern "C" void llama_memory_seq_cp(llama_memory_t memory, llama_seq_id source,
                                    llama_seq_id destination, llama_pos,
                                    llama_pos end) {
  auto &state = mock(memory);
  state.bytes[destination] = state.bytes.at(source);
  state.positions[destination] = end == 4 ? 3 : state.positions.at(source);
}
extern "C" llama_pos llama_memory_seq_pos_max(llama_memory_t memory,
                                             llama_seq_id sequence) {
  return mock(memory).positions.at(sequence);
}
extern "C" bool llama_memory_seq_rm(llama_memory_t memory, llama_seq_id sequence,
                                    llama_pos, llama_pos) {
  mock(memory).bytes.erase(sequence);
  mock(memory).positions.erase(sequence);
  return true;
}

int main() {
  try {
    ownership_and_restore();
    contracts_and_branches();
    reset_and_text();
    errors_and_guards();
    peek_without_decode();
    std::cout << "PASS: native root snapshots/peek (5 functional groups)\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
