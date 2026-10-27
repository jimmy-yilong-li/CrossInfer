// Exercise the production lease callback with no IOSurface/device allocation.
#include "../tools/edgeflow_llm_runtime/src/edgeflow_llm_ane_glue.mm"
#include <iostream>
#include <stdexcept>

namespace {
float payload[144 * 8] = {};
size_t available = sizeof(payload);
bool mapped = false;
int calls = 0;
int result = 0;
bool alias = true;
void check(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}
}

extern "C" int ane_bridge_consume_surface_lease(
    ANESurfaceLeaseHandle *, ane_bridge_output_consumer_fn consumer, void *context,
    char *error, size_t cap) {
  mapped = true;
  const int rc = consumer(payload, available, context, error, cap);
  mapped = false;
  return rc;
}

extern "C" int edgeflow_llm_session_append_qwen35_visual(
    edgeflow_llm_session *, int sequence_id, const edgeflow_llm_request *request,
    edgeflow_llm_surface_append_receipt *receipt) {
  ++calls;
  check(mapped, "consumer escaped mapped lease scope");
  check(sequence_id == 0 && request->visual_embd == payload,
        "callback copied or changed the visual input");
  receipt->decoder_pointer = alias ? uintptr_t(request->visual_embd) : 1;
  receipt->payload_bytes = sizeof(payload);
  receipt->next_position = 17;
  return result;
}

int main(int argc, char **argv) {
  try {
    check(argc == 2, "case required");
    const std::string test = argv[1];
    if (test == "short") available = sizeof(payload) - 1;
    if (test == "failure") result = EDGEFLOW_LLM_SESSION_INVALID_STATE;
    if (test == "alias") alias = false;
    edgeflow_llm_request request{};
    request.num_visual_tokens = 144;
    request.hidden_size = 8;
    edgeflow_llm_surface_append_receipt receipt{};
    const int rc = edgeflow_llm_session_append_qwen35_visual_from_surface_lease(
        reinterpret_cast<edgeflow_llm_session *>(1), 0, &request,
        reinterpret_cast<ANESurfaceLeaseHandle *>(1), &receipt);
    check(!mapped, "lease scope was not closed");
    if (test == "success") {
      check(rc == 0 && calls == 1 && receipt.surface_pointer == uintptr_t(payload) &&
                receipt.decoder_pointer == receipt.surface_pointer &&
                receipt.payload_bytes == sizeof(payload), "invalid borrowed receipt");
    } else if (test == "short") {
      check(rc == EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT && calls == 0,
            "undersized surface reached decoder");
    } else if (test == "failure") {
      check(rc == result && calls == 1, "native failure status was lost");
    } else if (test == "alias") {
      check(rc == EDGEFLOW_LLM_SESSION_RUNTIME_ERROR, "nonaliasing decoder was accepted");
    } else {
      throw std::runtime_error("unknown case");
    }
    std::cout << "PASS\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
