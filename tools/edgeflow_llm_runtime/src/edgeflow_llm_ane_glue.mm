#include "edgeflow_llm/edgeflow_llm.h"

#include "ane_bridge.h"

#include <cstdio>
#include <cstring>

namespace {

struct surface_generate_ctx {
  edgeflow_llm_ctx * ctx;
  const edgeflow_llm_request * req_template;      // v1 path (NULL for v2)
  const edgeflow_llm_request_v2 * req_v2;         // v2 path (NULL for v1)
  edgeflow_llm_result * out;
  int generate_rc;
  int prefix_tokens = 0;
  int suffix_tokens = 0;
};

void write_error(char * error_out, const size_t error_out_cap, const char * message) {
  if (error_out == nullptr || error_out_cap == 0) {
    return;
  }
  std::snprintf(error_out, error_out_cap, "%s", message != nullptr ? message : "unknown error");
}

int validate_callback_inputs(
    const void * base_address,
    const size_t nbytes,
    surface_generate_ctx * gctx,
    char * error_out,
    const size_t error_out_cap) {
  if (gctx == nullptr || gctx->ctx == nullptr || gctx->req_template == nullptr || gctx->out == nullptr) {
    write_error(error_out, error_out_cap, "surface generate context is invalid");
    return 2;
  }
  if (base_address == nullptr) {
    write_error(error_out, error_out_cap, "surface callback received null base_address");
    return 3;
  }

  const edgeflow_llm_request * req = gctx->req_template;
  if (req->num_visual_tokens < 0 || req->hidden_size < 0) {
    write_error(error_out, error_out_cap, "surface callback received invalid visual shape");
    return 4;
  }

  const size_t expected_bytes = static_cast<size_t>(req->num_visual_tokens) *
      static_cast<size_t>(req->hidden_size) * sizeof(float);
  if (expected_bytes > nbytes) {
    write_error(error_out, error_out_cap, "surface payload is smaller than requested visual embedding size");
    return 5;
  }

  return 0;
}

int surface_consume_callback(
    const void * base_address,
    const size_t nbytes,
    void * user_ctx,
    char * error_out,
    const size_t error_out_cap) {
  auto * gctx = static_cast<surface_generate_ctx *>(user_ctx);
  const int validation_rc = validate_callback_inputs(base_address, nbytes, gctx, error_out, error_out_cap);
  if (validation_rc != 0) {
    return validation_rc;
  }

  edgeflow_llm_request local_req = *gctx->req_template;
  local_req.visual_embd = static_cast<const float *>(base_address);

  @autoreleasepool {
    if (gctx->prefix_tokens || gctx->suffix_tokens) {
      const uint64_t rows = uint64_t(local_req.num_visual_tokens) + gctx->prefix_tokens + gctx->suffix_tokens;
      if (gctx->prefix_tokens <= 0 || gctx->suffix_tokens <= 0 ||
          rows * uint64_t(local_req.hidden_size) * sizeof(float) > nbytes) {
        write_error(error_out, error_out_cap, "packed prompt exceeds leased surface");
        return 5;
      }
      local_req.visual_embd += size_t(gctx->prefix_tokens) * local_req.hidden_size;
      gctx->generate_rc = edgeflow_llm_generate_packed(
          gctx->ctx, &local_req, static_cast<const float *>(base_address),
          gctx->prefix_tokens, gctx->suffix_tokens, gctx->out);
    } else {
      gctx->generate_rc = edgeflow_llm_generate(gctx->ctx, &local_req, gctx->out);
    }
  }
  if (gctx->generate_rc != 0) {
    std::snprintf(
        error_out,
        error_out_cap,
        "edgeflow_llm_generate failed rc=%d",
        gctx->generate_rc);
    return gctx->generate_rc;
  }

  return 0;
}

template <typename ConsumeFn>
int generate_from_mapped_surface(
    edgeflow_llm_ctx * ctx,
    const edgeflow_llm_request * req,
    edgeflow_llm_result * out,
    ConsumeFn consume_fn) {
  if (ctx == nullptr || req == nullptr || out == nullptr) {
    return 1;
  }

  surface_generate_ctx gctx = {
      .ctx = ctx,
      .req_template = req,
      .req_v2 = nullptr,
      .out = out,
      .generate_rc = 0,
  };

  char error_buf[256] = {0};
  const int bridge_rc = consume_fn(
      surface_consume_callback,
      &gctx,
      error_buf,
      sizeof(error_buf));
  if (bridge_rc != 0) {
    return bridge_rc;
  }
  return gctx.generate_rc;
}

int validate_callback_inputs_v2(
    const void * base_address,
    const size_t nbytes,
    surface_generate_ctx * gctx,
    char * error_out,
    const size_t error_out_cap) {
  if (gctx == nullptr || gctx->ctx == nullptr || gctx->req_v2 == nullptr || gctx->out == nullptr) {
    write_error(error_out, error_out_cap, "surface generate context is invalid");
    return 2;
  }
  if (base_address == nullptr) {
    write_error(error_out, error_out_cap, "surface callback received null base_address");
    return 3;
  }

  const edgeflow_llm_request_v2 * req = gctx->req_v2;
  if (req->num_visual_tokens < 0 || req->hidden_size < 0) {
    write_error(error_out, error_out_cap, "surface callback received invalid visual shape");
    return 4;
  }

  const size_t element_size = (req->visual_dtype == EDGEFLOW_DTYPE_FP16)
      ? sizeof(uint16_t)
      : sizeof(float);
  const size_t expected_bytes = static_cast<size_t>(req->num_visual_tokens) *
      static_cast<size_t>(req->hidden_size) * element_size;
  if (expected_bytes > nbytes) {
    write_error(error_out, error_out_cap, "surface payload is smaller than requested visual embedding size");
    return 5;
  }

  return 0;
}

int surface_consume_callback_v2(
    const void * base_address,
    const size_t nbytes,
    void * user_ctx,
    char * error_out,
    const size_t error_out_cap) {
  auto * gctx = static_cast<surface_generate_ctx *>(user_ctx);
  const int validation_rc = validate_callback_inputs_v2(base_address, nbytes, gctx, error_out, error_out_cap);
  if (validation_rc != 0) {
    return validation_rc;
  }

  edgeflow_llm_request_v2 local_req = *gctx->req_v2;
  local_req.visual_embd = base_address;

  @autoreleasepool {
    gctx->generate_rc = edgeflow_llm_generate_v2(gctx->ctx, &local_req, gctx->out);
  }
  if (gctx->generate_rc != 0) {
    std::snprintf(
        error_out,
        error_out_cap,
        "edgeflow_llm_generate_v2 failed rc=%d",
        gctx->generate_rc);
    return gctx->generate_rc;
  }

  return 0;
}

template <typename ConsumeFn>
int generate_from_mapped_surface_v2(
    edgeflow_llm_ctx * ctx,
    const edgeflow_llm_request_v2 * req,
    edgeflow_llm_result * out,
    ConsumeFn consume_fn) {
  if (ctx == nullptr || req == nullptr || out == nullptr) {
    return 1;
  }

  surface_generate_ctx gctx = {
      .ctx = ctx,
      .req_template = nullptr,
      .req_v2 = req,
      .out = out,
      .generate_rc = 0,
  };

  char error_buf[256] = {0};
  const int bridge_rc = consume_fn(
      surface_consume_callback_v2,
      &gctx,
      error_buf,
      sizeof(error_buf));
  if (bridge_rc != 0) {
    return bridge_rc;
  }
  return gctx.generate_rc;
}

}  // namespace

extern "C" int edgeflow_llm_generate_from_ane_surface(
    edgeflow_llm_ctx * ctx,
    const edgeflow_llm_request * req,
    ANEKernelHandle * kernel,
    const int output_idx,
    edgeflow_llm_result * out) {
  return generate_from_mapped_surface(
      ctx,
      req,
      out,
      [&](ane_bridge_output_consumer_fn consumer,
          void * user_ctx,
          char * error_out,
          const size_t error_out_cap) {
        return ane_bridge_consume_output_surface(
            kernel,
            output_idx,
            consumer,
            user_ctx,
            error_out,
            error_out_cap);
      });
}

extern "C" int edgeflow_llm_generate_from_surface_lease(
    edgeflow_llm_ctx * ctx,
    const edgeflow_llm_request * req,
    ANESurfaceLeaseHandle * lease,
    edgeflow_llm_result * out) {
  return generate_from_mapped_surface(
      ctx,
      req,
      out,
      [&](ane_bridge_output_consumer_fn consumer,
          void * user_ctx,
          char * error_out,
          const size_t error_out_cap) {
        return ane_bridge_consume_surface_lease(
            lease,
            consumer,
            user_ctx,
            error_out,
            error_out_cap);
      });
}

extern "C" int edgeflow_llm_generate_from_surface_lease_packed(
    edgeflow_llm_ctx *ctx, const edgeflow_llm_request *req, ANESurfaceLeaseHandle *lease,
    int prefix, int suffix, edgeflow_llm_result *out) {
  if (!ctx || !req || !lease || !out || prefix <= 0 || suffix <= 0) return 1;
  surface_generate_ctx call{ctx, req, nullptr, out, 0, prefix, suffix};
  char error[512] = {};
  const int status = ane_bridge_consume_surface_lease(
      lease, surface_consume_callback, &call, error, sizeof(error));
  if (status != 0) {
    std::fprintf(stderr, "packed surface generation failed: %s\n", error);
    return status;
  }
  return call.generate_rc;
}

extern "C" int edgeflow_llm_generate_from_surface_lease_v2(
    edgeflow_llm_ctx * ctx,
    const edgeflow_llm_request_v2 * req,
    ANESurfaceLeaseHandle * lease,
    edgeflow_llm_result * out) {
  return generate_from_mapped_surface_v2(
      ctx,
      req,
      out,
      [&](ane_bridge_output_consumer_fn consumer,
          void * user_ctx,
          char * error_out,
          const size_t error_out_cap) {
        return ane_bridge_consume_surface_lease(
            lease,
            consumer,
            user_ctx,
            error_out,
            error_out_cap);
      });
}

static int append_session_from_surface_lease(
    edgeflow_llm_session *session, int sequence_id,
    const edgeflow_llm_request *request, ANESurfaceLeaseHandle *lease,
    edgeflow_llm_surface_append_receipt *receipt,
    int (*consume)(edgeflow_llm_session *, int, const edgeflow_llm_request *,
                   edgeflow_llm_surface_append_receipt *)) {
  if (receipt != nullptr) *receipt = {};
  if (!session || !request || !lease || !receipt ||
      request->num_visual_tokens <= 0 || request->hidden_size <= 0) {
    return EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT;
  }
  struct Append {
    edgeflow_llm_session *session;
    int sequence_id;
    const edgeflow_llm_request *request;
    edgeflow_llm_surface_append_receipt *receipt;
    decltype(consume) consume;
    int status = EDGEFLOW_LLM_SESSION_RUNTIME_ERROR;
  } append{session, sequence_id, request, receipt, consume};
  char error[256] = {};
  const int bridge_rc = ane_bridge_consume_surface_lease(
      lease,
      [](const void *base, size_t size, void *context, char *error, size_t cap) -> int {
        auto &a = *static_cast<Append *>(context);
        const uint64_t bytes = uint64_t(a.request->num_visual_tokens) *
                               a.request->hidden_size * sizeof(float);
        if (!base || bytes > size || reinterpret_cast<uintptr_t>(base) % alignof(float)) {
          write_error(error, cap, "invalid FP32 session surface bounds/alignment");
          a.status = EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT;
          return a.status;
        }
        auto request = *a.request;
        request.visual_embd = static_cast<const float *>(base);
        @autoreleasepool {
          a.status = a.consume(
              a.session, a.sequence_id, &request, a.receipt);
        }
        a.receipt->surface_pointer = reinterpret_cast<uintptr_t>(base);
        if (a.status == 0 && a.receipt->decoder_pointer != a.receipt->surface_pointer) {
          write_error(error, cap, "session did not borrow the mapped surface");
          a.status = EDGEFLOW_LLM_SESSION_RUNTIME_ERROR;
        }
        return a.status;
      },
      &append, error, sizeof(error));
  return bridge_rc == 0 ? append.status
                       : (append.status ? append.status : EDGEFLOW_LLM_SESSION_RUNTIME_ERROR);
}

extern "C" int edgeflow_llm_session_append_qwen35_visual_from_surface_lease(
    edgeflow_llm_session *session, int sequence_id,
    const edgeflow_llm_request *request, ANESurfaceLeaseHandle *lease,
    edgeflow_llm_surface_append_receipt *receipt) {
  return append_session_from_surface_lease(session, sequence_id, request, lease,
      receipt, edgeflow_llm_session_append_qwen35_visual);
}

extern "C" int edgeflow_llm_session_append_qwen25vl_visual_from_surface_lease(
    edgeflow_llm_session *session, int sequence_id,
    const edgeflow_llm_request *request, ANESurfaceLeaseHandle *lease,
    edgeflow_llm_surface_append_receipt *receipt) {
  return append_session_from_surface_lease(session, sequence_id, request, lease,
      receipt, edgeflow_llm_session_append_qwen25vl_visual_borrowed);
}
