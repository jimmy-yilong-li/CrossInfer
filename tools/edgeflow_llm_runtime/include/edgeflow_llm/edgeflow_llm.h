#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct edgeflow_llm_ctx edgeflow_llm_ctx;
typedef struct edgeflow_llm_session edgeflow_llm_session;
typedef struct edgeflow_llm_root_snapshot edgeflow_llm_root_snapshot;
typedef struct ANEKernelHandle ANEKernelHandle;
typedef struct ANESurfaceLeaseHandle ANESurfaceLeaseHandle;

#define EDGEFLOW_LLM_ABI_VERSION 1u
#define EDGEFLOW_LLM_SESSION_ABI_VERSION 3u
#define EDGEFLOW_LLM_SHA256_HEX_SIZE 65u

typedef struct {
    int n_gpu_layers;
    int n_ctx;
    int n_batch;
    int n_threads;
} edgeflow_llm_init_params;

typedef struct {
    const char * prompt;
    const float * visual_embd;
    int num_visual_tokens;
    int hidden_size;
    int image_grid_thw[3];
    int max_new_tokens;
} edgeflow_llm_request;

typedef struct {
    char * text;
    float first_token_ms;
    float prompt_eval_ms;
    float eval_ms;
    int eval_count;
} edgeflow_llm_result;

uint32_t edgeflow_llm_abi_version(void);
uint32_t edgeflow_llm_init_params_sizeof(void);
uint32_t edgeflow_llm_request_sizeof(void);
uint32_t edgeflow_llm_request_v2_sizeof(void);
uint32_t edgeflow_llm_result_sizeof(void);

edgeflow_llm_ctx * edgeflow_llm_init(const char * model_path, edgeflow_llm_init_params params);
void edgeflow_llm_free(edgeflow_llm_ctx * ctx);

/* Optional stateless generation controls; defaults are natural EOG and no callback.
 * The caller must serialize access and reset options before releasing the context.
 * Pieces are raw bytes, valid during the callback only. Nonzero aborts generation. */
typedef int (*edgeflow_llm_token_callback)(int token_id, const char * piece,
    uint64_t size, void * user_data);
int edgeflow_llm_set_generation_options(edgeflow_llm_ctx * ctx, int ignore_eog,
    edgeflow_llm_token_callback callback, void * user_data);

int edgeflow_llm_generate(
    edgeflow_llm_ctx * ctx,
    const edgeflow_llm_request * req,
    edgeflow_llm_result * out);

int edgeflow_llm_generate_from_ane_surface(
    edgeflow_llm_ctx * ctx,
    const edgeflow_llm_request * req,
    ANEKernelHandle * kernel,
    int output_idx,
    edgeflow_llm_result * out);

int edgeflow_llm_generate_from_surface_lease(
    edgeflow_llm_ctx * ctx,
    const edgeflow_llm_request * req,
    ANESurfaceLeaseHandle * lease,
    edgeflow_llm_result * out);

/* Additive Qwen2.5 fresh-request path. sizes = {prefix, suffix, width}.
 * Query sizes with embeddings=NULL/capacity=0; otherwise write only text rows.
 * The packed lease is [prefix text, visual output, suffix text], all FP32. */
int edgeflow_llm_qwen25_prompt_embeddings(edgeflow_llm_ctx * ctx, const char * prompt,
    float * embeddings, uint64_t capacity_floats, int32_t sizes[3]);
int edgeflow_llm_generate_packed(edgeflow_llm_ctx * ctx, const edgeflow_llm_request * req,
    const float * packed_embeddings, int prefix_tokens, int suffix_tokens, edgeflow_llm_result * out);
int edgeflow_llm_generate_from_surface_lease_packed(edgeflow_llm_ctx * ctx,
    const edgeflow_llm_request * req, ANESurfaceLeaseHandle * lease,
    int prefix_tokens, int suffix_tokens, edgeflow_llm_result * out);

void edgeflow_llm_free_result(edgeflow_llm_result * result);

/* ============================================================ *
 *  v2 ABI — explicit model family + visual dtype                *
 *                                                               *
 *  Supports FastVLM / LLaVA-style multimodal injection        *
 *  (single `-200` sentinel, 1D RoPE positions,                 *
 *  fp16 or fp32 visual payload). The v1 request/entry point    *
 *  above stays bit-for-bit compatible with existing callers.   *
 * ============================================================ */

typedef enum {
    EDGEFLOW_MODEL_FAMILY_QWEN25VL = 0,  /* existing Qwen2.5-VL path (v1 semantics)   */
    EDGEFLOW_MODEL_FAMILY_FASTVLM  = 1,  /* FastVLM / LLaVA + Qwen2 text backbone     */
    EDGEFLOW_MODEL_FAMILY_LLAVA15  = 2,  /* LLaVA-1.5 + LLaMA-2 text backbone         */
    EDGEFLOW_MODEL_FAMILY_QWEN35   = 3,  /* hybrid Qwen3.5, session API only          */
} edgeflow_model_family;

typedef enum {
    EDGEFLOW_DTYPE_FP32 = 0,
    EDGEFLOW_DTYPE_FP16 = 1,
} edgeflow_dtype;

typedef enum {
  EDGEFLOW_KV_DTYPE_FP16 = 0,
} edgeflow_kv_dtype;

typedef enum {
  EDGEFLOW_POSITION_CONTRACT_TEXT_1D = 0,
  EDGEFLOW_POSITION_CONTRACT_QWEN25VL_MROPE = 1,
  EDGEFLOW_POSITION_CONTRACT_QWEN35_IMROPE = 2,
} edgeflow_position_contract;

typedef enum {
  EDGEFLOW_BACKEND_PLACEMENT_CPU_ONLY = 0,
  EDGEFLOW_BACKEND_PLACEMENT_GPU_FULL = 1,
  EDGEFLOW_BACKEND_PLACEMENT_MIXED = 2,
} edgeflow_backend_placement;

typedef struct {
    const char *            prompt;
    /* Opaque visual embedding bytes. Interpretation is governed by
     * `visual_dtype` and the `num_visual_tokens * hidden_size * element_size`
     * byte count. May be NULL for text-only generation. */
    const void *            visual_embd;
    int                     num_visual_tokens;
    int                     hidden_size;
    int                     image_grid_thw[3];  /* decorative for FASTVLM */
    int                     max_new_tokens;
    edgeflow_model_family   model_family;
    edgeflow_dtype          visual_dtype;
} edgeflow_llm_request_v2;

int edgeflow_llm_generate_v2(
    edgeflow_llm_ctx *                  ctx,
    const edgeflow_llm_request_v2 *     req,
    edgeflow_llm_result *               out);

int edgeflow_llm_generate_from_surface_lease_v2(
    edgeflow_llm_ctx *                  ctx,
    const edgeflow_llm_request_v2 *     req,
    ANESurfaceLeaseHandle *             lease,
    edgeflow_llm_result *               out);

/* ============================================================ *
 *  Stateful session ABI                                         *
 *                                                               *
 *  This API owns one persistent llama.cpp context. It supports  *
 *  local, unified-KV sequence forks and time-sliced decoding;   *
 *  it is not a continuous-batching API. Optional root-state     *
 *  handoff below replaces a compatible persistent session root. *
 *  All operations except free must run on the creating thread.  *
 * ============================================================ */

typedef struct {
  int n_gpu_layers;
  int n_ctx;
  int n_batch;
  int n_ubatch;
  int n_seq_max;
  int n_threads;
  int kv_unified;
  edgeflow_model_family model_family;
  edgeflow_kv_dtype kv_type_k;
  edgeflow_kv_dtype kv_type_v;
  edgeflow_position_contract position_contract;
  uint32_t prompt_contract_version;
} edgeflow_llm_session_params;

typedef struct {
  uint32_t session_abi_version;
  int requested_gpu_layers;
  int observed_gpu_layers;
  edgeflow_backend_placement backend_placement;
  int n_ctx;
  int n_batch;
  int n_ubatch;
  int n_seq_max;
  int n_threads;
  int kv_unified;
  edgeflow_model_family model_family;
  edgeflow_kv_dtype kv_type_k;
  edgeflow_kv_dtype kv_type_v;
  edgeflow_position_contract position_contract;
  uint32_t prompt_contract_version;
  char model_sha256[EDGEFLOW_LLM_SHA256_HEX_SIZE];
  /* SHA-256 of the loaded dylib; this is the authoritative binary identity. */
  char runtime_sha256[EDGEFLOW_LLM_SHA256_HEX_SIZE];
  /* Diagnostic fingerprint of selected source inputs, not a binary identity. */
  char runtime_source_fingerprint_sha256[EDGEFLOW_LLM_SHA256_HEX_SIZE];
} edgeflow_llm_session_signature;

typedef struct {
  int active;
  int next_position;
  int has_logits;
  uint64_t state_bytes;
} edgeflow_llm_sequence_info;

typedef struct {
  int token_id;
  int token_position;
  int is_eog;
  int next_token_id;
  int n_vocab;
} edgeflow_llm_decode_step_result;

enum {
  EDGEFLOW_LLM_SESSION_OK = 0,
  EDGEFLOW_LLM_SESSION_INVALID_ARGUMENT = 1,
  EDGEFLOW_LLM_SESSION_INVALID_STATE = 2,
  EDGEFLOW_LLM_SESSION_WRONG_THREAD = 3,
  EDGEFLOW_LLM_SESSION_RUNTIME_ERROR = 4,
  EDGEFLOW_LLM_SESSION_BUFFER_TOO_SMALL = 5,
};

uint32_t edgeflow_llm_session_abi_version(void);
uint32_t edgeflow_llm_session_params_sizeof(void);
uint32_t edgeflow_llm_session_signature_sizeof(void);
uint32_t edgeflow_llm_sequence_info_sizeof(void);
uint32_t edgeflow_llm_decode_step_result_sizeof(void);

edgeflow_llm_session *
edgeflow_llm_session_init(const char *model_path,
                          const edgeflow_llm_session_params *params);
void edgeflow_llm_session_free(edgeflow_llm_session *session);
const char *edgeflow_llm_session_last_init_error(void);
const char *
edgeflow_llm_session_last_error(const edgeflow_llm_session *session);

int edgeflow_llm_session_get_signature(edgeflow_llm_session *session,
                                       edgeflow_llm_session_signature *out);
int edgeflow_llm_session_vocab_size(edgeflow_llm_session *session);
int edgeflow_llm_session_get_sequence_info(edgeflow_llm_session *session,
                                           int sequence_id,
                                           edgeflow_llm_sequence_info *out);
/* Optional additive ABI-3 capability for text/completed ordered images only.
 * Owns seq-0 serialized KV, position/boundary metadata, and source signature;
 * survives donor append/free. No logits are transferred. size returns the
 * serialized KV byte count (zero for NULL); free accepts NULL. */
int edgeflow_llm_session_export_root(edgeflow_llm_session *session,
                                      edgeflow_llm_root_snapshot **out);
/* Replaces root only with no live branches and matching model/runtime, ABI,
 * context/KV, family, position and prompt contracts. Placement, thread and
 * batch counts are not compatibility keys. Append a query to get fresh logits.
 * Native M-RoPE cell-extension serialization remains unsupported. */
int edgeflow_llm_session_import_root(
    edgeflow_llm_session *session, const edgeflow_llm_root_snapshot *snapshot);
void edgeflow_llm_root_snapshot_free(edgeflow_llm_root_snapshot *snapshot);
uint64_t
edgeflow_llm_root_snapshot_size(const edgeflow_llm_root_snapshot *snapshot);
/* Untimed diagnostic: exact serialized KV and position/boundary equality.
 * Returns 1 for equal, 0 for different, -INVALID_ARGUMENT for NULL. */
int edgeflow_llm_root_snapshot_equal(const edgeflow_llm_root_snapshot *left,
                                      const edgeflow_llm_root_snapshot *right);
/* Healthy owner-thread session only; requires no live branches. Clears all KV
 * and sequence metadata while retaining the loaded model/context. */
int edgeflow_llm_session_reset_root(edgeflow_llm_session *session);
int edgeflow_llm_session_ingest_text(edgeflow_llm_session *session,
                                     int sequence_id, const char *text,
                                     int add_special, int parse_special,
                                     int *out_token_count);
/* Applies the same chat-template rendering used by edgeflow_llm_generate. */
int edgeflow_llm_session_ingest_user_prompt(edgeflow_llm_session *session,
                                            int sequence_id, const char *prompt,
                                            int *out_token_count);
int edgeflow_llm_session_ingest_tokens(edgeflow_llm_session *session,
                                       int sequence_id, const int32_t *tokens,
                                       int token_count);
/* Ingests one completed Qwen image/message into an empty sequence. */
int edgeflow_llm_session_ingest_qwen25vl(edgeflow_llm_session *session,
                                         int sequence_id,
                                         const edgeflow_llm_request *request);
/* Ingests only the reusable chat prefix, image markers, and image embeddings.
 * A fork at the resulting position may append one branch-local query. */
int edgeflow_llm_session_ingest_qwen25vl_visual_prefix(
    edgeflow_llm_session *session, int sequence_id,
    const edgeflow_llm_request *request);
/* Optional additive ABI-3 capability. Appends end/start markers and one image
 * to an unmodified root visual boundary. Requires all response branches
 * released; only latest-head forks are admitted after append. Ordered image
 * sequence only: no temporal video M-RoPE, historical multi-image forks,
 * active-branch append, or shared cross-backend KV. */
int edgeflow_llm_session_append_qwen25vl_visual(
    edgeflow_llm_session *session, int sequence_id,
    const edgeflow_llm_request *request);
/* Qwen3.5 prompt contract v1: one user turn, ordered images, no system/tools,
 * no vision IDs, enable_thinking=false. First append opens the user turn;
 * later appends close/open image markers. Only root zero, with no live branches.
 * Input is borrowed contiguous FP32; publication includes llama_synchronize.
 * Receipt covers the shared-output -> llama_batch boundary, not internal copies.
 */
typedef struct {
  uint64_t surface_pointer;
  uint64_t decoder_pointer;
  uint64_t payload_bytes;
  int next_position;
} edgeflow_llm_surface_append_receipt;
uint32_t edgeflow_llm_surface_append_receipt_sizeof(void);
/* Qwen2.5: empty root uses the existing chat-template visual prefix; subsequent
 * calls use ordered-image append. No live branches. Input is borrowed through
 * synchronized completion, with the same receipt scope as Qwen3.5. */
int edgeflow_llm_session_append_qwen25vl_visual_borrowed(
    edgeflow_llm_session *session, int sequence_id,
    const edgeflow_llm_request *request,
    edgeflow_llm_surface_append_receipt *receipt);
int edgeflow_llm_session_append_qwen25vl_visual_from_surface_lease(
    edgeflow_llm_session *session, int sequence_id,
    const edgeflow_llm_request *request, ANESurfaceLeaseHandle *lease,
    edgeflow_llm_surface_append_receipt *receipt);
int edgeflow_llm_session_append_qwen35_visual(
    edgeflow_llm_session *session, int sequence_id,
    const edgeflow_llm_request *request,
    edgeflow_llm_surface_append_receipt *receipt);
/* Borrows lease until synchronized return; caller retains/releases ownership. */
int edgeflow_llm_session_append_qwen35_visual_from_surface_lease(
    edgeflow_llm_session *session, int sequence_id,
    const edgeflow_llm_request *request, ANESurfaceLeaseHandle *lease,
    edgeflow_llm_surface_append_receipt *receipt);
int edgeflow_llm_session_ingest_qwen35_query(
    edgeflow_llm_session *session, int sequence_id, const char *prompt,
    int *out_token_count);
/* Completes a visual-prefix sequence with one user query and assistant header.
 */
int edgeflow_llm_session_ingest_qwen25vl_query(edgeflow_llm_session *session,
                                               int sequence_id,
                                               const char *prompt,
                                               int *out_token_count);
int edgeflow_llm_session_fork_prefix(edgeflow_llm_session *session,
                                     int source_sequence_id,
                                     int destination_sequence_id,
                                     int end_position);
/* Reads the cached greedy decision without decoding or advancing KV/position. */
int edgeflow_llm_session_peek_token(edgeflow_llm_session *session,
                                    int sequence_id, int *out_token_id,
                                    int *out_is_eog);
int edgeflow_llm_session_decode_step(edgeflow_llm_session *session,
                                     int sequence_id,
                                     edgeflow_llm_decode_step_result *out,
                                     float *out_logits, int logits_capacity);
/* Owner thread only. 1: peek/decode mask EOG; 0 restores natural EOG. */
int edgeflow_llm_session_set_ignore_eog(edgeflow_llm_session *session, int ignore_eog);
/* Writes the exact byte piece used by the legacy greedy generation loop. */
int edgeflow_llm_session_token_piece(edgeflow_llm_session *session,
                                     int token_id, char *out, int capacity,
                                     int *out_size);
/* Sequence zero is the canonical stream root and cannot be removed. */
int edgeflow_llm_session_remove_sequence(edgeflow_llm_session *session,
                                         int sequence_id);

#ifdef __cplusplus
}
#endif
