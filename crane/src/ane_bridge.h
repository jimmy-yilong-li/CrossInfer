// ane_bridge.h — C-callable bridge to ANE private APIs for Python ctypes
// Wraps _ANEInMemoryModel via private AppleNeuralEngine.framework

#ifndef ANE_BRIDGE_H
#define ANE_BRIDGE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque kernel handle
typedef struct ANEKernelHandle ANEKernelHandle;

// Opaque leased IOSurface handle used for overlap-safe output handoff.
typedef struct ANESurfaceLeaseHandle ANESurfaceLeaseHandle;

// Scoped output-surface consumer callback.
// `base_address` is valid only for the duration of the callback.
// Implementations MUST NOT retain, stash, or return `base_address` past the
// callback's return.
typedef int (*ane_bridge_output_consumer_fn)(
    const void *base_address,
    size_t nbytes,
    void *user_ctx,
    char *error_out,
    size_t error_out_cap);

// Initialize ANE runtime (load private framework, resolve classes)
// Returns 0 on success, -1 on failure
int ane_bridge_init(void);

// Compile a MIL program with weight blobs into an ANE kernel
// mil_text: UTF-8 MIL program text
// mil_len: length of MIL text
// weight_data: raw weight blob (can be NULL)
// weight_len: length of weight blob
// n_inputs: number of input tensors
// input_sizes: array of byte sizes for each input
// n_outputs: number of output tensors
// output_sizes: array of byte sizes for each output
// Returns kernel handle or NULL on failure
ANEKernelHandle *ane_bridge_compile(const char *mil_text, size_t mil_len,
                                     const uint8_t *weight_data, size_t weight_len,
                                     int n_inputs, const size_t *input_sizes,
                                     int n_outputs, const size_t *output_sizes);

// Compile with multiple named weight files (for transformer kernels)
// weight_names: array of weight file paths (e.g. "@model_path/weights/wq.bin")
// weight_datas: array of weight data pointers
// weight_lens: array of weight data lengths
// n_weights: number of weight files
ANEKernelHandle *ane_bridge_compile_multi_weights(
    const char *mil_text, size_t mil_len,
    const char **weight_names, const uint8_t **weight_datas,
    const size_t *weight_lens, int n_weights,
    int n_inputs, const size_t *input_sizes,
    int n_outputs, const size_t *output_sizes);

// Evaluate (run) a compiled kernel on ANE
// Returns true on success
bool ane_bridge_eval(ANEKernelHandle *kernel);

// Evaluate a sequence of compiled kernels on ANE in-order.
// This is a minimal batch wrapper around repeated evaluateWithQoS calls,
// not a true multi-graph submission primitive.
bool ane_bridge_eval_batch(ANEKernelHandle **kernels, int count);

// Write data to kernel input tensor. Returns false on invalid index, null
// buffer/surface, oversized copy, lock failure, or missing base address.
bool ane_bridge_write_input(ANEKernelHandle *kernel, int idx,
                             const void *data, size_t bytes);

// Read data from kernel output tensor. Returns false on invalid index, null
// buffer/surface, oversized copy, lock failure, or missing base address.
bool ane_bridge_read_output(ANEKernelHandle *kernel, int idx,
                              void *data, size_t bytes);

// Export current bound IOSurface IDs for chaining/shared-surface reuse.
// Returns 0 on invalid kernel/index.
uint32_t ane_bridge_get_input_surface_id(ANEKernelHandle *kernel, int idx);
uint32_t ane_bridge_get_output_surface_id(ANEKernelHandle *kernel, int idx);
size_t ane_bridge_get_output_surface_size_bytes(ANEKernelHandle *kernel, int idx);

// Locks the kernel output surface, invokes `consumer`, then unlocks before
// returning. The consumer MUST NOT retain `base_address` past its return.
// Returns 0 on success, non-zero on error with `error_out` populated.
int ane_bridge_consume_output_surface(
    ANEKernelHandle *kernel,
    int idx,
    ane_bridge_output_consumer_fn consumer,
    void *user_ctx,
    char *error_out,
    size_t error_out_cap);

// Allocates a leased output surface of `bytes` size. Each lease owns an
// independently allocated IOSurface. The leased surface is not shared with
// other leases or kernels.
ANESurfaceLeaseHandle *ane_bridge_alloc_surface_lease(size_t bytes);

// Releases a previously allocated surface lease and its owned IOSurface.
void ane_bridge_free_surface_lease(ANESurfaceLeaseHandle *lease);

// Returns the underlying IOSurface ID for the leased surface, or 0 on error.
uint32_t ane_bridge_surface_lease_id(ANESurfaceLeaseHandle *lease);

// Returns the allocated byte size for the leased surface, or 0 on error.
size_t ane_bridge_surface_lease_size_bytes(ANESurfaceLeaseHandle *lease);

// Temporarily rebinds the kernel's output slot to the leased surface. The
// kernel's original output surface remains owned by the kernel and is not
// released by this call.
bool ane_bridge_bind_kernel_output_surface_lease(
    ANEKernelHandle *kernel,
    int idx,
    ANESurfaceLeaseHandle *lease);

// Locks the leased surface, invokes `consumer`, then unlocks before returning.
// The consumer MUST NOT retain `base_address` past its return.
// Returns 0 on success, non-zero on error with `error_out` populated.
int ane_bridge_consume_surface_lease(
    ANESurfaceLeaseHandle *lease,
    ane_bridge_output_consumer_fn consumer,
    void *user_ctx,
    char *error_out,
    size_t error_out_cap);

// Writable counterpart for a native producer such as CoreML outputBackings.
// The callback may cast base_address to writable storage only in this scope.
// Complete all writes before return; the pointer must not escape the callback.
int ane_bridge_produce_surface_lease(
    ANESurfaceLeaseHandle *lease,
    ane_bridge_output_consumer_fn producer,
    void *user_ctx,
    char *error_out,
    size_t error_out_cap);

// Materializes the leased surface into `dst` using the existing readback path.
// `bytes` must not exceed the leased surface size.
void ane_bridge_read_surface_lease(
    ANESurfaceLeaseHandle *lease,
    void *dst,
    size_t bytes);

// Stages `bytes` from `src` into the leased surface via a CPU-to-IOSurface
// memcpy. This is the producer-side counterpart of `ane_bridge_read_surface_lease`
// for staged producers. Lock/unlock lifetime is
// contained inside the bridge call. `bytes` must not exceed the leased
// surface size.
void ane_bridge_write_surface_lease(
    ANESurfaceLeaseHandle *lease,
    const void *src,
    size_t bytes);

int ane_bridge_prove_output_surface_direct_map(
    ANEKernelHandle *kernel,
    int idx,
    const char *baseline_path,
    char *json_out,
    size_t json_out_cap,
    char *error_out,
    size_t error_out_cap);

// Rebind a kernel input/output to an existing IOSurface looked up by ID.
// Rebuilds the underlying _ANERequest. Returns true on success.
bool ane_bridge_bind_input_surface_id(ANEKernelHandle *kernel, int idx, uint32_t surface_id);
bool ane_bridge_bind_output_surface_id(ANEKernelHandle *kernel, int idx, uint32_t surface_id);

// Free a compiled kernel and all associated resources
void ane_bridge_free(ANEKernelHandle *kernel);

// Get compile count (for exec() restart budgeting)
int ane_bridge_get_compile_count(void);

// Reset compile count
void ane_bridge_reset_compile_count(void);

// Build a weight blob in ANE format (128-byte header + fp16 data)
// src: float32 weights [rows x cols]
// Returns allocated buffer and sets out_len. Caller must free().
uint8_t *ane_bridge_build_weight_blob(const float *src, int rows, int cols,
                                       size_t *out_len);

// Build a transposed weight blob in ANE format
uint8_t *ane_bridge_build_weight_blob_transposed(const float *src, int rows, int cols,
                                                   size_t *out_len);

// Free a blob allocated by ane_bridge_build_weight_blob*
void ane_bridge_free_blob(void *ptr);

#ifdef __cplusplus
}
#endif

#endif // ANE_BRIDGE_H
