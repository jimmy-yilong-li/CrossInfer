// CoreML-visible FP32 output and the native consumer share leased IOSurfaces.
// The fv ABI owns two legacy slots; the ef ABI receives TABM pool leases.
// Build as Objective-C++ with ARC, blocks, CoreML/Foundation/IOSurface, and
// the existing crane bridge. This does not claim CoreML-internal zero-copy.
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "ane_bridge.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

struct FVCoreMLOutput {
    void * lease_handle;
    void * pointer;  // Diagnostic only; consumers map lease_handle in their scope.
    uint64_t generation;
    size_t nbytes;
    int32_t shape[3];
    int32_t strides[3];
    uint32_t surface_id;
    uint32_t slot;
    int32_t object_matches;
    int32_t pointer_matches;
};

namespace {
thread_local std::string error;

std::runtime_error failure(NSError * detail, const char * fallback) {
    return std::runtime_error(detail ? detail.localizedDescription.UTF8String : fallback);
}

struct Slot {
    ANESurfaceLeaseHandle * lease = nullptr;
    uint64_t generation = 0;

    ~Slot() {
        if (lease) ane_bridge_free_surface_lease(lease);
    }
};

struct State {
    MLModel * model = nil;
    NSURL * temporary_compilation = nil;
    Slot slots[2];
    int32_t width;
    int32_t visual_tokens = 0;
    size_t pixel_count = 0;
    NSArray<NSNumber *> * input_shape = nil;
    NSArray<NSNumber *> * input_strides = nil;
    uint64_t pooled_generation = 0;

    explicit State(int32_t width) : width(width) {}

    ~State() {
        model = nil;
        if (temporary_compilation) {
            [[NSFileManager defaultManager] removeItemAtURL:temporary_compilation error:nullptr];
        }
    }
};

template <typename Result, typename Function>
Result checked(Result rejected, Function function) {
    @autoreleasepool {
        @try {
            try {
                error.clear();
                return function();
            } catch (const std::exception & exception) {
                error = exception.what();
            }
        } @catch (NSException * exception) {
            error = exception.reason ? exception.reason.UTF8String : "CoreML raised NSException";
        }
    }
    return rejected;
}

struct PredictionContext {
    State * state;
    const float * pixels;
    int32_t slot_index;
    FVCoreMLOutput * output;
    ANESurfaceLeaseHandle * lease;
    uint64_t * generation;
    const float * text = nullptr;
    int32_t prefix = 0;
    int32_t suffix = 0;
};

int predict_into_surface(const void * base_address, size_t capacity, void * user_context,
                         char * error_out, size_t error_out_cap) {
    const int status = checked<int>(-1, [&]() -> int {
        auto & request = *static_cast<PredictionContext *>(user_context);
        State & state = *request.state;
        FVCoreMLOutput * output = request.output;
        const int32_t visual_tokens = state.visual_tokens;
        const size_t required_bytes = size_t(visual_tokens) * size_t(state.width) * sizeof(float);
        const size_t prefix_bytes = size_t(request.prefix) * size_t(state.width) * sizeof(float);
        const size_t suffix_bytes = size_t(request.suffix) * size_t(state.width) * sizeof(float);
        // The producer bridge grants writable access to the original lease mapping.
        auto * storage = static_cast<uint8_t *>(const_cast<void *>(base_address));
        void * base = storage ? storage + prefix_bytes : nullptr;
        if (!base || reinterpret_cast<uintptr_t>(base) % alignof(float) != 0 ||
            capacity < prefix_bytes + required_bytes + suffix_bytes) {
            throw std::runtime_error("output surface callback has invalid base address or capacity");
        }
        if (request.text) {
            std::memcpy(storage, request.text, prefix_bytes);
            std::memcpy(storage + prefix_bytes + required_bytes,
                        request.text + size_t(request.prefix) * state.width, suffix_bytes);
        }
        @autoreleasepool {
            NSArray<NSNumber *> * shape = @[@1, @(visual_tokens), @(state.width)];
            NSArray<NSNumber *> * strides = @[@(visual_tokens * state.width), @(state.width), @1];
            NSError * detail = nil;
            // The lease owns storage. All wrappers die before this callback returns.
            MLMultiArray * backing = [[MLMultiArray alloc]
                initWithDataPointer:base shape:shape dataType:MLMultiArrayDataTypeFloat32
                strides:strides deallocator:nil error:&detail];
            if (!backing) throw failure(detail, "output MLMultiArray creation failed");
            MLMultiArray * input = [[MLMultiArray alloc]
                initWithDataPointer:const_cast<float *>(request.pixels) shape:state.input_shape
                dataType:MLMultiArrayDataTypeFloat32 strides:state.input_strides
                deallocator:nil error:&detail];
            if (!input) throw failure(detail, "input MLMultiArray creation failed");
            MLDictionaryFeatureProvider * inputs = [[MLDictionaryFeatureProvider alloc]
                initWithDictionary:@{@"pixels": [MLFeatureValue featureValueWithMultiArray:input]}
                error:&detail];
            if (!inputs) throw failure(detail, "input feature provider creation failed");
            MLPredictionOptions * options = [[MLPredictionOptions alloc] init];
            options.outputBackings = @{@"image_features": backing};
            id<MLFeatureProvider> prediction = [state.model predictionFromFeatures:inputs
                options:options error:&detail];
            if (!prediction) throw failure(detail, "CoreML prediction failed");
            MLMultiArray * actual = [prediction featureValueForName:@"image_features"].multiArrayValue;
            if (actual != backing || actual.dataPointer != base ||
                actual.dataType != MLMultiArrayDataTypeFloat32 ||
                ![actual.shape isEqualToArray:shape] || ![actual.strides isEqualToArray:strides]) {
                throw std::runtime_error("CoreML output backing contract failed: identity, pointer, dtype, shape or strides");
            }
            output->lease_handle = request.lease;
            output->pointer = base;
            output->generation = ++*request.generation;
            output->nbytes = required_bytes;
            output->shape[0] = 1;
            output->shape[1] = visual_tokens;
            output->shape[2] = state.width;
            output->strides[0] = visual_tokens * state.width;
            output->strides[1] = state.width;
            output->strides[2] = 1;
            output->surface_id = ane_bridge_surface_lease_id(request.lease);
            output->slot = uint32_t(request.slot_index);
            output->object_matches = output->pointer_matches = 1;
        }
        return 0;
    });
    if (status != 0 && error_out && error_out_cap > 0) {
        std::snprintf(error_out, error_out_cap, "%s", error.c_str());
    }
    return status;
}
void * initialize(const char * package_path, int32_t width, bool owned_slots) {
    return checked<void *>(nullptr, [&]() -> void * {
        if (!package_path || width <= 0) {
            throw std::runtime_error("invalid CoreML package path or output width");
        }
        auto state = std::make_unique<State>(width);
        NSURL * url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:package_path]];
        NSError * detail = nil;
        if (![url.pathExtension isEqualToString:@"mlmodelc"]) {
            url = [MLModel compileModelAtURL:url error:&detail];
            if (!url) throw failure(detail, "CoreML compilation failed");
            state->temporary_compilation = url;
        }
        MLModelConfiguration * configuration = [[MLModelConfiguration alloc] init];
        configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        state->model = [MLModel modelWithContentsOfURL:url configuration:configuration error:&detail];
        if (!state->model) throw failure(detail, "CoreML model load failed");

        auto input = state->model.modelDescription.inputDescriptionsByName[@"pixels"].multiArrayConstraint;
        auto output = state->model.modelDescription.outputDescriptionsByName[@"image_features"].multiArrayConstraint;
        if (!input || !output || input.dataType != MLMultiArrayDataTypeFloat32 ||
            output.dataType != MLMultiArrayDataTypeFloat32 || output.shape.count != 3 ||
            output.shape[0].intValue != 1 || output.shape[2].intValue != width) {
            throw std::runtime_error("expected FP32 pixels and [1,tokens,width] image_features");
        }
        state->visual_tokens = output.shape[1].intValue;
        if (state->visual_tokens <= 0 || width > std::numeric_limits<int32_t>::max() / state->visual_tokens) {
            throw std::runtime_error("invalid visual output dimensions");
        }
        state->input_shape = input.shape;
        NSMutableArray<NSNumber *> * strides = [NSMutableArray array];
        size_t count = 1;
        for (NSInteger index = input.shape.count - 1; index >= 0; --index) {
            const size_t dimension = input.shape[index].unsignedLongLongValue;
            if (!dimension || count > size_t(std::numeric_limits<int32_t>::max()) / dimension)
                throw std::runtime_error("invalid visual input dimensions");
            [strides insertObject:@(count) atIndex:0];
            count *= dimension;
        }
        state->pixel_count = count;
        state->input_strides = strides;
        const size_t bytes = size_t(state->visual_tokens) * size_t(width) * sizeof(float);
        if (owned_slots) {
            for (Slot & slot : state->slots) {
                slot.lease = ane_bridge_alloc_surface_lease(bytes);
                if (!slot.lease) throw std::runtime_error("output surface lease allocation failed");
                if (!ane_bridge_surface_lease_id(slot.lease) ||
                    ane_bridge_surface_lease_size_bytes(slot.lease) < bytes) {
                    throw std::runtime_error("output surface lease identity or capacity check failed");
                }
            }
        }
        return state.release();
    });
}
}  // namespace

extern "C" {
const char * fv_coreml_error() { return error.c_str(); }

void * fv_coreml_init(const char * package_path, int32_t width) {
    return initialize(package_path, width, true);
}

void * ef_coreml_init(const char * package_path, int32_t width) {
    return initialize(package_path, width, false);
}

int fv_coreml_predict(void * context, const float * pixels, size_t count,
                      int32_t slot_index, FVCoreMLOutput * output) {
    return checked<int>(-1, [&]() -> int {
        if (!context || !pixels || !output || slot_index < 0 || slot_index >= 2) {
            throw std::runtime_error("invalid CoreML prediction arguments");
        }
        *output = {};
        auto & state = *static_cast<State *>(context);
        if (count != state.pixel_count) throw std::runtime_error("invalid CoreML input size");
        Slot & slot = state.slots[slot_index];
        PredictionContext request{&state, pixels, slot_index, output, slot.lease, &slot.generation};
        char bridge_error[512] = {};
        const int status = ane_bridge_produce_surface_lease(
            slot.lease, predict_into_surface, &request, bridge_error, sizeof(bridge_error));
        if (status != 0) {
            *output = {};
            throw std::runtime_error(bridge_error[0] ? bridge_error : "CoreML output producer callback failed");
        }
        return 0;
    });
}

int ef_coreml_predict_into(void * context, const float * pixels, size_t count,
                          ANESurfaceLeaseHandle * lease, FVCoreMLOutput * output) {
    return checked<int>(-1, [&]() -> int {
        if (!context || !pixels || !lease || !output)
            throw std::runtime_error("invalid pooled CoreML prediction arguments");
        auto & state = *static_cast<State *>(context);
        if (count != state.pixel_count) throw std::runtime_error("invalid CoreML input size");
        *output = {};
        PredictionContext request{&state, pixels, -1, output, lease, &state.pooled_generation};
        char bridge_error[512] = {};
        if (ane_bridge_produce_surface_lease(lease, predict_into_surface, &request,
                                            bridge_error, sizeof(bridge_error)) != 0) {
            *output = {};
            throw std::runtime_error(bridge_error);
        }
        return 0;
    });
}

int ef_coreml_predict_packed(void * context, const float * pixels, size_t count,
                            ANESurfaceLeaseHandle * lease, const float * text,
                            int32_t prefix, int32_t suffix, FVCoreMLOutput * output) {
    return checked<int>(-1, [&]() -> int {
        if (!context || !pixels || !lease || !text || !output || prefix <= 0 || suffix <= 0)
            throw std::runtime_error("invalid packed CoreML prediction arguments");
        auto & state = *static_cast<State *>(context);
        if (count != state.pixel_count) throw std::runtime_error("invalid CoreML input size");
        *output = {};
        PredictionContext request{&state, pixels, -1, output, lease, &state.pooled_generation,
                                  text, prefix, suffix};
        char bridge_error[512] = {};
        if (ane_bridge_produce_surface_lease(lease, predict_into_surface, &request,
                                            bridge_error, sizeof(bridge_error)) != 0) {
            *output = {};
            throw std::runtime_error(bridge_error);
        }
        return 0;
    });
}

void fv_coreml_free(void * context) {
    @autoreleasepool {
        delete static_cast<State *>(context);
    }
}
}
