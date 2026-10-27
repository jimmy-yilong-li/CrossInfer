// ane_bridge.m — Objective-C implementation of ANE bridge for Python ctypes
// Wraps _ANEInMemoryModel private APIs into C-callable functions

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#include "ane_bridge.h"

// --- Private class references ---
static Class g_ANEDesc = nil;
static Class g_ANEInMem = nil;
static Class g_ANEReq = nil;
static Class g_ANEIO = nil;
static bool g_initialized = false;
static int g_compile_count = 0;

// --- Kernel handle struct ---
struct ANEKernelHandle {
    id model;               // _ANEInMemoryModel
    IOSurfaceRef *ioInputs;
    IOSurfaceRef *ioOutputs;
    IOSurfaceRef *ownedOutputSurfaces;
    id request;             // _ANERequest
    NSString *tmpDir;
    int nInputs, nOutputs;
    size_t *inputBytes;
    size_t *outputBytes;
};

struct ANESurfaceLeaseHandle {
    IOSurfaceRef surface;
    size_t bytes;
    uint32_t surface_id;
};

void ane_bridge_free(ANEKernelHandle *kernel);

// --- Public API ---

int ane_bridge_init(void) {
    if (g_initialized) return 0;

    void *handle = dlopen(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "ane_bridge: Failed to load AppleNeuralEngine.framework\n");
        return -1;
    }

    g_ANEDesc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_ANEInMem = NSClassFromString(@"_ANEInMemoryModel");
    g_ANEReq   = NSClassFromString(@"_ANERequest");
    g_ANEIO    = NSClassFromString(@"_ANEIOSurfaceObject");

    if (!g_ANEDesc || !g_ANEInMem || !g_ANEReq || !g_ANEIO) {
        fprintf(stderr, "ane_bridge: Failed to resolve ANE private classes\n");
        return -1;
    }

    g_initialized = true;
    g_compile_count = 0;
    return 0;
}

static IOSurfaceRef create_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0
    });
}

static void set_error_message(char *error_out, size_t error_out_cap, const char *message) {
    if (!error_out || error_out_cap == 0) {
        return;
    }
    snprintf(error_out, error_out_cap, "%s", message ? message : "unknown error");
}

static id build_request_for_kernel(ANEKernelHandle *kernel) {
    if (!kernel) return nil;

    NSMutableArray *wIns = [NSMutableArray arrayWithCapacity:kernel->nInputs];
    NSMutableArray *iIdx = [NSMutableArray arrayWithCapacity:kernel->nInputs];
    for (int i = 0; i < kernel->nInputs; i++) {
        if (!kernel->ioInputs[i]) return nil;
        [wIns addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            g_ANEIO, @selector(objectWithIOSurface:), kernel->ioInputs[i])];
        [iIdx addObject:@(i)];
    }

    NSMutableArray *wOuts = [NSMutableArray arrayWithCapacity:kernel->nOutputs];
    NSMutableArray *oIdx = [NSMutableArray arrayWithCapacity:kernel->nOutputs];
    for (int i = 0; i < kernel->nOutputs; i++) {
        if (!kernel->ioOutputs[i]) return nil;
        [wOuts addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            g_ANEIO, @selector(objectWithIOSurface:), kernel->ioOutputs[i])];
        [oIdx addObject:@(i)];
    }

    return ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
        g_ANEReq,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        wIns, iIdx, wOuts, oIdx, nil, nil, @0);
}

static bool rebind_surface_slot(
    ANEKernelHandle *kernel,
    bool isInput,
    int idx,
    uint32_t surface_id)
{
    if (!kernel) return false;
    int count = isInput ? kernel->nInputs : kernel->nOutputs;
    if (idx < 0 || idx >= count) return false;

    IOSurfaceRef newSurface = IOSurfaceLookup(surface_id);
    if (!newSurface) return false;

    size_t requiredBytes = isInput ? kernel->inputBytes[idx] : kernel->outputBytes[idx];
    if (IOSurfaceGetAllocSize(newSurface) < requiredBytes) {
        CFRelease(newSurface);
        return false;
    }

    IOSurfaceRef *slots = isInput ? kernel->ioInputs : kernel->ioOutputs;
    IOSurfaceRef oldSurface = slots[idx];
    id oldRequest __unused = kernel->request;

    slots[idx] = newSurface;
    id newRequest = build_request_for_kernel(kernel);
    if (!newRequest) {
        slots[idx] = oldSurface;
        CFRelease(newSurface);
        return false;
    }

    kernel->request = newRequest;
    if (oldSurface) CFRelease(oldSurface);
    oldRequest = nil;
    return true;
}

static bool rebind_output_slot_with_surface(
    ANEKernelHandle *kernel,
    int idx,
    IOSurfaceRef newSurface)
{
    if (!kernel || idx < 0 || idx >= kernel->nOutputs || !newSurface) return false;

    size_t requiredBytes = kernel->outputBytes[idx];
    if (IOSurfaceGetAllocSize(newSurface) < requiredBytes) return false;

    IOSurfaceRef *slots = kernel->ioOutputs;
    IOSurfaceRef oldSurface = slots[idx];
    id oldRequest __unused = kernel->request;
    IOSurfaceRef retainedSurface = (IOSurfaceRef)CFRetain(newSurface);

    slots[idx] = retainedSurface;
    id newRequest = build_request_for_kernel(kernel);
    if (!newRequest) {
        slots[idx] = oldSurface;
        CFRelease(retainedSurface);
        return false;
    }

    kernel->request = newRequest;
    if (oldSurface) CFRelease(oldSurface);
    oldRequest = nil;
    return true;
}

static int access_surface_locked(
    IOSurfaceRef surface,
    ane_bridge_output_consumer_fn consumer,
    void *user_ctx,
    char *error_out,
    size_t error_out_cap,
    IOSurfaceLockOptions options)
{
    if (!surface) {
        set_error_message(error_out, error_out_cap, "surface is null");
        return 2;
    }
    if (!consumer) {
        set_error_message(error_out, error_out_cap, "consumer callback is null");
        return 3;
    }

    const IOReturn lockStatus = IOSurfaceLock(surface, options, NULL);
    if (lockStatus != kIOReturnSuccess) {
        set_error_message(error_out, error_out_cap, "IOSurfaceLock failed");
        return 4;
    }

    int rc = 0;
    @try {
        void *baseAddress = IOSurfaceGetBaseAddress(surface);
        if (!baseAddress) {
            set_error_message(error_out, error_out_cap, "IOSurfaceGetBaseAddress failed");
            rc = 5;
        } else {
            rc = consumer(baseAddress, IOSurfaceGetAllocSize(surface), user_ctx, error_out, error_out_cap);
        }
    } @catch (NSException *exc) {
        const char *reason = exc.reason ? exc.reason.UTF8String : "consumer callback raised NSException";
        set_error_message(error_out, error_out_cap, reason);
        rc = 6;
    } @finally {
        IOSurfaceUnlock(surface, options, NULL);
    }

    return rc;
}

static int consume_surface_locked(
    IOSurfaceRef surface,
    ane_bridge_output_consumer_fn consumer,
    void *user_ctx,
    char *error_out,
    size_t error_out_cap)
{
    return access_surface_locked(surface, consumer, user_ctx, error_out,
                                 error_out_cap, kIOSurfaceLockReadOnly);
}

ANEKernelHandle *ane_bridge_compile_multi_weights(
    const char *mil_text, size_t mil_len,
    const char **weight_names, const uint8_t **weight_datas,
    const size_t *weight_lens, int n_weights,
    int n_inputs, const size_t *input_sizes,
    int n_outputs, const size_t *output_sizes)
{
    @autoreleasepool {
        if (!g_initialized) {
            fprintf(stderr, "ane_bridge: Not initialized\n");
            return NULL;
        }

        NSData *milData = [NSData dataWithBytes:mil_text length:mil_len];
        NSError *e = nil;

        // Build weight dictionary
        NSMutableDictionary *wdict = [NSMutableDictionary dictionary];
        for (int i = 0; i < n_weights; i++) {
            NSString *name = [NSString stringWithUTF8String:weight_names[i]];
            NSData *data = [NSData dataWithBytes:weight_datas[i] length:weight_lens[i]];
            wdict[name] = @{@"offset": @0, @"data": data};
        }

        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, wdict.count > 0 ? wdict : nil, nil);
        if (!desc) {
            fprintf(stderr, "ane_bridge: modelWithMILText failed\n");
            return NULL;
        }

        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
            g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
        if (!mdl) {
            fprintf(stderr, "ane_bridge: inMemoryModelWithDescriptor failed\n");
            return NULL;
        }

        // Pre-populate temp dir
        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];

        for (int i = 0; i < n_weights; i++) {
            NSString *name = [NSString stringWithUTF8String:weight_names[i]];
            // Extract filename from path like "@model_path/weights/wq.bin" -> "weights/wq.bin"
            NSString *relPath = name;
            if ([name hasPrefix:@"@model_path/"]) {
                relPath = [name substringFromIndex:12];
            }
            NSString *fullPath = [td stringByAppendingPathComponent:relPath];
            NSString *dir = [fullPath stringByDeletingLastPathComponent];
            [fm createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil];
            NSData *data = [NSData dataWithBytes:weight_datas[i] length:weight_lens[i]];
            [data writeToFile:fullPath atomically:YES];
        }

        // Compile
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
            fprintf(stderr, "ane_bridge: ANE compile failed: %s\n",
                    e ? [[e description] UTF8String] : "unknown");
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }

        // Load (with one retry after a brief pause for ANE slot reclamation)
        BOOL loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        if (!loaded) {
            fprintf(stderr, "ane_bridge: ANE load failed (retrying in 100ms): %s\n",
                    e ? [[e description] UTF8String] : "unknown");
            usleep(100000); // 100ms
            e = nil;
            loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                    mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        }
        if (!loaded) {
            fprintf(stderr, "ane_bridge: ANE load failed after retry: %s\n",
                    e ? [[e description] UTF8String] : "unknown");
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }

        g_compile_count++;

        // Create kernel handle
        ANEKernelHandle *k = (ANEKernelHandle *)calloc(1, sizeof(ANEKernelHandle));
        if (!k) {
            fprintf(stderr, "ane_bridge: kernel handle allocation failed\n");
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }
        k->model = mdl;
        k->tmpDir = td;
        k->nInputs = n_inputs;
        k->nOutputs = n_outputs;
        k->inputBytes = (size_t *)malloc(n_inputs * sizeof(size_t));
        k->outputBytes = (size_t *)malloc(n_outputs * sizeof(size_t));
        k->ownedOutputSurfaces = (IOSurfaceRef *)calloc(n_outputs, sizeof(IOSurfaceRef));
        if (!k->inputBytes || !k->outputBytes || !k->ownedOutputSurfaces) {
            fprintf(stderr, "ane_bridge: kernel metadata allocation failed\n");
            ane_bridge_free(k);
            return NULL;
        }
        memcpy(k->inputBytes, input_sizes, n_inputs * sizeof(size_t));
        memcpy(k->outputBytes, output_sizes, n_outputs * sizeof(size_t));

        // Create IOSurfaces
        k->ioInputs = (IOSurfaceRef *)calloc(n_inputs, sizeof(IOSurfaceRef));
        k->ioOutputs = (IOSurfaceRef *)calloc(n_outputs, sizeof(IOSurfaceRef));
        if (!k->ioInputs || !k->ioOutputs) {
            fprintf(stderr, "ane_bridge: IOSurface slot allocation failed\n");
            ane_bridge_free(k);
            return NULL;
        }
        for (int i = 0; i < n_inputs; i++) {
            k->ioInputs[i] = create_surface(input_sizes[i]);
            if (!k->ioInputs[i]) {
                fprintf(stderr, "ane_bridge: input IOSurface allocation failed at index %d\n", i);
                ane_bridge_free(k);
                return NULL;
            }
        }
        for (int i = 0; i < n_outputs; i++) {
            k->ioOutputs[i] = create_surface(output_sizes[i]);
            if (!k->ioOutputs[i]) {
                fprintf(stderr, "ane_bridge: output IOSurface allocation failed at index %d\n", i);
                ane_bridge_free(k);
                return NULL;
            }
            k->ownedOutputSurfaces[i] = k->ioOutputs[i] ? (IOSurfaceRef)CFRetain(k->ioOutputs[i]) : NULL;
        }

        // Build request
        k->request = build_request_for_kernel(k);
        if (!k->request) {
            fprintf(stderr, "ane_bridge: request construction failed\n");
            ane_bridge_free(k);
            return NULL;
        }

        return k;
    }
}

ANEKernelHandle *ane_bridge_compile(const char *mil_text, size_t mil_len,
                                     const uint8_t *weight_data, size_t weight_len,
                                     int n_inputs, const size_t *input_sizes,
                                     int n_outputs, const size_t *output_sizes) {
    if (weight_data && weight_len > 0) {
        const char *name = "@model_path/weights/weight.bin";
        return ane_bridge_compile_multi_weights(
            mil_text, mil_len,
            &name, &weight_data, &weight_len, 1,
            n_inputs, input_sizes,
            n_outputs, output_sizes);
    } else {
        return ane_bridge_compile_multi_weights(
            mil_text, mil_len,
            NULL, NULL, NULL, 0,
            n_inputs, input_sizes,
            n_outputs, output_sizes);
    }
}

bool ane_bridge_eval(ANEKernelHandle *kernel) {
    @autoreleasepool {
        if (!kernel || !kernel->model || !kernel->request) return false;
        NSError *e = nil;
        return ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            kernel->model, @selector(evaluateWithQoS:options:request:error:),
            21, @{}, kernel->request, &e);
    }
}

bool ane_bridge_eval_batch(ANEKernelHandle **kernels, int count) {
    @autoreleasepool {
        if (!kernels || count < 0) return false;
        for (int i = 0; i < count; i++) {
            ANEKernelHandle *kernel = kernels[i];
            if (!kernel || !kernel->model || !kernel->request) return false;
            NSError *e = nil;
            BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                kernel->model, @selector(evaluateWithQoS:options:request:error:),
                21, @{}, kernel->request, &e);
            if (!ok) return false;
        }
        return true;
    }
}

bool ane_bridge_write_input(ANEKernelHandle *kernel, int idx,
                             const void *data, size_t bytes) {
    if (!kernel || idx < 0 || idx >= kernel->nInputs || !data) return false;
    if (!kernel->ioInputs || !kernel->ioInputs[idx] || !kernel->inputBytes) return false;
    if (bytes > kernel->inputBytes[idx]) {
        fprintf(stderr, "ane_bridge: input copy too large at index %d: %zu > %zu\n",
                idx, bytes, kernel->inputBytes[idx]);
        return false;
    }
    IOReturn lockStatus = IOSurfaceLock(kernel->ioInputs[idx], 0, NULL);
    if (lockStatus != kIOReturnSuccess) return false;
    void *base = IOSurfaceGetBaseAddress(kernel->ioInputs[idx]);
    if (!base) {
        IOSurfaceUnlock(kernel->ioInputs[idx], 0, NULL);
        return false;
    }
    memcpy(base, data, bytes);
    IOSurfaceUnlock(kernel->ioInputs[idx], 0, NULL);
    return true;
}

bool ane_bridge_read_output(ANEKernelHandle *kernel, int idx,
                              void *data, size_t bytes) {
    if (!kernel || idx < 0 || idx >= kernel->nOutputs || !data) return false;
    if (!kernel->ioOutputs || !kernel->ioOutputs[idx] || !kernel->outputBytes) return false;
    if (bytes > kernel->outputBytes[idx]) {
        fprintf(stderr, "ane_bridge: output copy too large at index %d: %zu > %zu\n",
                idx, bytes, kernel->outputBytes[idx]);
        return false;
    }
    IOReturn lockStatus = IOSurfaceLock(kernel->ioOutputs[idx], kIOSurfaceLockReadOnly, NULL);
    if (lockStatus != kIOReturnSuccess) return false;
    void *base = IOSurfaceGetBaseAddress(kernel->ioOutputs[idx]);
    if (!base) {
        IOSurfaceUnlock(kernel->ioOutputs[idx], kIOSurfaceLockReadOnly, NULL);
        return false;
    }
    memcpy(data, base, bytes);
    IOSurfaceUnlock(kernel->ioOutputs[idx], kIOSurfaceLockReadOnly, NULL);
    return true;
}

uint32_t ane_bridge_get_input_surface_id(ANEKernelHandle *kernel, int idx) {
    if (!kernel || idx < 0 || idx >= kernel->nInputs || !kernel->ioInputs[idx]) return 0;
    return IOSurfaceGetID(kernel->ioInputs[idx]);
}

uint32_t ane_bridge_get_output_surface_id(ANEKernelHandle *kernel, int idx) {
    if (!kernel || idx < 0 || idx >= kernel->nOutputs || !kernel->ioOutputs[idx]) return 0;
    return IOSurfaceGetID(kernel->ioOutputs[idx]);
}

size_t ane_bridge_get_output_surface_size_bytes(ANEKernelHandle *kernel, int idx) {
    if (!kernel || idx < 0 || idx >= kernel->nOutputs || !kernel->ioOutputs[idx]) return 0;
    return IOSurfaceGetAllocSize(kernel->ioOutputs[idx]);
}

int ane_bridge_consume_output_surface(
    ANEKernelHandle *kernel,
    int idx,
    ane_bridge_output_consumer_fn consumer,
    void *user_ctx,
    char *error_out,
    size_t error_out_cap)
{
    if (!kernel || idx < 0 || idx >= kernel->nOutputs || !kernel->ioOutputs[idx]) {
        set_error_message(error_out, error_out_cap, "invalid kernel or output index");
        return 2;
    }
    return consume_surface_locked(kernel->ioOutputs[idx], consumer, user_ctx, error_out, error_out_cap);
}

ANESurfaceLeaseHandle *ane_bridge_alloc_surface_lease(size_t bytes) {
    if (bytes == 0) {
        return NULL;
    }

    ANESurfaceLeaseHandle *lease = (ANESurfaceLeaseHandle *)calloc(1, sizeof(ANESurfaceLeaseHandle));
    if (!lease) {
        return NULL;
    }

    lease->surface = create_surface(bytes);
    if (!lease->surface) {
        free(lease);
        return NULL;
    }

    lease->bytes = IOSurfaceGetAllocSize(lease->surface);
    lease->surface_id = IOSurfaceGetID(lease->surface);
    return lease;
}

void ane_bridge_free_surface_lease(ANESurfaceLeaseHandle *lease) {
    if (!lease) {
        return;
    }
    if (lease->surface) {
        CFRelease(lease->surface);
    }
    free(lease);
}

uint32_t ane_bridge_surface_lease_id(ANESurfaceLeaseHandle *lease) {
    if (!lease || !lease->surface) return 0;
    return lease->surface_id;
}

size_t ane_bridge_surface_lease_size_bytes(ANESurfaceLeaseHandle *lease) {
    if (!lease || !lease->surface) return 0;
    return lease->bytes;
}

bool ane_bridge_bind_kernel_output_surface_lease(
    ANEKernelHandle *kernel,
    int idx,
    ANESurfaceLeaseHandle *lease)
{
    if (!kernel || !lease || !lease->surface) return false;
    return rebind_output_slot_with_surface(kernel, idx, lease->surface);
}

int ane_bridge_consume_surface_lease(
    ANESurfaceLeaseHandle *lease,
    ane_bridge_output_consumer_fn consumer,
    void *user_ctx,
    char *error_out,
    size_t error_out_cap)
{
    if (!lease || !lease->surface) {
        set_error_message(error_out, error_out_cap, "invalid surface lease");
        return 2;
    }
    return consume_surface_locked(lease->surface, consumer, user_ctx, error_out, error_out_cap);
}

int ane_bridge_produce_surface_lease(
    ANESurfaceLeaseHandle *lease,
    ane_bridge_output_consumer_fn producer,
    void *user_ctx,
    char *error_out,
    size_t error_out_cap)
{
    if (!lease || !lease->surface) {
        set_error_message(error_out, error_out_cap, "invalid surface lease");
        return 2;
    }
    return access_surface_locked(lease->surface, producer, user_ctx, error_out,
                                 error_out_cap, 0);
}

void ane_bridge_read_surface_lease(
    ANESurfaceLeaseHandle *lease,
    void *dst,
    size_t bytes)
{
    if (!lease || !lease->surface || !dst || bytes == 0) return;

    const IOReturn lockStatus = IOSurfaceLock(lease->surface, kIOSurfaceLockReadOnly, NULL);
    if (lockStatus != kIOReturnSuccess) return;

    @try {
        void *baseAddress = IOSurfaceGetBaseAddress(lease->surface);
        if (!baseAddress) return;
        const size_t available = IOSurfaceGetAllocSize(lease->surface);
        const size_t copyBytes = bytes < available ? bytes : available;
        memcpy(dst, baseAddress, copyBytes);
    } @finally {
        IOSurfaceUnlock(lease->surface, kIOSurfaceLockReadOnly, NULL);
    }
}

void ane_bridge_write_surface_lease(
    ANESurfaceLeaseHandle *lease,
    const void *src,
    size_t bytes)
{
    if (!lease || !lease->surface || !src || bytes == 0) return;

    const IOReturn lockStatus = IOSurfaceLock(lease->surface, 0, NULL);
    if (lockStatus != kIOReturnSuccess) return;

    @try {
        void *baseAddress = IOSurfaceGetBaseAddress(lease->surface);
        if (!baseAddress) return;
        const size_t capacity = IOSurfaceGetAllocSize(lease->surface);
        const size_t copyBytes = bytes < capacity ? bytes : capacity;
        memcpy(baseAddress, src, copyBytes);
    } @finally {
        IOSurfaceUnlock(lease->surface, 0, NULL);
    }
}

int ane_bridge_prove_output_surface_direct_map(
    ANEKernelHandle *kernel,
    int idx,
    const char *baseline_path,
    char *json_out,
    size_t json_out_cap,
    char *error_out,
    size_t error_out_cap)
{
    if (!kernel || idx < 0 || idx >= kernel->nOutputs || !kernel->ioOutputs[idx]) {
        snprintf(error_out, error_out_cap, "invalid kernel or output index");
        return 2;
    }
    if (!baseline_path || !json_out || json_out_cap == 0) {
        snprintf(error_out, error_out_cap, "invalid arguments");
        return 3;
    }

    @autoreleasepool {
        NSString *baselinePath = [NSString stringWithUTF8String:baseline_path];
        NSData *baselineData = [NSData dataWithContentsOfFile:baselinePath];
        size_t surfaceBytes = IOSurfaceGetAllocSize(kernel->ioOutputs[idx]);
        if (!baselineData || baselineData.length < surfaceBytes) {
            snprintf(error_out, error_out_cap, "baseline file missing or too small");
            return 4;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            snprintf(error_out, error_out_cap, "Metal device unavailable");
            return 5;
        }

        double startedMs = CFAbsoluteTimeGetCurrent() * 1000.0;
        IOReturn lockStatus = IOSurfaceLock(kernel->ioOutputs[idx], kIOSurfaceLockReadOnly, NULL);
        if (lockStatus != kIOReturnSuccess) {
            snprintf(error_out, error_out_cap, "IOSurfaceLock failed");
            return 6;
        }

        void *baseAddress = IOSurfaceGetBaseAddress(kernel->ioOutputs[idx]);
        if (!baseAddress) {
            IOSurfaceUnlock(kernel->ioOutputs[idx], kIOSurfaceLockReadOnly, NULL);
            snprintf(error_out, error_out_cap, "IOSurfaceGetBaseAddress failed");
            return 7;
        }

        id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:baseAddress
                                                         length:surfaceBytes
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        double directMapSetupMs = (CFAbsoluteTimeGetCurrent() * 1000.0) - startedMs;
        if (buffer == nil) {
            IOSurfaceUnlock(kernel->ioOutputs[idx], kIOSurfaceLockReadOnly, NULL);
            snprintf(error_out, error_out_cap, "newBufferWithBytesNoCopy failed");
            return 8;
        }

        bool bytewiseEqual = false;
        void *bufferContents = [buffer contents];
        if (bufferContents) {
            bytewiseEqual = memcmp(bufferContents, baselineData.bytes, surfaceBytes) == 0;
        }

        IOSurfaceUnlock(kernel->ioOutputs[idx], kIOSurfaceLockReadOnly, NULL);
        snprintf(
            json_out,
            json_out_cap,
            "{\"surface_id\":%u,\"surface_bytes\":%zu,\"direct_map_setup_ms\":%.6f,\"bytewise_equal\":%s}",
            IOSurfaceGetID(kernel->ioOutputs[idx]),
            surfaceBytes,
            directMapSetupMs,
            bytewiseEqual ? "true" : "false"
        );
        return 0;
    }
}

bool ane_bridge_bind_input_surface_id(ANEKernelHandle *kernel, int idx, uint32_t surface_id) {
    return rebind_surface_slot(kernel, true, idx, surface_id);
}

bool ane_bridge_bind_output_surface_id(ANEKernelHandle *kernel, int idx, uint32_t surface_id) {
    return rebind_surface_slot(kernel, false, idx, surface_id);
}

// Direct ref-based surface rebind. Bypasses IOSurfaceGetID → IOSurfaceLookup
// round-trip, which fails for private (non-globally-registered) IOSurfaces
// where IOSurfaceGetID returns 0. Used by the fused-block ping-pong chain.
bool ane_bridge_rebind_surface(
    ANEKernelHandle *dst, bool dst_is_input, int dst_idx,
    ANEKernelHandle *src, bool src_is_input, int src_idx)
{
    if (!dst || !src) return false;
    int dst_count = dst_is_input ? dst->nInputs : dst->nOutputs;
    int src_count = src_is_input ? src->nInputs : src->nOutputs;
    if (dst_idx < 0 || dst_idx >= dst_count) return false;
    if (src_idx < 0 || src_idx >= src_count) return false;

    IOSurfaceRef surface = src_is_input ? src->ioInputs[src_idx] : src->ioOutputs[src_idx];
    if (!surface) return false;

    IOSurfaceRef *dst_slots = dst_is_input ? dst->ioInputs : dst->ioOutputs;
    // NOTE: we do NOT check IOSurfaceGetAllocSize here. ANE-managed kernel
    // surfaces report alloc_size = 0 because their memory is managed by the
    // ANE runtime, not by IOKit. The ANE guarantees that compiled kernels with
    // matching shapes have compatible surface sizes. The alloc-size check in
    // rebind_surface_slot (ID-based path) is appropriate for lease-owned
    // surfaces, but not for inter-kernel surface chaining.

    IOSurfaceRef oldSurface = dst_slots[dst_idx];
    CFRetain(surface);
    dst_slots[dst_idx] = surface;

    id newRequest = build_request_for_kernel(dst);
    if (!newRequest) {
        dst_slots[dst_idx] = oldSurface;
        CFRelease(surface);
        return false;
    }

    if (oldSurface) CFRelease(oldSurface);
    dst->request = newRequest;
    return true;
}

void ane_bridge_free(ANEKernelHandle *kernel) {
    @autoreleasepool {
        if (!kernel) return;
        NSError *e = nil;
        if (kernel->model) {
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                kernel->model, @selector(unloadWithQoS:error:), 21, &e);
        }
        if (kernel->ioInputs) {
            for (int i = 0; i < kernel->nInputs; i++)
                if (kernel->ioInputs[i]) CFRelease(kernel->ioInputs[i]);
        }
        if (kernel->ioOutputs) {
            for (int i = 0; i < kernel->nOutputs; i++)
                if (kernel->ioOutputs[i]) CFRelease(kernel->ioOutputs[i]);
        }
        if (kernel->ownedOutputSurfaces) {
            for (int i = 0; i < kernel->nOutputs; i++)
                if (kernel->ownedOutputSurfaces[i]) CFRelease(kernel->ownedOutputSurfaces[i]);
        }
        if (kernel->tmpDir) {
            [[NSFileManager defaultManager] removeItemAtPath:kernel->tmpDir error:nil];
        }
        free(kernel->ioInputs);
        free(kernel->ioOutputs);
        free(kernel->ownedOutputSurfaces);
        free(kernel->inputBytes);
        free(kernel->outputBytes);

        // Explicitly nil Objective-C objects to trigger ARC release before freeing struct
        kernel->model = nil;
        kernel->request = nil;
        kernel->tmpDir = nil;

        free(kernel);
    }
}

int ane_bridge_get_compile_count(void) {
    return g_compile_count;
}

void ane_bridge_reset_compile_count(void) {
    g_compile_count = 0;
}

uint8_t *ane_bridge_build_weight_blob(const float *src, int rows, int cols,
                                       size_t *out_len) {
    int wsize = rows * cols * 2; // fp16
    int total = 128 + wsize;
    uint8_t *buf = (uint8_t *)calloc(total, 1);

    // ANE blob header
    buf[0] = 0x01; buf[4] = 0x02;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 0x01;
    *(uint32_t*)(buf + 72) = wsize;
    *(uint32_t*)(buf + 80) = 128;

    // Convert float32 -> float16
    _Float16 *fp16 = (_Float16 *)(buf + 128);
    for (int i = 0; i < rows * cols; i++) {
        fp16[i] = (_Float16)src[i];
    }

    *out_len = total;
    return buf;
}

uint8_t *ane_bridge_build_weight_blob_transposed(const float *src, int rows, int cols,
                                                   size_t *out_len) {
    int wsize = rows * cols * 2;
    int total = 128 + wsize;
    uint8_t *buf = (uint8_t *)calloc(total, 1);

    buf[0] = 0x01; buf[4] = 0x02;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 0x01;
    *(uint32_t*)(buf + 72) = wsize;
    *(uint32_t*)(buf + 80) = 128;

    _Float16 *fp16 = (_Float16 *)(buf + 128);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            fp16[j * rows + i] = (_Float16)src[i * cols + j];

    *out_len = total;
    return buf;
}
