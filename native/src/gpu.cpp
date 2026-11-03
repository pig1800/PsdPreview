#include "gpu.h"

#include <nppcore.h>
#include <nppi_geometry_transforms.h>

#ifndef PP_NPP_DLL_SUFFIX
#error "PP_NPP_DLL_SUFFIX must be defined (CUDA major version, e.g. 13)"
#endif
#define PP_WIDEN2(x) L##x
#define PP_WIDEN(x) PP_WIDEN2(x)

namespace pp {
namespace {

struct GpuState {
    std::mutex m;
    std::atomic<bool> ready{false};
    int device = 0;
    cudaDeviceProp prop{};
    cudaMemPool_t pool = nullptr;  // our own stream-ordered pool (null: device lacks pools -> cudaMalloc)
};
GpuState g;

std::wstring cuda_msg(cudaError_t e) { return widen(cudaGetErrorName(e)) + L": " + widen(cudaGetErrorString(e)); }

// NPP is delay-loaded so psdpreview_native.dll loads on machines without CUDA. Load it
// explicitly first (preferring our own directory) so a missing DLL is an error code
// instead of a delay-load exception.
bool load_npp(std::wstring& error) {
    wchar_t self[MAX_PATH * 4] = {};
    HMODULE hm = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCWSTR>(&load_npp), &hm);
    GetModuleFileNameW(hm, self, DWORD(std::size(self)));
    std::wstring dir(self);
    dir.resize(dir.find_last_of(L"\\/") + 1);

    for (const wchar_t* name : {L"nppc64_" PP_WIDEN(PP_NPP_DLL_SUFFIX) L".dll", L"nppig64_" PP_WIDEN(PP_NPP_DLL_SUFFIX) L".dll"}) {
        if (GetModuleHandleW(name)) continue;
        if (LoadLibraryExW((dir + name).c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH)) continue;
        if (LoadLibraryW(name)) continue;
        error = std::format(L"{} not found (put it next to psdpreview_native.dll): {}", name, win32_message(GetLastError()));
        return false;
    }
    return true;
}

NppStreamContext make_context(cudaStream_t stream) {
    NppStreamContext c{};
    c.hStream = stream;
    c.nCudaDeviceId = g.device;
    c.nMultiProcessorCount = g.prop.multiProcessorCount;
    c.nMaxThreadsPerMultiProcessor = g.prop.maxThreadsPerMultiProcessor;
    c.nMaxThreadsPerBlock = g.prop.maxThreadsPerBlock;
    c.nSharedMemPerBlock = g.prop.sharedMemPerBlock;
    c.nCudaDevAttrComputeCapabilityMajor = g.prop.major;
    c.nCudaDevAttrComputeCapabilityMinor = g.prop.minor;
    cudaStreamGetFlags(stream, &c.nStreamFlags);
    return c;
}

// RGBA8888 bicubic. (nppiResize maps pixels slightly differently from IPP; visually identical.)
NppStatus npp_resize(const uint8_t* src, int srcStep, int w, int h, uint8_t* dst, int dstStep, int dw, int dh,
                     const NppStreamContext& ctx) {
    const NppiSize ss{w, h}, ds{dw, dh};
    const NppiRect sr{0, 0, w, h}, dr{0, 0, dw, dh};
    return nppiResize_8u_C4R_Ctx(src, srcStep, ss, sr, dst, dstStep, ds, dr, NPPI_INTER_CUBIC, ctx);
}

cudaError_t dev_alloc(uint8_t** p, size_t bytes, cudaStream_t s) {
    void* v = nullptr;
    cudaError_t e = g.pool ? cudaMallocFromPoolAsync(&v, bytes, g.pool, s) : cudaMalloc(&v, bytes);
    *p = static_cast<uint8_t*>(v);
    return e;
}

void dev_free(uint8_t*& p, cudaStream_t s) {
    if (!p) return;
    if (g.pool) {
        cudaFreeAsync(p, s);  // stream-ordered: the memory is reusable once the stream gets there
    } else {
        cudaStreamSynchronize(s);
        cudaFree(p);
    }
    p = nullptr;
}

// Runs one small resize: proves NPP kernels load on this GPU and warms them up.
bool npp_self_test(std::wstring& error) {
    const int w = 64, h = 64, dw = 37, dh = 37, px = kPixelBytes;
    uint8_t *dSrc = nullptr, *dDst = nullptr;
    cudaStream_t s = nullptr;
    bool ok = false;
    cudaError_t e = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    if (e == cudaSuccess) e = dev_alloc(&dSrc, w * h * px, s);
    if (e == cudaSuccess) e = dev_alloc(&dDst, dw * dh * px, s);
    if (e == cudaSuccess) e = cudaMemsetAsync(dSrc, 0x80, w * h * px, s);
    if (e == cudaSuccess) {
        NppStatus ns = npp_resize(dSrc, w * px, w, h, dDst, dw * px, dw, dh, make_context(s));
        e = cudaStreamSynchronize(s);
        if (ns < NPP_SUCCESS) error = std::format(L"NPP self-test resize failed (NppStatus {})", int(ns));
        else if (e != cudaSuccess) error = L"NPP self-test: " + cuda_msg(e);
        else ok = true;
    } else {
        error = L"NPP self-test setup: " + cuda_msg(e);
    }
    if (s) {
        dev_free(dSrc, s);
        dev_free(dDst, s);
        cudaStreamSynchronize(s);
        cudaStreamDestroy(s);
    }
    return ok;
}

void fill_info(PP_DeviceInfo& out) {
    out.computeMajor = g.prop.major;
    out.computeMinor = g.prop.minor;
    out.multiProcessors = g.prop.multiProcessorCount;
    copy_str(out.name, widen(g.prop.name));
    size_t freeB = 0, totalB = 0;
    cudaMemGetInfo(&freeB, &totalB);
    out.freeMemory = freeB;
    out.totalMemory = totalB;
    if (const NppLibraryVersion* v = nppGetLibVersion()) {
        out.nppMajor = v->major;
        out.nppMinor = v->minor;
        out.nppBuild = v->build;
    }
}

// A private pool (not the device default pool, which other code in the process may tune)
// that keeps freed blocks for reuse until gpu_trim().
void create_pool(int device) {
    int supported = 0;
    cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, device);
    if (!supported) return;
    cudaMemPoolProps props{};
    props.allocType = cudaMemAllocationTypePinned;
    props.handleTypes = cudaMemHandleTypeNone;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = device;
    if (cudaMemPoolCreate(&g.pool, &props) != cudaSuccess) {
        g.pool = nullptr;
        return;
    }
    uint64_t keep = UINT64_MAX;
    cudaMemPoolSetAttribute(g.pool, cudaMemPoolAttrReleaseThreshold, &keep);
}

}  // namespace

int gpu_init(int device, PP_DeviceInfo* info) {
    std::lock_guard lk(g.m);
    PP_DeviceInfo scratch{};
    PP_DeviceInfo& out = info ? *info : scratch;
    out = {};
    out.deviceId = device;
    auto fail = [&](int code, const std::wstring& msg) {
        copy_str(out.message, msg);
        return code;
    };

    cudaDriverGetVersion(&out.cudaDriverVersion);
    cudaRuntimeGetVersion(&out.cudaRuntimeVersion);
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess) return fail(PP_ERR_CUDA_UNAVAILABLE, cuda_msg(e));
    if (n == 0) return fail(PP_ERR_CUDA_UNAVAILABLE, L"no CUDA device found");
    if (device < 0 || device >= n) return fail(PP_ERR_INVALID_ARGUMENT, std::format(L"device {} does not exist ({} found)", device, n));

    if (g.ready && g.device == device) {
        cudaSetDevice(device);
        fill_info(out);
        copy_str(out.message, L"already initialised");
        return PP_OK;
    }
    g.ready = false;
    if ((e = cudaSetDevice(device)) != cudaSuccess) return fail(PP_ERR_CUDA_FAILED, cuda_msg(e));
    if ((e = cudaFree(nullptr)) != cudaSuccess) return fail(PP_ERR_CUDA_FAILED, cuda_msg(e));  // creates the context
    if ((e = cudaGetDeviceProperties(&g.prop, device)) != cudaSuccess) return fail(PP_ERR_CUDA_FAILED, cuda_msg(e));
    g.device = device;
    copy_str(out.name, widen(g.prop.name));
    if (!g.pool) create_pool(device);

    std::wstring err;
    if (!load_npp(err)) return fail(PP_ERR_NPP_UNAVAILABLE, err);
    if (!npp_self_test(err)) return fail(PP_ERR_NPP_UNAVAILABLE, err);
    fill_info(out);
    g.ready = true;
    return PP_OK;
}

bool gpu_ready() { return g.ready; }

void gpu_shutdown() {
    std::lock_guard lk(g.m);
    g.ready = false;  // the primary context is left alone; other code in the process may use it
    gpu_trim();
}

void gpu_trim() {
    if (g.pool) cudaMemPoolTrimTo(g.pool, 0);
}

bool gpu_memory(uint64_t& freeBytes, uint64_t& totalBytes) {
    freeBytes = totalBytes = 0;
    if (!g.ready) return false;
    cudaSetDevice(g.device);
    size_t f = 0, t = 0;
    if (cudaMemGetInfo(&f, &t) != cudaSuccess) return false;
    freeBytes = f;
    totalBytes = t;
    return true;
}

std::wstring gpu_description() {
    const NppLibraryVersion* v = g.ready ? nppGetLibVersion() : nullptr;
    return v ? std::format(L"{} (NPP {}.{}.{})", widen(g.prop.name), v->major, v->minor, v->build) : widen(g.prop.name);
}

// ---- GpuStreamer ------------------------------------------------------------------------------

GpuStreamer::~GpuStreamer() {
    if (!stream_) return;
    abort();
    cudaStreamSynchronize(stream_);
    for (int i = 0; i < 2; ++i) {
        if (uploaded_[i]) cudaEventDestroy(uploaded_[i]);
        if (band_[i]) cudaFreeHost(band_[i]);
    }
    cudaStreamDestroy(stream_);
}

bool GpuStreamer::init(std::wstring& error) {
    if (!g.ready) { error = L"CUDA not initialised"; return false; }
    cudaSetDevice(g.device);  // current device is per host thread
    cudaError_t e = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    for (int i = 0; i < 2 && e == cudaSuccess; ++i) {
        e = cudaEventCreateWithFlags(&uploaded_[i], cudaEventDisableTiming);
        if (e == cudaSuccess) e = cudaHostAlloc(reinterpret_cast<void**>(&band_[i]), kGpuBandBytes, cudaHostAllocDefault);
    }
    if (e != cudaSuccess) { error = L"GPU stream setup: " + cuda_msg(e); return false; }
    ctx_ = make_context(stream_);
    return true;
}

bool GpuStreamer::begin(uint32_t w, uint32_t h, std::wstring& error) {
    w_ = w;
    h_ = h;
    rowBytes_ = size_t(w) * kPixelBytes;
    pitch_ = gpu_pitch(rowBytes_);
    bandRows_ = uint32_t(std::max<size_t>(1, kGpuBandBytes / rowBytes_));
    bandY0_ = 0;
    cur_ = 0;
    err_ = cudaSuccess;
    if (rowBytes_ > kGpuBandBytes) { error = L"row too wide for the GPU band buffer"; return false; }
    cudaError_t e = dev_alloc(&dSrc_, pitch_ * h, stream_);
    if (e != cudaSuccess) {
        cudaGetLastError();  // clear the (non-sticky) allocation error
        error = L"GPU memory: " + cuda_msg(e);
        return false;
    }
    return true;
}

uint8_t* GpuStreamer::row(uint32_t y) {
    if (y - bandY0_ == bandRows_) flush(y);  // current band is full: send it
    return band_[cur_] + size_t(y - bandY0_) * rowBytes_;
}

void GpuStreamer::flush(uint32_t yEnd) {
    const uint32_t n = yEnd - bandY0_;
    if (n && err_ == cudaSuccess) {
        err_ = cudaMemcpy2DAsync(dSrc_ + size_t(bandY0_) * pitch_, pitch_, band_[cur_], rowBytes_, rowBytes_, n,
                                 cudaMemcpyHostToDevice, stream_);
        if (err_ == cudaSuccess) err_ = cudaEventRecord(uploaded_[cur_], stream_);
    }
    bandY0_ = yEnd;
    cur_ ^= 1;
    // Before decoding into the other buffer, its previous upload (one band ago) must be done.
    cudaError_t e = cudaEventSynchronize(uploaded_[cur_]);
    if (err_ == cudaSuccess) err_ = e;
}

bool GpuStreamer::finish(int dw, int dh, uint8_t* dst, std::wstring& error) {
    flush(h_);  // the last (partial) band
    uint8_t* dDst = nullptr;
    const size_t dstRow = size_t(dw) * kPixelBytes, dstPitch = gpu_pitch(dstRow);
    cudaError_t e = err_;
    if (e == cudaSuccess) e = dev_alloc(&dDst, dstPitch * dh, stream_);
    if (e == cudaSuccess) {
        const NppStatus ns = npp_resize(dSrc_, int(pitch_), int(w_), int(h_), dDst, int(dstPitch), dw, dh, ctx_);
        if (ns < NPP_SUCCESS) {
            error = std::format(L"NPP resize failed (NppStatus {})", int(ns));
            e = cudaErrorUnknown;
        }
    }
    dev_free(dSrc_, stream_);
    // Pull straight into the caller's (pageable) buffer; returns once the copy is done.
    if (e == cudaSuccess) e = cudaMemcpy2DAsync(dst, dstRow, dDst, dstPitch, dstRow, dh, cudaMemcpyDeviceToHost, stream_);
    dev_free(dDst, stream_);
    const cudaError_t s = cudaStreamSynchronize(stream_);
    if (e == cudaSuccess) e = s;
    if (e != cudaSuccess) {
        if (error.empty()) error = L"GPU: " + cuda_msg(e);
        cudaGetLastError();
        return false;
    }
    return true;
}

void GpuStreamer::abort() {
    if (stream_) cudaStreamSynchronize(stream_);
    dev_free(dSrc_, stream_);
    err_ = cudaSuccess;
}

}  // namespace pp
