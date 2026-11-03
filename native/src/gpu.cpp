#include "gpu.h"

#include "gpu_kernels_ptx.h"  // kKernelsPtx: native/src/gpu_kernels.ptx, embedded by CMake
#include "resize_filter.h"

namespace pp {
namespace {

struct GpuState {
    std::mutex m;
    std::atomic<bool> ready{false};
    int device = 0;
    cudaDeviceProp prop{};
    cudaMemPool_t pool = nullptr;  // our own stream-ordered pool (null: device lacks pools -> cudaMalloc)
    cudaLibrary_t library = nullptr;  // gpu_kernels.ptx, JIT-compiled for this GPU by the driver
    cudaKernel_t unpack = nullptr, resize1 = nullptr, resize3 = nullptr, resize4 = nullptr;
};
GpuState g;

std::wstring cuda_msg(cudaError_t e) { return widen(cudaGetErrorName(e)) + L": " + widen(cudaGetErrorString(e)); }

// Loads the embedded kernels once; the driver JIT-compiles the PTX for the device (and caches it).
cudaError_t load_kernels() {
    if (g.resize4) return cudaSuccess;
    cudaError_t e = cudaLibraryLoadData(&g.library, kKernelsPtx, nullptr, nullptr, 0, nullptr, nullptr, 0);
    if (e == cudaSuccess) e = cudaLibraryGetKernel(&g.unpack, g.library, "pp_unpack_rows");
    if (e == cudaSuccess) e = cudaLibraryGetKernel(&g.resize1, g.library, "pp_resize_cubic1");
    if (e == cudaSuccess) e = cudaLibraryGetKernel(&g.resize3, g.library, "pp_resize_cubic3");
    if (e == cudaSuccess) e = cudaLibraryGetKernel(&g.resize4, g.library, "pp_resize_cubic4");
    if (e != cudaSuccess) g.resize4 = nullptr;
    return e;
}

// PackBits rows -> 8-bit planes, see pp_unpack_rows in gpu_kernels.cu.
cudaError_t gpu_unpack(const uint8_t* comp, const uint32_t* offsets, int rows, int rowBytes, int shift, uint8_t* planes,
                       int width, uint32_t* firstBad, cudaStream_t s) {
    void* args[] = {&comp, &offsets, &rows, &rowBytes, &shift, &planes, &width, &firstBad};
    return cudaLaunchKernel(reinterpret_cast<const void*>(g.unpack), dim3(unsigned(rows + 3) / 4), dim3(128), args, 0, s);  // a warp per row
}

// `channels` 8-bit planes of w x h (`plane` bytes apart) -> dw x dh RGBA (gray expanded) or CMYK.
cudaError_t gpu_resize(int channels, const uint8_t* planes, size_t plane, int w, int h, uint8_t* dst, size_t dstPitch, int dw,
                       int dh, cudaStream_t s) {
    CubicFilter f = resize_filter(w, dw);
    void* args[] = {&planes, &plane, &w, &h, &dst, &dstPitch, &dw, &dh, &f.B, &f.C};
    const cudaKernel_t k = channels == 1 ? g.resize1 : channels == 3 ? g.resize3 : g.resize4;
    const dim3 block(32, 8), grid(unsigned(dw + 31) / 32, unsigned(dh + 7) / 8);
    return cudaLaunchKernel(reinterpret_cast<const void*>(k), grid, block, args, 0, s);
}

template <class T>
cudaError_t dev_alloc(T** p, size_t bytes, cudaStream_t s) {
    void* v = nullptr;
    cudaError_t e = g.pool ? cudaMallocFromPoolAsync(&v, bytes, g.pool, s) : cudaMalloc(&v, bytes);
    *p = static_cast<T*>(v);
    if (e != cudaSuccess) cudaGetLastError();  // allocation errors aren't sticky; clear them
    return e;
}

template <class T>
void dev_free(T*& p, cudaStream_t s) {
    if (!p) return;
    if (g.pool) {
        cudaFreeAsync(p, s);  // stream-ordered: the memory is reusable once the stream gets there
    } else {
        cudaStreamSynchronize(s);
        cudaFree(p);
    }
    p = nullptr;
}

// Unpacks and resizes a tiny image: proves the kernels load and run on this GPU (and warms them
// up). 3 planes of 64 x 64, every row one PackBits run of 64 x 0x80, must come out flat.
bool gpu_self_test(std::wstring& error) {
    const int w = 64, h = 64, rows = 3 * h, dw = 37, dh = 37;
    std::vector<uint8_t> comp;
    std::vector<uint32_t> offsets;
    for (int r = 0; r < rows; ++r) {
        offsets.push_back(uint32_t(comp.size()));
        comp.push_back(uint8_t(257 - w));
        comp.push_back(0x80);
    }
    offsets.push_back(uint32_t(comp.size()));
    uint8_t *dComp = nullptr, *dPlanes = nullptr, *dDst = nullptr;
    uint32_t *dOffsets = nullptr, *dBad = nullptr;
    cudaStream_t s = nullptr;
    std::vector<uint8_t> out(size_t(dw) * dh * kPixelBytes);
    uint32_t bad = 0;
    cudaError_t e = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    if (e == cudaSuccess) e = dev_alloc(&dComp, comp.size(), s);
    if (e == cudaSuccess) e = dev_alloc(&dOffsets, offsets.size() * 4, s);
    if (e == cudaSuccess) e = dev_alloc(&dBad, 4, s);
    if (e == cudaSuccess) e = dev_alloc(&dPlanes, size_t(rows) * w, s);
    if (e == cudaSuccess) e = dev_alloc(&dDst, out.size(), s);
    if (e == cudaSuccess) e = cudaMemcpyAsync(dComp, comp.data(), comp.size(), cudaMemcpyHostToDevice, s);
    if (e == cudaSuccess) e = cudaMemcpyAsync(dOffsets, offsets.data(), offsets.size() * 4, cudaMemcpyHostToDevice, s);
    if (e == cudaSuccess) e = cudaMemsetAsync(dBad, 0xFF, 4, s);
    if (e == cudaSuccess) e = gpu_unpack(dComp, dOffsets, rows, w, 0, dPlanes, w, dBad, s);
    if (e == cudaSuccess) e = gpu_resize(3, dPlanes, size_t(w) * h, w, h, dDst, size_t(dw) * kPixelBytes, dw, dh, s);
    if (e == cudaSuccess) e = cudaMemcpyAsync(out.data(), dDst, out.size(), cudaMemcpyDeviceToHost, s);
    if (e == cudaSuccess) e = cudaMemcpyAsync(&bad, dBad, 4, cudaMemcpyDeviceToHost, s);
    if (e == cudaSuccess) e = cudaStreamSynchronize(s);
    bool ok = false;
    if (e != cudaSuccess) error = L"GPU self-test: " + cuda_msg(e);
    else if (bad != UINT32_MAX) error = L"GPU self-test: PackBits kernel reported a bad row";
    else {
        ok = true;
        for (size_t i = 0; i < out.size(); ++i) ok &= out[i] == (i % 4 == 3 ? 0xFF : 0x80);
        if (!ok) error = L"GPU self-test: wrong resize result";
    }
    if (s) {
        dev_free(dComp, s);
        dev_free(dOffsets, s);
        dev_free(dBad, s);
        dev_free(dPlanes, s);
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

    if ((e = load_kernels()) != cudaSuccess) return fail(PP_ERR_KERNEL_UNAVAILABLE, L"loading the GPU kernels: " + cuda_msg(e));
    std::wstring err;
    if (!gpu_self_test(err)) return fail(PP_ERR_KERNEL_UNAVAILABLE, err);
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

std::wstring gpu_description() { return widen(g.prop.name); }

// ---- GpuStreamer ------------------------------------------------------------------------------

bool GpuStreamer::supports(const PsdHeader& h) {
    if (h.compression == 1) return h.depth == 8 || h.depth == 16;
    return h.compression == 0 && h.depth == 8;
}

uint64_t GpuStreamer::device_bytes(const PsdHeader& h, int dw, int dh) {
    const uint64_t rows = uint64_t(h.sourceChannels()) * h.height;
    const uint64_t planes = rows * h.width;
    // The compressed colour channels are part of the image data section.
    const uint64_t packed = h.compression == 1 ? h.imageDataBytes + (rows + 1) * 4 + 4 : 0;
    return planes + packed + gpu_pitch(uint64_t(dw) * kPixelBytes) * dh;
}

GpuStreamer::~GpuStreamer() {
    if (!stream_) return;
    abort();
    for (int i = 0; i < 2; ++i) {
        if (uploaded_[i]) cudaEventDestroy(uploaded_[i]);
        if (band_[i]) cudaFreeHost(band_[i]);
    }
    if (status_) cudaFreeHost(status_);
    cudaStreamDestroy(stream_);
}

bool GpuStreamer::init(std::wstring& error) {
    if (!g.ready) { error = L"CUDA not initialised"; return false; }
    cudaSetDevice(g.device);  // current device is per host thread
    // Waits use events with cudaEventBlockingSync: the default spin-wait would keep 16 extract
    // threads busy while the encoders need the cores.
    constexpr unsigned kEventFlags = cudaEventDisableTiming | cudaEventBlockingSync;
    cudaError_t e = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    if (e == cudaSuccess) e = cudaHostAlloc(reinterpret_cast<void**>(&status_), sizeof(uint32_t), cudaHostAllocDefault);
    for (int i = 0; i < 2 && e == cudaSuccess; ++i) {
        e = cudaEventCreateWithFlags(&uploaded_[i], kEventFlags);
        if (e == cudaSuccess) e = cudaHostAlloc(reinterpret_cast<void**>(&band_[i]), kGpuBandBytes, cudaHostAllocDefault);
    }
    if (e != cudaSuccess) { error = L"GPU stream setup: " + cuda_msg(e); return false; }
    return true;
}

cudaError_t GpuStreamer::send(uint8_t* devDst, size_t n) {
    cudaError_t e = cudaMemcpyAsync(devDst, band_[cur_], n, cudaMemcpyHostToDevice, stream_);
    if (e == cudaSuccess) e = cudaEventRecord(uploaded_[cur_], stream_);
    cur_ ^= 1;
    // Before filling the other band, its previous upload (one band ago) must be done.
    const cudaError_t w = cudaEventSynchronize(uploaded_[cur_]);
    return e != cudaSuccess ? e : w;
}

bool GpuStreamer::upload(const FileReader& f, const PsdHeader& h, FileStatus& file, std::wstring& error) {
    file = {};
    abort();
    w_ = h.width;
    h_ = h.height;
    planes_ = h.sourceChannels();
    depth_ = h.depth;
    compression_ = h.compression;
    const uint64_t rows = uint64_t(planes_) * h_;
    cudaError_t e = dev_alloc(&dPlanes_, rows * w_, stream_);

    // Streams `bytes` of the file from `pos` through the pinned bands to `dst`.
    auto stream_file = [&](uint64_t pos, uint64_t bytes, uint8_t* dst) {
        for (uint64_t done = 0; done < bytes && e == cudaSuccess;) {
            const size_t n = size_t(std::min<uint64_t>(kGpuBandBytes, bytes - done));
            if (!f.read_exact(pos + done, band_[cur_], n)) {
                file = {PP_FILE_TRUNCATED, L"file truncated inside the composite"};
                return;
            }
            e = send(dst + done, n);
            done += n;
        }
    };

    if (compression_ == 0) {  // raw 8-bit: the colour planes are contiguous in the file
        if (h.imageDataPos + rows * w_ > h.fileSize) file = {PP_FILE_TRUNCATED, L"file ends inside the composite"};
        else stream_file(h.imageDataPos, rows * w_, dPlanes_);
    } else {
        RleLayout lay;
        file = read_rle_layout(f, h, lay);
        if (file.ok() && lay.dataBytes > UINT32_MAX) {
            error = L"compressed composite over 4 GB";  // 32-bit row offsets; the CPU path takes it
            e = cudaErrorInvalidValue;
        }
        if (file.ok() && e == cudaSuccess) e = dev_alloc(&dComp_, std::max<uint64_t>(lay.dataBytes, 1), stream_);
        if (file.ok() && e == cudaSuccess) e = dev_alloc(&dOffsets_, (rows + 1) * 4, stream_);
        if (file.ok() && e == cudaSuccess) e = dev_alloc(&dBad_, 4, stream_);
        if (file.ok() && e == cudaSuccess) e = cudaMemsetAsync(dBad_, 0xFF, 4, stream_);
        // Row offsets (prefix sums of the RLE table), through the bands too.
        uint64_t acc = 0;
        for (uint64_t i = 0; file.ok() && e == cudaSuccess && i <= rows;) {
            const uint64_t i0 = i, n = std::min<uint64_t>(rows + 1 - i, kGpuBandBytes / 4);
            uint32_t* o = reinterpret_cast<uint32_t*>(band_[cur_]);
            for (uint64_t j = 0; j < n; ++j, ++i) {
                o[j] = uint32_t(acc);
                if (i < rows) acc += lay.counts[i];
            }
            e = send(reinterpret_cast<uint8_t*>(dOffsets_ + i0), size_t(n * 4));
        }
        if (file.ok()) stream_file(lay.dataPos, lay.dataBytes, dComp_);
    }
    if (file.ok() && e == cudaSuccess) return true;
    if (e != cudaSuccess && error.empty()) error = L"GPU upload: " + cuda_msg(e);
    abort();
    return false;
}

bool GpuStreamer::finish(int dw, int dh, uint8_t* dst, FileStatus& file, std::wstring& error) {
    file = {};
    cudaError_t e = cudaSuccess;
    if (compression_ == 1) {
        const int shift = depth_ == 16 ? 1 : 0;
        const int rowBytes = int(uint64_t(w_) * depth_ / 8);
        e = gpu_unpack(dComp_, dOffsets_, planes_ * int(h_), rowBytes, shift, dPlanes_, int(w_), dBad_, stream_);
        if (e == cudaSuccess) e = cudaMemcpyAsync(status_, dBad_, 4, cudaMemcpyDeviceToHost, stream_);
        dev_free(dComp_, stream_);
        dev_free(dOffsets_, stream_);
        dev_free(dBad_, stream_);
    }
    uint8_t* dDst = nullptr;
    const size_t total = size_t(dw) * kPixelBytes * dh;  // packed rows
    if (e == cudaSuccess) e = dev_alloc(&dDst, total, stream_);
    if (e == cudaSuccess)
        e = gpu_resize(planes_, dPlanes_, size_t(w_) * h_, int(w_), int(h_), dDst, size_t(dw) * kPixelBytes, dw, dh, stream_);
    dev_free(dPlanes_, stream_);
    // Pull through the pinned bands, not straight into the caller's pageable buffer: the driver
    // stages a pageable download synchronously, which stalls every other thread's CUDA calls.
    // Chunk k is copied out of one band while chunk k + 1 downloads into the other.
    size_t pendingOff = 0, pending = 0;  // previous chunk, complete in band_[cur_ ^ 1]
    for (size_t off = 0; e == cudaSuccess && off < total; off += pending) {
        const size_t n = std::min(kGpuBandBytes, total - off);
        e = cudaMemcpyAsync(band_[cur_], dDst + off, n, cudaMemcpyDeviceToHost, stream_);
        if (e == cudaSuccess) e = cudaEventRecord(uploaded_[cur_], stream_);
        if (pending) std::memcpy(dst + pendingOff, band_[cur_ ^ 1], pending);
        if (e == cudaSuccess) e = cudaEventSynchronize(uploaded_[cur_]);
        pendingOff = off;
        pending = n;
        cur_ ^= 1;
    }
    if (e == cudaSuccess && pending) std::memcpy(dst + pendingOff, band_[cur_ ^ 1], pending);
    dev_free(dDst, stream_);
    if (e != cudaSuccess) {
        error = L"GPU: " + cuda_msg(e);
        cudaGetLastError();
        abort();
        return false;
    }
    if (compression_ == 1 && *status_ != UINT32_MAX) {
        file = {PP_FILE_CORRUPT, std::format(L"corrupt RLE data in composite channel {} row {}", *status_ / h_, *status_ % h_)};
        return false;
    }
    return true;
}

void GpuStreamer::abort() {
    if (!stream_) return;
    cudaStreamSynchronize(stream_);  // in-flight uploads still read the bands
    dev_free(dComp_, stream_);
    dev_free(dOffsets_, stream_);
    dev_free(dBad_, stream_);
    dev_free(dPlanes_, stream_);
}

}  // namespace pp
