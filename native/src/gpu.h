#pragma once
#include "common.h"
#include "psd.h"

namespace pp {

// Pinned (CUDA) / upload-heap (D3D12) bytes per band buffer; a streamer has two.
constexpr size_t kGpuBandBytes = 4u << 20;

// A resized image in memory the GPU wrote and the CPU reads in place (hybrid path, see below):
// the encoder takes it as it is. Destroying it hands the memory back to the backend.
class GpuImage {
public:
    virtual ~GpuImage() = default;
    uint8_t* data = nullptr;  // dw x dh x 4 bytes
};

// Takes one image at a time through the GPU, push -> process -> pull:
//   upload  reads the composite's colour channels from the file to the GPU, still
//           PackBits-compressed for RLE (~1.35 bytes/px for webtoon pages instead of 4 decoded,
//           and no CPU work beyond the read)
//   finish  unpacks the rows into 8-bit planes on the GPU, resizes them with the filter of
//           resize_filter.h and brings the result back into the caller's buffer
// Used by one thread at a time (the engine shares a pool of them between its extract threads).
class GpuStreamer {
public:
    virtual ~GpuStreamer() = default;
    // A file problem (truncated, blank...) is returned in `file`; false with `file` ok means a
    // GPU problem (`error`): redo the image on the CPU.
    virtual bool upload(const FileReader& f, const PsdHeader& h, FileStatus& file, std::wstring& error) = 0;
    // dst: dw x dh x 4 bytes. Corrupt PackBits data is a file problem (`file`); false with `file`
    // ok is a GPU problem (`error`).
    virtual bool finish(int dw, int dh, uint8_t* dst, FileStatus& file, std::wstring& error) = 0;
    virtual void abort() = 0;  // drops the current image

    // Hybrid path (integrated GPUs, GpuBackend::hybrid): the CPU decodes the composite into
    // decode_target() - memory the GPU reads, so nothing is copied - and resize_decoded() resizes
    // it on the GPU into a GpuImage the encoder reads where the GPU wrote it.
    virtual RowSink* decode_target(const PsdHeader&, std::wstring& error) {
        error = L"no hybrid path";
        return nullptr;
    }
    virtual std::unique_ptr<GpuImage> resize_decoded(int, int, std::wstring& error) {
        error = L"no hybrid path";
        return nullptr;
    }
};

// An initialised GPU (gpu_init): CUDA (gpu_cuda.cpp, x64 builds) or D3D12 compute (gpu_d3d12.cpp).
class GpuBackend {
public:
    virtual ~GpuBackend() = default;
    // Formats the kernels take (RLE 8/16-bit, raw 8-bit); anything else goes the CPU way.
    virtual bool supports(const PsdHeader& h) const = 0;
    // GPU memory a streamer needs for this image (upper bound, from the header). On an integrated
    // GPU (UMA) this is system memory.
    virtual uint64_t device_bytes(const PsdHeader& h, int dw, int dh) const = 0;
    virtual std::unique_ptr<GpuStreamer> make_streamer(std::wstring& error) = 0;
    // Memory the GPU may still use (D3D12: the OS budget), and its total.
    virtual bool memory(uint64_t& freeBytes, uint64_t& totalBytes) = 0;
    virtual std::wstring description() const = 0;  // for the batch's backend string
    virtual void trim() {}                          // returns cached memory after a batch
    // PP_GPU_PROFILE=1: GPU time per image since the last call ("" when not profiling).
    virtual std::wstring profile() { return {}; }
    // Measured GPU time per source megapixel (kernels only, recent images weighted most), for the
    // full GPU path or the hybrid one (resize only); 0 = not measured: treat the GPU as fast. The
    // engine sends an image the CPU way when the GPU's backlog would keep it longer than the CPU
    // takes.
    virtual double seconds_per_mpx(bool /*hybrid*/) const { return 0; }
    virtual bool integrated() const { return false; }  // shares the CPU's memory (UMA)
    virtual bool hybrid() const { return false; }      // streamers do decode_target / resize_decoded
};

// backend: PP_GpuBackend; adapter: CUDA device ordinal, or D3D12 adapter index in DXGI's
// high-performance order (software adapters left out). Replaces any earlier backend.
int gpu_init(int backend, int adapter, PP_DeviceInfo* info);
std::shared_ptr<GpuBackend> gpu_backend();  // null until gpu_init succeeds
void gpu_shutdown();

// Backend factories: create, load the kernels and run a self-test. On failure the code is
// returned and info.message says why.
#if PP_HAVE_CUDA
int cuda_create(int device, PP_DeviceInfo& info, std::shared_ptr<GpuBackend>& out);
#endif
int d3d12_create(int adapter, bool warp, PP_DeviceInfo& info, std::shared_ptr<GpuBackend>& out);

}  // namespace pp
