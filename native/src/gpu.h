#pragma once
#include <cuda_runtime_api.h>

#include "common.h"
#include "psd.h"

namespace pp {

int gpu_init(int device, PP_DeviceInfo* info);
bool gpu_ready();
void gpu_shutdown();
bool gpu_memory(uint64_t& freeBytes, uint64_t& totalBytes);
std::wstring gpu_description();  // device name
void gpu_trim();                 // returns pooled device memory after a batch

// Device row pitch used for a packed row of `rowBytes`.
inline uint64_t gpu_pitch(uint64_t rowBytes) { return (rowBytes + 255) & ~uint64_t(255); }

// Pinned bytes per band buffer (each streamer has two).
constexpr size_t kGpuBandBytes = 4u << 20;

// One per extract thread: its own CUDA stream and two pinned band buffers. The composite is
// decoded on the GPU (see gpu_kernels.cu):
//   push    upload() reads the colour channels' data straight from the file into the current
//           pinned band, still PackBits-compressed (~1.35 bytes/px instead of 4 decoded, and no
//           CPU work beyond the read); a full band is uploaded asynchronously while the next one
//           is read into the other
//   process finish() unpacks every row (one GPU thread per row) into 8-bit planes and resizes
//   pull    the result is downloaded into the caller's buffer
// Device memory comes from a stream-ordered pool, so concurrent images on other threads'
// streams overlap freely and nothing is reallocated per image.
class GpuStreamer final {
public:
    GpuStreamer() = default;
    GpuStreamer(const GpuStreamer&) = delete;
    GpuStreamer& operator=(const GpuStreamer&) = delete;
    ~GpuStreamer();

    // RLE 8/16-bit and raw 8-bit composites; anything else goes the CPU path.
    static bool supports(const PsdHeader& h);
    // Device memory needed for one image (upper bound, from the header).
    static uint64_t device_bytes(const PsdHeader& h, int dw, int dh);

    bool init(std::wstring& error);  // once per thread
    // Reads and uploads the composite. A file problem (truncated, blank...) is returned in
    // `file`; false with `file` ok means a GPU problem (`error`): redo the image on the CPU.
    bool upload(const FileReader& f, const PsdHeader& h, FileStatus& file, std::wstring& error);
    // Unpacks, resizes and downloads into dst (dw x dh x 4 bytes). Corrupt PackBits data is a
    // file problem (`file`); false with `file` ok is a GPU problem (`error`).
    bool finish(int dw, int dh, uint8_t* dst, FileStatus& file, std::wstring& error);
    void abort();  // drops the current image

private:
    cudaError_t send(uint8_t* devDst, size_t n);  // uploads band_[cur_], switches bands

    cudaStream_t stream_ = nullptr;
    cudaEvent_t uploaded_[2] = {};  // per band; blocking-sync events: waiting threads sleep
    uint8_t* band_[2] = {};
    uint32_t* status_ = nullptr;  // pinned: first corrupt row reported by the unpack kernel
    int cur_ = 0;

    // the current image
    uint8_t *dComp_ = nullptr, *dPlanes_ = nullptr;
    uint32_t* dOffsets_ = nullptr;  // rows + 1 offsets into dComp_
    uint32_t* dBad_ = nullptr;
    uint32_t w_ = 0, h_ = 0;
    int planes_ = 0, depth_ = 8, compression_ = 0;
};

}  // namespace pp
