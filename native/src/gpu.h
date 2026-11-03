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

// One per extract thread: its own CUDA stream and two pinned band buffers.
//   push    the decoder writes rows straight into the current pinned band (row()); a full
//           band is uploaded asynchronously while the next one is decoded into the other
//   process as soon as the last band is queued, the resize kernel (gpu_resize.cu) runs on the same stream
//   pull    the result is downloaded into the caller's buffer
// Device memory comes from a stream-ordered pool, so concurrent images on other threads'
// streams overlap freely and nothing is reallocated per image.
class GpuStreamer final : public RowSink {
public:
    GpuStreamer() = default;
    GpuStreamer(const GpuStreamer&) = delete;
    GpuStreamer& operator=(const GpuStreamer&) = delete;
    ~GpuStreamer();

    bool init(std::wstring& error);                              // once per thread
    bool begin(uint32_t w, uint32_t h, std::wstring& error);     // allocates the device image
    uint8_t* row(uint32_t y) override;                           // decoder output, rows in order
    bool finish(int dw, int dh, uint8_t* dst, std::wstring& error);  // tail upload, resize, download
    void abort();                                                // after a decode error

private:
    void flush(uint32_t yEnd);

    cudaStream_t stream_ = nullptr;
    cudaEvent_t uploaded_[2] = {};
    uint8_t* band_[2] = {};
    uint8_t* dSrc_ = nullptr;
    size_t pitch_ = 0, rowBytes_ = 0;
    uint32_t w_ = 0, h_ = 0, bandRows_ = 0, bandY0_ = 0;
    int cur_ = 0;
    cudaError_t err_ = cudaSuccess;
};

}  // namespace pp
