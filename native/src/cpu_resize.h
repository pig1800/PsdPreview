#pragma once
#include "common.h"

namespace pp {

struct CpuBackend {
    bool avx512Supported = false;
    bool avx512Active = false;
    bool forcedDispatch = false;
    bool stb = false;                  // stb_image_resize2 instead of IPP
    std::wstring target, name, version;  // resizer: IPP dispatch target ("k0", "l9"...) or SIMD set
};

// Picks and initialises the resizer once (thread-safe); must run before any resize:
//   x64    Intel IPP, making sure its AVX-512 code path is used when the CPU/OS support it.
//          PP_CPU_RESIZE=stb selects stb_image_resize2 instead (testing the ARM64 resizer).
//   ARM64  stb_image_resize2 (Neon); IPP is x64-only.
const CpuBackend& cpu_backend();

// Cubic resize of 4-channel 8-bit pixels with the filter of resize_filter.h (antialiased when
// shrinking), same as the GPU kernels. Both resizers give the same result (within 1 level).
bool cpu_resize(const uint8_t* src, int w, int h, uint8_t* dst, int dw, int dh, std::wstring& error);

}  // namespace pp
