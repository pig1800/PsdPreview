#pragma once
#include "common.h"

namespace pp {

struct CpuBackend {
    bool avx512Supported = false;
    bool avx512Active = false;
    bool forcedDispatch = false;
    std::wstring target, name, version;
};

// Initialises IPP once (thread-safe) and makes sure the AVX-512 code path is
// used when the CPU/OS support it. Must run before any other IPP call.
const CpuBackend& cpu_backend();

// Cubic resize of 4-channel 8-bit pixels with the filter of resize_filter.h (antialiased when
// shrinking), same as the GPU kernel.
bool cpu_resize(const uint8_t* src, int w, int h, uint8_t* dst, int dw, int dh, std::wstring& error);

}  // namespace pp
