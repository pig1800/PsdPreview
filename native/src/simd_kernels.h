// Kernel entry points, one namespace per instruction set. Each set lives in its own
// translation unit compiled with the matching /arch flag (x64), or simd_neon.cpp (ARM64). This header only declares
// functions (no inline code), so nothing compiled for AVX2/AVX-512 can be picked up by
// baseline code through inline-function merging.
#pragma once
#include <cstddef>
#include <cstdint>

#define PP_DECLARE_SIMD_KERNELS(ns)                                                                          \
    namespace ns {                                                                                           \
    /* big-endian 16-bit samples -> 8-bit by truncation (keep the high byte). dst may equal src. */          \
    void narrow16(const uint8_t* src, uint8_t* dst, size_t n);                                               \
    /* planar R,G,B -> RGBA8888 with A = 255 */                                                              \
    void merge_rgb(const uint8_t* r, const uint8_t* g, const uint8_t* b, uint8_t* dst, size_t n);           \
    /* planar gray -> RGBA8888 (G,G,G,255) */                                                                \
    void merge_gray(const uint8_t* g, uint8_t* dst, size_t n);                                               \
    /* four planes -> interleaved 4-byte pixels (used for CMYK) */                                           \
    void merge_4(const uint8_t* c0, const uint8_t* c1, const uint8_t* c2, const uint8_t* c3, uint8_t* dst, \
                 size_t n);                                                                                  \
    }

namespace pp::simd {
PP_DECLARE_SIMD_KERNELS(scalar)
PP_DECLARE_SIMD_KERNELS(sse2)
PP_DECLARE_SIMD_KERNELS(avx2)
PP_DECLARE_SIMD_KERNELS(avx512)
PP_DECLARE_SIMD_KERNELS(neon)
}  // namespace pp::simd

#undef PP_DECLARE_SIMD_KERNELS
