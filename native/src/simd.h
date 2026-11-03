#pragma once
#include <cstddef>
#include <cstdint>

namespace pp::simd {

enum class Level : int { Sse2 = 0, Avx2 = 1, Avx512 = 2 };

struct Kernels {
    void (*narrow16)(const uint8_t* src, uint8_t* dst, size_t n);
    void (*merge_rgb)(const uint8_t* r, const uint8_t* g, const uint8_t* b, uint8_t* dst, size_t n);
    void (*merge_gray)(const uint8_t* g, uint8_t* dst, size_t n);
    void (*merge_4)(const uint8_t* c0, const uint8_t* c1, const uint8_t* c2, const uint8_t* c3, uint8_t* dst, size_t n);
    Level level;
    const wchar_t* name;
};

// Best level the CPU and OS support: AVX-512 (F+BW), else AVX2, else SSE2.
Level detected_level();

// Kernels in use: the detected level, optionally lowered with the environment variable
// PP_SIMD=sse2|avx2|avx512 (read once; cannot exceed what the hardware supports).
const Kernels& kernels();

// Kernels for a specific level (tests/benchmarks). Must not exceed detected_level().
const Kernels& kernels_for(Level level);

// Test hook: switch the active kernels. Not thread-safe; don't call while decoding.
void force_level(Level level);

}  // namespace pp::simd
