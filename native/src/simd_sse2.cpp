// SSE2 kernels (x64 baseline). AVX1 has no 256-bit integer shuffles/unpacks, so CPUs
// with AVX but not AVX2 use these too.
#include <emmintrin.h>

#include "simd_kernels.h"

namespace pp::simd::sse2 {
namespace {

// 16 pixels: four byte planes -> interleaved (a0 b0 c0 d0 a1 b1 c1 d1 ...)
inline void interleave4(uint8_t* dst, __m128i a, __m128i b, __m128i c, __m128i d) {
    const __m128i ab0 = _mm_unpacklo_epi8(a, b), ab1 = _mm_unpackhi_epi8(a, b);
    const __m128i cd0 = _mm_unpacklo_epi8(c, d), cd1 = _mm_unpackhi_epi8(c, d);
    auto* out = reinterpret_cast<__m128i*>(dst);
    _mm_storeu_si128(out + 0, _mm_unpacklo_epi16(ab0, cd0));  // px  0-3
    _mm_storeu_si128(out + 1, _mm_unpackhi_epi16(ab0, cd0));  // px  4-7
    _mm_storeu_si128(out + 2, _mm_unpacklo_epi16(ab1, cd1));  // px  8-11
    _mm_storeu_si128(out + 3, _mm_unpackhi_epi16(ab1, cd1));  // px 12-15
}

inline __m128i load(const uint8_t* p) { return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p)); }

}  // namespace

void narrow16(const uint8_t* src, uint8_t* dst, size_t n) {
    // The high byte of a big-endian sample is the low byte of a little-endian 16-bit lane.
    const __m128i lowByte = _mm_set1_epi16(0x00FF);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {  // both loads happen before the store, so dst == src is safe
        const __m128i a = _mm_and_si128(load(src + 2 * i), lowByte);
        const __m128i b = _mm_and_si128(load(src + 2 * i + 16), lowByte);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), _mm_packus_epi16(a, b));
    }
    scalar::narrow16(src + 2 * i, dst + i, n - i);
}

void merge_rgb(const uint8_t* r, const uint8_t* g, const uint8_t* b, uint8_t* dst, size_t n) {
    const __m128i alpha = _mm_set1_epi8(-1);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) interleave4(dst + 4 * i, load(r + i), load(g + i), load(b + i), alpha);
    scalar::merge_rgb(r + i, g + i, b + i, dst + 4 * i, n - i);
}

void merge_gray(const uint8_t* g, uint8_t* dst, size_t n) {
    const __m128i alpha = _mm_set1_epi8(-1);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m128i v = load(g + i);
        interleave4(dst + 4 * i, v, v, v, alpha);
    }
    scalar::merge_gray(g + i, dst + 4 * i, n - i);
}

void merge_4(const uint8_t* c0, const uint8_t* c1, const uint8_t* c2, const uint8_t* c3, uint8_t* dst, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) interleave4(dst + 4 * i, load(c0 + i), load(c1 + i), load(c2 + i), load(c3 + i));
    scalar::merge_4(c0 + i, c1 + i, c2 + i, c3 + i, dst + 4 * i, n - i);
}

}  // namespace pp::simd::sse2
