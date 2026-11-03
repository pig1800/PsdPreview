// AVX2 kernels. Compiled with /arch:AVX2; only called after CPUID + XGETBV checks.
#include <immintrin.h>

#include "simd_kernels.h"

namespace pp::simd::avx2 {
namespace {

// 32 pixels. Unpacks work inside 128-bit lanes, so after them register pJ holds pixels
// 4J..4J+3 in its low lane and 16+4J.. in its high lane; permute2x128 restores order.
inline void interleave4(uint8_t* dst, __m256i a, __m256i b, __m256i c, __m256i d) {
    const __m256i ab0 = _mm256_unpacklo_epi8(a, b), ab1 = _mm256_unpackhi_epi8(a, b);
    const __m256i cd0 = _mm256_unpacklo_epi8(c, d), cd1 = _mm256_unpackhi_epi8(c, d);
    const __m256i p0 = _mm256_unpacklo_epi16(ab0, cd0);  // px  0-3  | 16-19
    const __m256i p1 = _mm256_unpackhi_epi16(ab0, cd0);  // px  4-7  | 20-23
    const __m256i p2 = _mm256_unpacklo_epi16(ab1, cd1);  // px  8-11 | 24-27
    const __m256i p3 = _mm256_unpackhi_epi16(ab1, cd1);  // px 12-15 | 28-31
    auto* out = reinterpret_cast<__m256i*>(dst);
    _mm256_storeu_si256(out + 0, _mm256_permute2x128_si256(p0, p1, 0x20));  // px  0-7
    _mm256_storeu_si256(out + 1, _mm256_permute2x128_si256(p2, p3, 0x20));  // px  8-15
    _mm256_storeu_si256(out + 2, _mm256_permute2x128_si256(p0, p1, 0x31));  // px 16-23
    _mm256_storeu_si256(out + 3, _mm256_permute2x128_si256(p2, p3, 0x31));  // px 24-31
}

inline __m256i load(const uint8_t* p) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)); }

}  // namespace

void narrow16(const uint8_t* src, uint8_t* dst, size_t n) {
    const __m256i lowByte = _mm256_set1_epi16(0x00FF);
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {  // loads precede the store: dst == src is safe
        const __m256i a = _mm256_and_si256(load(src + 2 * i), lowByte);
        const __m256i b = _mm256_and_si256(load(src + 2 * i + 32), lowByte);
        // packus is per-lane: [a0-7 b0-7 | a8-15 b8-15] -> reorder 64-bit blocks to a, b
        const __m256i packed = _mm256_permute4x64_epi64(_mm256_packus_epi16(a, b), 0xD8);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), packed);
    }
    scalar::narrow16(src + 2 * i, dst + i, n - i);
}

void merge_rgb(const uint8_t* r, const uint8_t* g, const uint8_t* b, uint8_t* dst, size_t n) {
    const __m256i alpha = _mm256_set1_epi8(-1);
    size_t i = 0;
    for (; i + 32 <= n; i += 32) interleave4(dst + 4 * i, load(r + i), load(g + i), load(b + i), alpha);
    scalar::merge_rgb(r + i, g + i, b + i, dst + 4 * i, n - i);
}

void merge_gray(const uint8_t* g, uint8_t* dst, size_t n) {
    const __m256i alpha = _mm256_set1_epi8(-1);
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        const __m256i v = load(g + i);
        interleave4(dst + 4 * i, v, v, v, alpha);
    }
    scalar::merge_gray(g + i, dst + 4 * i, n - i);
}

void merge_4(const uint8_t* c0, const uint8_t* c1, const uint8_t* c2, const uint8_t* c3, uint8_t* dst, size_t n) {
    size_t i = 0;
    for (; i + 32 <= n; i += 32) interleave4(dst + 4 * i, load(c0 + i), load(c1 + i), load(c2 + i), load(c3 + i));
    scalar::merge_4(c0 + i, c1 + i, c2 + i, c3 + i, dst + 4 * i, n - i);
}

}  // namespace pp::simd::avx2
