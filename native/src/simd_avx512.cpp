// AVX-512 (F + BW) kernels. Compiled with /arch:AVX512; only called after CPUID + XGETBV
// confirm AVX-512F/BW and OS support for ZMM state.
#include <immintrin.h>

#include "simd_kernels.h"

namespace pp::simd::avx512 {
namespace {

// 64 pixels. After the in-lane unpacks, register pJ holds pixels 16k+4J..16k+4J+3 in
// 128-bit lane k; a 4x4 transpose of 128-bit lanes (two rounds of shuffle_i64x2)
// puts them back in order.
inline void interleave4(uint8_t* dst, __m512i a, __m512i b, __m512i c, __m512i d) {
    const __m512i ab0 = _mm512_unpacklo_epi8(a, b), ab1 = _mm512_unpackhi_epi8(a, b);
    const __m512i cd0 = _mm512_unpacklo_epi8(c, d), cd1 = _mm512_unpackhi_epi8(c, d);
    const __m512i p0 = _mm512_unpacklo_epi16(ab0, cd0);
    const __m512i p1 = _mm512_unpackhi_epi16(ab0, cd0);
    const __m512i p2 = _mm512_unpacklo_epi16(ab1, cd1);
    const __m512i p3 = _mm512_unpackhi_epi16(ab1, cd1);
    const __m512i t0 = _mm512_shuffle_i64x2(p0, p1, 0x44);  // p0.L0 p0.L1 p1.L0 p1.L1
    const __m512i t1 = _mm512_shuffle_i64x2(p0, p1, 0xEE);  // p0.L2 p0.L3 p1.L2 p1.L3
    const __m512i t2 = _mm512_shuffle_i64x2(p2, p3, 0x44);  // p2.L0 p2.L1 p3.L0 p3.L1
    const __m512i t3 = _mm512_shuffle_i64x2(p2, p3, 0xEE);  // p2.L2 p2.L3 p3.L2 p3.L3
    _mm512_storeu_si512(dst + 0, _mm512_shuffle_i64x2(t0, t2, 0x88));    // px  0-15
    _mm512_storeu_si512(dst + 64, _mm512_shuffle_i64x2(t0, t2, 0xDD));   // px 16-31
    _mm512_storeu_si512(dst + 128, _mm512_shuffle_i64x2(t1, t3, 0x88));  // px 32-47
    _mm512_storeu_si512(dst + 192, _mm512_shuffle_i64x2(t1, t3, 0xDD));  // px 48-63
}

inline __m512i load(const uint8_t* p) { return _mm512_loadu_si512(p); }

}  // namespace

void narrow16(const uint8_t* src, uint8_t* dst, size_t n) {
    size_t i = 0;
    for (; i + 64 <= n; i += 64) {  // both loads precede the stores: dst == src is safe
        const __m512i a = load(src + 2 * i), b = load(src + 2 * i + 64);
        // vpmovwb keeps the low byte of each 16-bit lane = the high byte of the big-endian sample
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm512_cvtepi16_epi8(a));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i + 32), _mm512_cvtepi16_epi8(b));
    }
    scalar::narrow16(src + 2 * i, dst + i, n - i);
}

void merge_rgb(const uint8_t* r, const uint8_t* g, const uint8_t* b, uint8_t* dst, size_t n) {
    const __m512i alpha = _mm512_set1_epi8(-1);
    size_t i = 0;
    for (; i + 64 <= n; i += 64) interleave4(dst + 4 * i, load(r + i), load(g + i), load(b + i), alpha);
    scalar::merge_rgb(r + i, g + i, b + i, dst + 4 * i, n - i);
}

void merge_gray(const uint8_t* g, uint8_t* dst, size_t n) {
    const __m512i alpha = _mm512_set1_epi8(-1);
    size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        const __m512i v = load(g + i);
        interleave4(dst + 4 * i, v, v, v, alpha);
    }
    scalar::merge_gray(g + i, dst + 4 * i, n - i);
}

void merge_4(const uint8_t* c0, const uint8_t* c1, const uint8_t* c2, const uint8_t* c3, uint8_t* dst, size_t n) {
    size_t i = 0;
    for (; i + 64 <= n; i += 64) interleave4(dst + 4 * i, load(c0 + i), load(c1 + i), load(c2 + i), load(c3 + i));
    scalar::merge_4(c0 + i, c1 + i, c2 + i, c3 + i, dst + 4 * i, n - i);
}

}  // namespace pp::simd::avx512
