// Neon kernels (ARM64 baseline: every ARMv8 CPU has Neon). The interleaving loads/stores
// (vld2q/vst4q) do the whole job of the x64 unpack shuffles.
#include <arm_neon.h>

#include "simd_kernels.h"

namespace pp::simd::neon {

void narrow16(const uint8_t* src, uint8_t* dst, size_t n) {
    // vld2q splits even/odd bytes: val[0] holds the high byte of each big-endian sample. Each
    // store lands below the bytes later loads read, so dst == src is safe.
    size_t i = 0;
    for (; i + 16 <= n; i += 16) vst1q_u8(dst + i, vld2q_u8(src + 2 * i).val[0]);
    scalar::narrow16(src + 2 * i, dst + i, n - i);
}

void merge_rgb(const uint8_t* r, const uint8_t* g, const uint8_t* b, uint8_t* dst, size_t n) {
    uint8x16x4_t v;
    v.val[3] = vdupq_n_u8(0xFF);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        v.val[0] = vld1q_u8(r + i);
        v.val[1] = vld1q_u8(g + i);
        v.val[2] = vld1q_u8(b + i);
        vst4q_u8(dst + 4 * i, v);
    }
    scalar::merge_rgb(r + i, g + i, b + i, dst + 4 * i, n - i);
}

void merge_gray(const uint8_t* g, uint8_t* dst, size_t n) {
    uint8x16x4_t v;
    v.val[3] = vdupq_n_u8(0xFF);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        v.val[0] = v.val[1] = v.val[2] = vld1q_u8(g + i);
        vst4q_u8(dst + 4 * i, v);
    }
    scalar::merge_gray(g + i, dst + 4 * i, n - i);
}

void merge_4(const uint8_t* c0, const uint8_t* c1, const uint8_t* c2, const uint8_t* c3, uint8_t* dst, size_t n) {
    uint8x16x4_t v;
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        v.val[0] = vld1q_u8(c0 + i);
        v.val[1] = vld1q_u8(c1 + i);
        v.val[2] = vld1q_u8(c2 + i);
        v.val[3] = vld1q_u8(c3 + i);
        vst4q_u8(dst + 4 * i, v);
    }
    scalar::merge_4(c0 + i, c1 + i, c2 + i, c3 + i, dst + 4 * i, n - i);
}

}  // namespace pp::simd::neon
