#include "simd.h"

#if !defined(_M_ARM64)
#include <intrin.h>
#endif

#include <cstdlib>
#include <cstring>

#include "simd_kernels.h"

namespace pp::simd {

// ---- scalar reference / tails ---------------------------------------------------------------
namespace scalar {
void narrow16(const uint8_t* src, uint8_t* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = src[2 * i];
}
void merge_rgb(const uint8_t* r, const uint8_t* g, const uint8_t* b, uint8_t* dst, size_t n) {
    for (size_t i = 0; i < n; ++i, dst += 4) {
        dst[0] = r[i];
        dst[1] = g[i];
        dst[2] = b[i];
        dst[3] = 0xFF;
    }
}
void merge_gray(const uint8_t* g, uint8_t* dst, size_t n) {
    for (size_t i = 0; i < n; ++i, dst += 4) {
        dst[0] = dst[1] = dst[2] = g[i];
        dst[3] = 0xFF;
    }
}
void merge_4(const uint8_t* c0, const uint8_t* c1, const uint8_t* c2, const uint8_t* c3, uint8_t* dst, size_t n) {
    for (size_t i = 0; i < n; ++i, dst += 4) {
        dst[0] = c0[i];
        dst[1] = c1[i];
        dst[2] = c2[i];
        dst[3] = c3[i];
    }
}
}  // namespace scalar

// ---- dispatch -------------------------------------------------------------------------------
namespace {

#if defined(_M_ARM64)

const Kernels kNeon = {neon::narrow16, neon::merge_rgb, neon::merge_gray, neon::merge_4, Level::Neon, L"NEON"};

Level detect() { return Level::Neon; }
Level initial_level() { return Level::Neon; }
const Kernels& table(Level) { return kNeon; }

#else

const Kernels kTable[] = {
    {sse2::narrow16, sse2::merge_rgb, sse2::merge_gray, sse2::merge_4, Level::Sse2, L"SSE2"},
    {avx2::narrow16, avx2::merge_rgb, avx2::merge_gray, avx2::merge_4, Level::Avx2, L"AVX2"},
    {avx512::narrow16, avx512::merge_rgb, avx512::merge_gray, avx512::merge_4, Level::Avx512, L"AVX-512"},
};

Level detect() {
    int r[4];
    __cpuid(r, 0);
    const int maxLeaf = r[0];
    __cpuid(r, 1);
    const bool osxsave = (r[2] & (1 << 27)) != 0, avx = (r[2] & (1 << 28)) != 0;
    if (!osxsave || !avx || maxLeaf < 7) return Level::Sse2;
    const unsigned long long xcr0 = _xgetbv(0);
    const bool ymmState = (xcr0 & 0x6) == 0x6;     // XMM + YMM
    const bool zmmState = (xcr0 & 0xE6) == 0xE6;   // + opmask, ZMM_Hi256, Hi16_ZMM
    __cpuidex(r, 7, 0);
    const bool avx2 = (r[1] & (1 << 5)) != 0;
    const bool avx512f = (r[1] & (1 << 16)) != 0, avx512bw = (r[1] & (1 << 30)) != 0;
    if (avx512f && avx512bw && zmmState) return Level::Avx512;
    if (avx2 && ymmState) return Level::Avx2;
    return Level::Sse2;  // AVX1-only CPUs lack 256-bit integer ops: SSE2 kernels
}

const Kernels& table(Level level) { return kTable[int(level)]; }

Level initial_level() {
    Level level = detected_level();
    char buf[16] = {};
    size_t len = 0;
    if (getenv_s(&len, buf, sizeof buf, "PP_SIMD") == 0 && len > 0) {
        Level want = level;
        if (!_stricmp(buf, "sse2") || !_stricmp(buf, "sse") || !_stricmp(buf, "avx")) want = Level::Sse2;
        else if (!_stricmp(buf, "avx2")) want = Level::Avx2;
        else if (!_stricmp(buf, "avx512")) want = Level::Avx512;
        if (int(want) < int(level)) level = want;
    }
    return level;
}

#endif

const Kernels* g_active = nullptr;

}  // namespace

Level detected_level() {
    static const Level level = detect();
    return level;
}

const Kernels& kernels_for(Level level) {
    if (int(level) > int(detected_level())) level = detected_level();
    return table(level);
}

const Kernels& kernels() {
    static const Kernels* initial = &kernels_for(initial_level());
    return g_active ? *g_active : *initial;
}

void force_level(Level level) { g_active = &kernels_for(level); }

}  // namespace pp::simd
