#include "cpu_resize.h"

#if PP_HAVE_IPP
#include <ipp.h>
#endif

#include "resize_filter.h"

// stb_image_resize2: one translation unit holds the implementation. Its SIMD set is chosen at
// compile time: Neon on ARM64, SSE2 on x64 (only used there for testing).
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#pragma warning(push, 1)
#include "stb_image_resize2.h"
#pragma warning(pop)

namespace pp {
namespace {

// ---- stb_image_resize2 --------------------------------------------------------------------

// The Mitchell-Netravali kernel of resize_filter.h. stb widens it by the reduction factor itself
// (its x is in output pixels when shrinking) and normalises the weights.
stbir_real mitchell(stbir_real x, stbir_real, void* user) {
    const CubicFilter& f = *static_cast<const CubicFilter*>(user);
    const double B = f.B, C = f.C;
    x = std::fabs(x);
    if (x < 1) return stbir_real(((12 - 9 * B - 6 * C) * x + (-18 + 12 * B + 6 * C)) * x * x + (6 - 2 * B));
    if (x < 2) return stbir_real((((-B - 6 * C) * x + (6 * B + 30 * C)) * x + (-12 * B - 48 * C)) * x + (8 * B + 24 * C));
    return 0;
}
stbir_real cubic_support(stbir_real, void*) { return 2; }

bool stb_resize(const uint8_t* src, int w, int h, uint8_t* dst, int dw, int dh, std::wstring& error) {
    CubicFilter f = resize_filter(w, dw);
    STBIR_RESIZE r;
    stbir_resize_init(&r, src, w, h, int64_t(w) * kPixelBytes, dst, dw, dh, int64_t(dw) * kPixelBytes, STBIR_4CHANNEL,
                      STBIR_TYPE_UINT8);  // 4 independent channels, no alpha weighting, no sRGB
    stbir_set_edgemodes(&r, STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP);
    stbir_set_user_data(&r, &f);
    if (dw < w) {
        stbir_set_filter_callbacks_new(&r, mitchell, cubic_support, nullptr, mitchell, cubic_support, nullptr);
    } else {
        stbir_set_filters(&r, STBIR_FILTER_CATMULLROM, STBIR_FILTER_CATMULLROM);  // = B 0, C 0.5
    }
    if (!stbir_resize_extended(&r)) {
        error = L"stb_image_resize2 failed (out of memory?)";
        return false;
    }
    return true;
}

// ---- Intel IPP ----------------------------------------------------------------------------
#if PP_HAVE_IPP

// IPP's feature set for its AVX-512 ("k0") code path, see ippSetCpuFeatures docs.
constexpr Ipp64u kAvx512Features = ippCPUID_AVX512F | ippCPUID_AVX512CD | ippCPUID_AVX512VL | ippCPUID_AVX512BW |
                                   ippCPUID_AVX512DQ | ippAVX512_ENABLEDBYOS;

bool is_avx512_target(const char* t) { return t[0] == 'k' || t[0] == 'd'; }  // k0 = AVX-512, d1 = AVX10.2

void init_ipp(CpuBackend& b) {
    // ippInit() returns ippStsNonIntelCpu on AMD but still dispatches by features;
    // older IPP builds (and some CPU/OS combinations) fall back to l9 (AVX2), so
    // verify and force the AVX-512 path explicitly when the hardware has it.
    ippInit();
    Ipp64u features = 0;
    ippGetCpuFeatures(&features, nullptr);
    b.avx512Supported = (features & kAvx512Features) == kAvx512Features;

    const IppLibraryVersion* v = ippiGetLibVersion();
    if (b.avx512Supported && !is_avx512_target(v->targetCpu)) {
        if (ippSetCpuFeatures(features) >= ippStsNoErr) b.forcedDispatch = true;
        v = ippiGetLibVersion();
    }
    b.avx512Active = is_avx512_target(v->targetCpu);
    b.target = widen(std::string(v->targetCpu, strnlen(v->targetCpu, 4)).c_str());
    b.name = widen(v->Name);
    b.version = widen(v->Version);
}

bool ipp_resize(const uint8_t* src, int w, int h, uint8_t* dst, int dw, int dh, std::wstring& error) {
    const IppiSize srcSize{w, h}, dstSize{dw, dh};
    const CubicFilter f = resize_filter(w, dw);
    const bool shrink = dw < w;  // antialiasing widens the kernel by the reduction factor
    int specSize = 0, initSize = 0, bufSize = 0;
    IppStatus st = ippiResizeGetSize_8u(srcSize, dstSize, ippCubic, shrink ? 1 : 0, &specSize, &initSize);
    if (st < ippStsNoErr) { error = L"IPP: " + widen(ippGetStatusString(st)); return false; }

    struct IppFree { void operator()(Ipp8u* p) const { ippsFree(p); } };
    std::unique_ptr<Ipp8u, IppFree> spec(ippsMalloc_8u(specSize)), init(ippsMalloc_8u(initSize));
    if (!spec || !init) { error = L"IPP out of memory"; return false; }
    auto* pSpec = reinterpret_cast<IppiResizeSpec_32f*>(spec.get());
    st = shrink ? ippiResizeAntialiasingCubicInit(srcSize, dstSize, f.B, f.C, pSpec, init.get())
                : ippiResizeCubicInit_8u(srcSize, dstSize, f.B, f.C, pSpec, init.get());
    if (st < ippStsNoErr) { error = L"IPP: " + widen(ippGetStatusString(st)); return false; }
    init.reset();

    st = ippiResizeGetBufferSize_8u(pSpec, dstSize, kPixelBytes, &bufSize);
    if (st < ippStsNoErr) { error = L"IPP: " + widen(ippGetStatusString(st)); return false; }
    std::unique_ptr<Ipp8u, IppFree> buffer(ippsMalloc_8u(bufSize));
    if (!buffer) { error = L"IPP out of memory"; return false; }

    const IppiPoint origin{0, 0};
    st = shrink ? ippiResizeAntialiasing_8u_C4R(src, w * kPixelBytes, dst, dw * kPixelBytes, origin, dstSize, ippBorderRepl,
                                                nullptr, pSpec, buffer.get())
                : ippiResizeCubic_8u_C4R(src, w * kPixelBytes, dst, dw * kPixelBytes, origin, dstSize, ippBorderRepl, nullptr,
                                         pSpec, buffer.get());
    if (st < ippStsNoErr) { error = L"IPP: " + widen(ippGetStatusString(st)); return false; }
    return true;
}

#endif  // PP_HAVE_IPP

CpuBackend init_backend() {
    CpuBackend b;
#if PP_HAVE_IPP
    char want[8] = {};
    size_t len = 0;
    b.stb = getenv_s(&len, want, sizeof want, "PP_CPU_RESIZE") == 0 && len > 0 && !_stricmp(want, "stb");
    if (!b.stb) {
        init_ipp(b);
        return b;
    }
#endif
    b.stb = true;
    b.name = L"stb_image_resize2";
    b.version = L"2.50";
#if defined(_M_ARM64)
    b.target = L"NEON";
#else
    b.target = L"SSE2";
#endif
    return b;
}

}  // namespace

const CpuBackend& cpu_backend() {
    static const CpuBackend backend = init_backend();  // thread-safe static init
    return backend;
}

bool cpu_resize(const uint8_t* src, int w, int h, uint8_t* dst, int dw, int dh, std::wstring& error) {
#if PP_HAVE_IPP
    if (!cpu_backend().stb) return ipp_resize(src, w, h, dst, dw, dh, error);
#endif
    return stb_resize(src, w, h, dst, dw, dh, error);
}

}  // namespace pp
