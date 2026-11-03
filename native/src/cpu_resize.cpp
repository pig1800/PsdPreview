#include "cpu_resize.h"

#include <ipp.h>

namespace pp {
namespace {

// IPP's feature set for its AVX-512 ("k0") code path, see ippSetCpuFeatures docs.
constexpr Ipp64u kAvx512Features = ippCPUID_AVX512F | ippCPUID_AVX512CD | ippCPUID_AVX512VL | ippCPUID_AVX512BW |
                                   ippCPUID_AVX512DQ | ippAVX512_ENABLEDBYOS;

bool is_avx512_target(const char* t) { return t[0] == 'k' || t[0] == 'd'; }  // k0 = AVX-512, d1 = AVX10.2

CpuBackend init_backend() {
    CpuBackend b;
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
    return b;
}

}  // namespace

const CpuBackend& cpu_backend() {
    static const CpuBackend backend = init_backend();  // thread-safe static init
    return backend;
}

bool cpu_resize(const uint8_t* src, int w, int h, uint8_t* dst, int dw, int dh, std::wstring& error) {
    cpu_backend();
    const IppiSize srcSize{w, h}, dstSize{dw, dh};
    int specSize = 0, initSize = 0, bufSize = 0;
    IppStatus st = ippiResizeGetSize_8u(srcSize, dstSize, ippCubic, 0, &specSize, &initSize);
    if (st < ippStsNoErr) { error = widen(ippGetStatusString(st)); return false; }

    struct IppFree { void operator()(Ipp8u* p) const { ippsFree(p); } };
    std::unique_ptr<Ipp8u, IppFree> spec(ippsMalloc_8u(specSize)), init(ippsMalloc_8u(initSize));
    if (!spec || !init) { error = L"IPP out of memory"; return false; }
    auto* pSpec = reinterpret_cast<IppiResizeSpec_32f*>(spec.get());
    st = ippiResizeCubicInit_8u(srcSize, dstSize, 0.0f, 0.5f, pSpec, init.get());
    if (st < ippStsNoErr) { error = widen(ippGetStatusString(st)); return false; }
    init.reset();

    st = ippiResizeGetBufferSize_8u(pSpec, dstSize, kPixelBytes, &bufSize);
    if (st < ippStsNoErr) { error = widen(ippGetStatusString(st)); return false; }
    std::unique_ptr<Ipp8u, IppFree> buffer(ippsMalloc_8u(bufSize));
    if (!buffer) { error = L"IPP out of memory"; return false; }

    const IppiPoint origin{0, 0};
    st = ippiResizeCubic_8u_C4R(src, w * kPixelBytes, dst, dw * kPixelBytes, origin, dstSize, ippBorderRepl, nullptr,
                                pSpec, buffer.get());
    if (st < ippStsNoErr) { error = widen(ippGetStatusString(st)); return false; }
    return true;
}

}  // namespace pp
