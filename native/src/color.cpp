#include "color.h"

#include <cwctype>

#include <lcms2.h>

namespace pp {

struct ColorTransform {
    cmsHTRANSFORM handle = nullptr;
};

struct ColorCache::Entry {
    int colorMode = 0;
    std::vector<uint8_t> icc;
    std::wstring description;
    ColorTransform xf;
};

namespace {

// Input layouts of our 4-byte pixels.
cmsUInt32Number input_format(int colorMode) {
    switch (colorMode) {
        case 1: return COLORSPACE_SH(PT_GRAY) | CHANNELS_SH(1) | EXTRA_SH(3) | BYTES_SH(1);  // G,G,G,A: use the first
        case 4: return TYPE_CMYK_8_REV;  // PSD stores CMYK inverted (255 = no ink)
        default: return TYPE_RGBA_8;
    }
}

cmsColorSpaceSignature expected_space(int colorMode) {
    return colorMode == 1 ? cmsSigGrayData : colorMode == 4 ? cmsSigCmykData : cmsSigRgbData;
}

bool contains_ci(const std::wstring& s, const wchar_t* what) {
    std::wstring a = s, b = what;
    for (auto& c : a) c = wchar_t(std::towlower(c));
    for (auto& c : b) c = wchar_t(std::towlower(c));
    return a.find(b) != std::wstring::npos;
}

std::wstring describe(cmsHPROFILE p) {
    wchar_t buf[256] = {};
    cmsGetProfileInfo(p, cmsInfoDescription, "en", "US", buf, sizeof buf / sizeof buf[0]);
    std::wstring s(buf);
    return s.empty() ? L"(unnamed ICC profile)" : s;
}

}  // namespace

ColorCache::ColorCache() = default;

ColorCache::~ColorCache() {
    for (auto& e : entries_)
        if (e->xf.handle) cmsDeleteTransform(e->xf.handle);
}

const ColorTransform* ColorCache::get(const std::vector<uint8_t>& icc, int colorMode, std::wstring& description) {
    std::lock_guard lk(m_);
    for (auto& e : entries_) {
        if (e->colorMode == colorMode && e->icc == icc) {
            description = e->description;
            return e->xf.handle ? &e->xf : nullptr;
        }
    }
    auto e = std::make_unique<Entry>();
    e->colorMode = colorMode;
    e->icc = icc;
    if (cmsHPROFILE in = cmsOpenProfileFromMem(icc.data(), cmsUInt32Number(icc.size()))) {
        e->description = describe(in);
        if (cmsGetColorSpace(in) != expected_space(colorMode)) {
            e->description += L" (doesn't match the document colour mode; ignored)";
        } else if (colorMode == 4 || !contains_ci(e->description, L"sRGB")) {
            // Photoshop's default conversion: relative colorimetric + black point compensation.
            cmsHPROFILE srgb = cmsCreate_sRGBProfile();
            e->xf.handle = cmsCreateTransform(in, input_format(colorMode), srgb, TYPE_RGBA_8,
                                              INTENT_RELATIVE_COLORIMETRIC, cmsFLAGS_BLACKPOINTCOMPENSATION);
            cmsCloseProfile(srgb);
            if (!e->xf.handle) e->description += L" (lcms2 couldn't build a transform; ignored)";
        }
        cmsCloseProfile(in);
    } else {
        e->description = L"(unreadable ICC profile; ignored)";
    }
    description = e->description;
    const ColorTransform* result = e->xf.handle ? &e->xf : nullptr;
    entries_.push_back(std::move(e));
    return result;
}

void to_srgb(const ColorTransform* xf, int colorMode, uint8_t* px, size_t count) {
    if (xf) {
        // lcms2 doesn't document in-place transforms: go through a small scratch block.
        // It never writes the output alpha byte (no cmsFLAGS_COPY_ALPHA), so pre-filling
        // the scratch with 0xFF yields A = 255.
        constexpr size_t kChunk = 4096;
        alignas(64) uint8_t scratch[kChunk * kPixelBytes];
        std::memset(scratch, 0xFF, sizeof scratch);
        for (size_t i = 0; i < count; i += kChunk) {
            const size_t n = std::min(kChunk, count - i);
            cmsDoTransform(xf->handle, px + i * kPixelBytes, scratch, cmsUInt32Number(n));
            std::memcpy(px + i * kPixelBytes, scratch, n * kPixelBytes);
        }
        return;
    }
    if (colorMode == 4) {  // naive CMYK -> RGB; stored inverted, so R = C' * K' / 255
        for (size_t i = 0; i < count; ++i, px += kPixelBytes) {
            const uint32_t k = px[3];
            px[0] = uint8_t((px[0] * k + 127) / 255);
            px[1] = uint8_t((px[1] * k + 127) / 255);
            px[2] = uint8_t((px[2] * k + 127) / 255);
            px[3] = 0xFF;
        }
    }
}

}  // namespace pp
