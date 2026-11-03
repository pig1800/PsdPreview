#pragma once
#include "common.h"

namespace pp {

struct ColorTransform;

// Per-batch cache of lcms2 transforms (embedded profile -> sRGB), keyed by the exact
// profile bytes, so a batch of files sharing a profile builds one transform.
class ColorCache {
public:
    ColorCache();
    ColorCache(const ColorCache&) = delete;
    ColorCache& operator=(const ColorCache&) = delete;
    ~ColorCache();

    // Returns the transform to apply, or nullptr when none is needed: the profile is
    // sRGB, unreadable, or doesn't match the colour mode. `description` receives the
    // profile's name (with a note when it is ignored). Thread-safe.
    const ColorTransform* get(const std::vector<uint8_t>& icc, int colorMode, std::wstring& description);

private:
    struct Entry;
    std::mutex m_;
    std::vector<std::unique_ptr<Entry>> entries_;
};

// Converts a resized 4-byte-per-pixel buffer to sRGB RGBA8888 in place. Runs after the
// resize so it touches only the small image. With no transform: CMYK gets a naive
// conversion (no profile available), anything else is left alone.
void to_srgb(const ColorTransform* xf, int colorMode, uint8_t* pixels, size_t count);

}  // namespace pp
