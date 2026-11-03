#pragma once
#include "common.h"

namespace pp {

// mozjpeg, progressive, optimised Huffman, 4:2:0, overshoot deringing. By default without
// trellis quantisation and without the progressive scan search: ~5x faster than mozjpeg's
// max-compression profile for ~11% larger files at the same quality setting (which then also
// measures better: SSIM 0.974 vs 0.972 at q82). maxCompression = `cjpeg -quality N` exactly.
// Input is RGBA8888; the 4th byte is ignored (JCS_EXT_RGBX).
bool encode_jpeg(const uint8_t* rgba, int w, int h, int quality, bool maxCompression, Bytes& out, std::wstring& error);

// Rough output size (for memory planning), and encoder working memory.
inline uint64_t jpeg_estimated_bytes(uint64_t w, uint64_t h) { return w * h / 2 + 4096; }
inline uint64_t jpeg_working_bytes(uint64_t w, uint64_t h) { return w * h * 4 + (1 << 20); }

}  // namespace pp
