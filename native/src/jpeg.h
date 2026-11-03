#pragma once
#include "common.h"

namespace pp {

// mozjpeg with its default (max-compression) profile: progressive, trellis
// quantisation, optimised Huffman, 4:2:0 — i.e. what `cjpeg -quality N` does.
// Input is RGBA8888; the 4th byte is ignored (JCS_EXT_RGBX).
bool encode_jpeg(const uint8_t* rgba, int w, int h, int quality, Bytes& out, std::wstring& error);

// Rough output size (for memory planning), and encoder working memory.
inline uint64_t jpeg_estimated_bytes(uint64_t w, uint64_t h) { return w * h / 2 + 4096; }
inline uint64_t jpeg_working_bytes(uint64_t w, uint64_t h) { return w * h * 4 + (1 << 20); }

}  // namespace pp
