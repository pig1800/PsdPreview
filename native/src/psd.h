#pragma once
#include "common.h"

namespace pp {

struct PsdHeader {
    int version = 0;     // 1 = PSD, 2 = PSB
    int channels = 0;    // channels stored in the composite (incl. alpha/spot)
    uint32_t width = 0;
    uint32_t height = 0;
    int depth = 0;       // bits per channel: 8, 16 or 32
    int colorMode = 0;   // 1 gray, 3 RGB, 4 CMYK
    int mergedFlag = -1; // Version Info hasRealMergedData; -1 if the resource is absent
    int compression = -1;
    uint64_t imageDataPos = 0; // first byte after the composite's compression field
    uint64_t fileSize = 0;
    uint64_t iccPos = 0;       // embedded ICC profile (image resource 1039); iccSize 0 = none
    uint32_t iccSize = 0;

    int sourceChannels() const { return colorMode == 1 ? 1 : colorMode == 4 ? 4 : 3; }
    uint64_t rowBytes() const { return (uint64_t(width) * depth + 7) / 8; }  // one channel row in the file
    uint64_t decodedBytes() const { return uint64_t(width) * height * kPixelBytes; }
};

// Receives decoded rows (4 bytes/pixel) in order y = 0..height-1. The decoder writes row y
// into the memory returned by row(y) before asking for row y+1.
class RowSink {
public:
    virtual uint8_t* row(uint32_t y) = 0;

protected:
    ~RowSink() = default;
};

struct FileStatus {
    int code = PP_FILE_OK;
    std::wstring message;
    bool ok() const { return code == PP_FILE_OK; }
};

// Parses everything up to the composite's compression field and checks the
// format is supported. Reads only section lengths and resource headers.
FileStatus read_psd_header(const FileReader& f, PsdHeader& h);

// Structural check of the composite (RLE table vs. file size, blank composite).
FileStatus check_composite(const FileReader& f, const PsdHeader& h);

// Decodes the composite into 4 bytes per pixel. Only the colour channels are read
// (alpha/spot channels skipped; Photoshop stores the composite colour already matted):
//   RGB  -> R,G,B,255      Gray -> G,G,G,255      CMYK -> C,M,Y,K as stored (inverted)
// Deeper channels are reduced to 8 bits: 16-bit by truncation, 32-bit (linear float)
// by sRGB encoding. Row bands are decompressed, narrowed and merged with SIMD kernels.
FileStatus decode_composite(const FileReader& f, const PsdHeader& h, RowSink& sink);
FileStatus decode_composite(const FileReader& f, const PsdHeader& h, Bytes& out);  // whole image, reusing out's capacity

// Reads the embedded ICC profile, if any.
bool read_icc_profile(const FileReader& f, const PsdHeader& h, std::vector<uint8_t>& icc);

// Upper bound of the per-thread scratch memory decode_composite needs.
uint64_t decode_scratch_bytes(const PsdHeader& h);

}  // namespace pp
