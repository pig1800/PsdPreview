// PSD/PSB composite reader.
//
// psd_sdk provides the document structure (CreateDocument: header + section offsets,
// PSD/PSB aware) and the bounds-checked PackBits decoder (imageUtil::DecompressRle).
// The composite is not read with psd_sdk's ParseImageDataSection because that decodes
// every channel (incl. alpha) into full planes; here only the colour channels are read,
// a band of rows at a time, and merged straight into 4-byte pixels with SIMD.
#include "psd.h"

#include "simd.h"

// psd_sdk headers expect its precompiled header first.
#include "PsdPch.h"
#include "PsdDecompressRle.h"
#include "PsdDocument.h"
#include "PsdFile.h"
#include "PsdMallocAllocator.h"
#include "PsdParseDocument.h"
#include "PsdSection.h"

namespace pp {
namespace {

// psd::File over our positional reader; short reads are zero-filled and reported
// through DoWaitForRead (which the patched SyncFileReader checks).
class SdkFile final : public psd::File {
public:
    SdkFile(const FileReader& r, psd::Allocator* a) : psd::File(a), r_(r) {}
    bool readFailed() const { return failed_; }

private:
    bool DoOpenRead(const wchar_t*) override { return true; }
    bool DoOpenWrite(const wchar_t*) override { return false; }
    bool DoClose() override { return true; }
    ReadOperation DoRead(void* buffer, uint32_t count, uint64_t position) override {
        size_t got = r_.read_at(position, buffer, count);
        if (got == count) return buffer;
        failed_ = true;
        std::memset(static_cast<uint8_t*>(buffer) + got, 0, count - got);
        return nullptr;
    }
    bool DoWaitForRead(ReadOperation& op) override { return op != nullptr; }
    WriteOperation DoWrite(const void*, uint32_t, uint64_t) override { return nullptr; }
    bool DoWaitForWrite(WriteOperation&) override { return false; }
    uint64_t DoGetSize() const override { return r_.size(); }

    const FileReader& r_;
    bool failed_ = false;
};

const wchar_t* color_mode_name(int m) {
    switch (m) {
        case 0: return L"Bitmap";
        case 1: return L"Grayscale";
        case 2: return L"Indexed";
        case 3: return L"RGB";
        case 4: return L"CMYK";
        case 7: return L"Multichannel";
        case 8: return L"Duotone";
        case 9: return L"Lab";
        default: return L"unknown";
    }
}

constexpr uint16_t kIccProfileResource = 0x040F;
constexpr uint16_t kVersionInfoResource = 0x0421;
constexpr uint64_t kBandBytes = 4ull << 20;  // compressed bytes read per band (all channels)

bool is_resource_signature(const uint8_t* s) {
    return !memcmp(s, "8BIM", 4) || !memcmp(s, "MeSa", 4) || !memcmp(s, "AgHg", 4) || !memcmp(s, "PHUT", 4) ||
           !memcmp(s, "DCSR", 4);
}

// Walks image-resource block headers (payloads are not read) to find Version Info's
// hasRealMergedData byte and the location of the embedded ICC profile.
void scan_resources(const FileReader& f, uint64_t pos, uint64_t end, PsdHeader& h) {
    uint8_t hdr[8];
    while (pos + 12 <= end && f.read_exact(pos, hdr, 7)) {
        if (!is_resource_signature(hdr)) break;
        const uint16_t id = be16(hdr + 4);
        uint64_t nameLen = 1 + hdr[6];
        nameLen += nameLen & 1;  // Pascal string padded to even length
        const uint64_t sizePos = pos + 6 + nameLen;
        if (sizePos + 4 > end || !f.read_exact(sizePos, hdr, 4)) break;
        const uint32_t size = be32(hdr);
        const uint64_t dataPos = sizePos + 4;
        if (dataPos + size > end) break;
        uint8_t flag;
        if (id == kVersionInfoResource && size >= 5 && f.read_exact(dataPos + 4, &flag, 1)) h.mergedFlag = flag ? 1 : 0;
        if (id == kIccProfileResource && size > 0) {
            h.iccPos = dataPos;
            h.iccSize = size;
        }
        pos = dataPos + size + (size & 1);
    }
}

uint64_t rle_count_bytes(const PsdHeader& h) { return h.version == 2 ? 4 : 2; }

// Byte counts of every row of the colour channels we need (the channels*height table
// precedes the data; alpha/spot channel counts are not read).
FileStatus read_rle_counts(const FileReader& f, const PsdHeader& h, std::vector<uint32_t>& counts) {
    const uint64_t cs = rle_count_bytes(h);
    const uint64_t rows = uint64_t(h.sourceChannels()) * h.height;
    const uint64_t tableBytes = uint64_t(h.channels) * h.height * cs;
    if (h.imageDataPos + tableBytes > h.fileSize)
        return {PP_FILE_TRUNCATED, L"file ends inside the composite's RLE row table"};
    std::vector<uint8_t> raw(rows * cs);
    if (!f.read_exact(h.imageDataPos, raw.data(), raw.size()))
        return {PP_FILE_TRUNCATED, L"could not read the composite's RLE row table"};
    counts.resize(rows);
    for (uint64_t i = 0; i < rows; ++i) counts[i] = cs == 4 ? be32(&raw[i * 4]) : be16(&raw[i * 2]);
    return {};
}

// Every row encodes to a single run => the composite is a flat placeholder.
bool rle_is_blank(const PsdHeader& h, const std::vector<uint32_t>& counts) {
    const uint64_t rb = h.rowBytes();
    const uint64_t blankRow = (rb + 126) / 127 * 2 + 2;
    return !counts.empty() && *std::max_element(counts.begin(), counts.end()) <= blankRow;
}

// 32-bit documents are linear light: encode with the sRGB curve when reducing to 8 bits.
const uint8_t* linear_to_srgb_lut() {
    static const auto lut = [] {
        std::array<uint8_t, 4096> t{};
        for (int i = 0; i < 4096; ++i) {
            const double v = i / 4095.0;
            const double s = v <= 0.0031308 ? 12.92 * v : 1.055 * std::pow(v, 1 / 2.4) - 0.055;
            t[i] = uint8_t(std::lround(s * 255));
        }
        return t;
    }();
    return lut.data();
}

// Big-endian float32 -> 8-bit, in place (rare format, scalar).
void float32_to_8(uint8_t* row, uint32_t n) {
    const uint8_t* lut = linear_to_srgb_lut();
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t bits = be32(row + 4 * size_t(i));
        float v;
        std::memcpy(&v, &bits, 4);
        v = v > 0 ? (v < 1 ? v : 1) : 0;  // also maps NaN to 0
        row[i] = lut[int(v * 4095 + 0.5f)];
    }
}

}  // namespace

FileStatus read_psd_header(const FileReader& f, PsdHeader& h) {
    h = {};
    h.fileSize = f.size();

    // Pre-check what CreateDocument would reject, so we get precise codes.
    uint8_t b[26];
    if (!f.read_exact(0, b, sizeof b) || memcmp(b, "8BPS", 4) != 0)
        return {PP_FILE_NOT_PSD, L"not a Photoshop file (bad signature)"};
    const int version = be16(b + 4);
    if (version != 1 && version != 2) return {PP_FILE_UNSUPPORTED_VERSION, std::format(L"unsupported PSD version {}", version)};
    if (b[6] | b[7] | b[8] | b[9] | b[10] | b[11]) return {PP_FILE_CORRUPT, L"corrupt header (reserved bytes not zero)"};

    psd::MallocAllocator allocator;
    SdkFile sdkFile(f, &allocator);
    psd::Document* doc = psd::CreateDocument(&sdkFile, &allocator);
    if (!doc) return {PP_FILE_CORRUPT, L"psd_sdk could not parse the document header"};
    h.version = doc->isLargeDocument ? 2 : 1;
    h.channels = int(doc->channelCount);
    h.width = doc->width;
    h.height = doc->height;
    h.depth = int(doc->bitsPerChannel);
    h.colorMode = int(doc->colorMode);
    const psd::Section resources = doc->imageResourcesSection;
    const psd::Section layers = doc->layerMaskInfoSection;
    const psd::Section image = doc->imageDataSection;
    psd::DestroyDocument(doc, &allocator);
    if (sdkFile.readFailed()) return {PP_FILE_TRUNCATED, L"file ends inside the header"};

    if (h.width == 0 || h.height == 0 || h.channels == 0)
        return {PP_FILE_BAD_DIMENSIONS, std::format(L"invalid dimensions {}x{}, {} channels", h.width, h.height, h.channels)};
    if (h.colorMode != 1 && h.colorMode != 3 && h.colorMode != 4)
        return {PP_FILE_UNSUPPORTED_COLOR_MODE, std::format(L"unsupported colour mode {}", color_mode_name(h.colorMode))};
    if (h.depth != 8 && h.depth != 16 && h.depth != 32)
        return {PP_FILE_UNSUPPORTED_DEPTH, std::format(L"unsupported bit depth {}", h.depth)};
    if (h.channels < h.sourceChannels())
        return {PP_FILE_CORRUPT, std::format(L"{} mode needs {} channels, file has {}", color_mode_name(h.colorMode),
                                             h.sourceChannels(), h.channels)};

    scan_resources(f, resources.offset, resources.offset + resources.length, h);
    if (h.mergedFlag == 0)
        return {PP_FILE_NO_COMPOSITE, L"no flattened composite: saved with \"Maximize Compatibility\" off"};

    // CreateDocument clamps section lengths to the file size, so a cut-off file
    // shows up as an empty image data section. Re-read the declared length for the message.
    if (image.length < 2) {
        const uint64_t lenBytes = h.version == 2 ? 8 : 4;
        uint8_t raw[8] = {};
        f.read_exact(layers.offset - lenBytes, raw, size_t(lenBytes));
        const uint64_t declared = lenBytes == 8 ? be64(raw) : be32(raw);
        return {PP_FILE_TRUNCATED, std::format(L"file truncated: layer data should end at byte {}, file is only {} bytes",
                                               layers.offset + declared, h.fileSize)};
    }
    uint8_t comp[2];
    if (!f.read_exact(image.offset, comp, 2)) return {PP_FILE_TRUNCATED, L"file ends before the composite"};
    h.compression = be16(comp);
    h.imageDataPos = image.offset + 2;
    if (h.compression == 2 || h.compression == 3)
        return {PP_FILE_UNSUPPORTED_COMPRESSION, L"ZIP-compressed composite is not supported"};
    if (h.compression != 0 && h.compression != 1)
        return {PP_FILE_UNSUPPORTED_COMPRESSION, std::format(L"unknown composite compression {}", h.compression)};
    return {};
}

FileStatus check_composite(const FileReader& f, const PsdHeader& h) {
    const uint64_t rb = h.rowBytes();
    const int need = h.sourceChannels();
    if (h.compression == 0) {
        const uint64_t plane = rb * h.height;
        if (h.imageDataPos + plane * need > h.fileSize) return {PP_FILE_TRUNCATED, L"file ends inside the composite"};
        if (h.mergedFlag == -1) {  // no Photoshop flag: sample the data for content
            uint8_t first = 0, buf[4096];
            bool varied = false;
            for (int k = 0; k < 16 && !varied; ++k) {
                uint64_t off = h.imageDataPos + plane * need * k / 16;
                size_t n = f.read_at(off, buf, sizeof buf);
                if (k == 0 && n) first = buf[0];
                for (size_t i = 0; i < n && !varied; ++i) varied = buf[i] != first;
            }
            if (!varied) return {PP_FILE_BLANK_COMPOSITE, L"composite is blank (written without a real flattened image)"};
        }
        return {};
    }
    std::vector<uint32_t> counts;
    FileStatus st = read_rle_counts(f, h, counts);
    if (!st.ok()) return st;
    // Sanity bound only (catches misread tables): an ideal PackBits encoder needs
    // rb + rb/128, but third-party encoders can be less efficient; 2x is the worst case.
    const uint64_t maxRow = rb * 2 + 16;
    uint64_t total = 0;
    for (uint32_t c : counts) {
        if (c == 0 || c > maxRow) return {PP_FILE_CORRUPT, L"composite RLE row table is corrupt"};
        total += c;
    }
    const uint64_t dataStart = h.imageDataPos + uint64_t(h.channels) * h.height * rle_count_bytes(h);
    if (dataStart + total > h.fileSize)
        return {PP_FILE_TRUNCATED, std::format(L"file truncated inside the composite: needs {} bytes, file is {} bytes",
                                               dataStart + total, h.fileSize)};
    if (h.mergedFlag == -1 && rle_is_blank(h, counts))
        return {PP_FILE_BLANK_COMPOSITE, L"composite is blank (written without a real flattened image)"};
    return {};
}

uint64_t decode_scratch_bytes(const PsdHeader& h) {
    const uint64_t rb = h.rowBytes(), need = uint64_t(h.sourceChannels());
    if (h.compression == 0) return need * std::max<uint64_t>(rb, kBandBytes / need + rb);
    const uint64_t counts = need * h.height * (rle_count_bytes(h) + 4);
    return need * rb + kBandBytes + need * (2 * rb + 16) + counts;
}

bool read_icc_profile(const FileReader& f, const PsdHeader& h, std::vector<uint8_t>& icc) {
    if (h.iccSize == 0) return false;
    icc.resize(h.iccSize);
    return f.read_exact(h.iccPos, icc.data(), icc.size());
}

FileStatus decode_composite(const FileReader& f, const PsdHeader& h, Bytes& out) {
    struct ImageSink final : RowSink {
        uint8_t* base;
        size_t stride;
        uint8_t* row(uint32_t y) override { return base + y * stride; }
    };
    out.resize(h.decodedBytes());
    ImageSink sink;
    sink.base = out.data();
    sink.stride = size_t(h.width) * kPixelBytes;
    return decode_composite(f, h, sink);
}

FileStatus decode_composite(const FileReader& f, const PsdHeader& h, RowSink& sink) {
    const uint32_t W = h.width, H = h.height;
    const uint64_t rb = h.rowBytes();
    const int need = h.sourceChannels();
    const simd::Kernels& k = simd::kernels();

    auto to8 = [&](uint8_t* row) {  // in place: deep samples -> 8-bit at the start of the row
        if (h.depth == 16) k.narrow16(row, row, W);
        else if (h.depth == 32) float32_to_8(row, W);
    };
    auto emit = [&](uint8_t* const* rows, uint32_t y) {
        uint8_t* dst = sink.row(y);
        if (h.colorMode == 1) k.merge_gray(rows[0], dst, W);
        else if (h.colorMode == 3) k.merge_rgb(rows[0], rows[1], rows[2], dst, W);
        else k.merge_4(rows[0], rows[1], rows[2], rows[3], dst, W);  // CMYK: converted after the resize
    };
    uint8_t* rows[4] = {};

    if (h.compression == 0) {  // raw planes: read a strip of rows of every channel
        const uint32_t strip = uint32_t(std::clamp<uint64_t>(kBandBytes / (rb * need), 1, H));
        Bytes buf(size_t(need) * strip * rb);
        for (uint32_t y0 = 0; y0 < H; y0 += strip) {
            const uint32_t n = std::min(strip, H - y0);
            for (int c = 0; c < need; ++c) {
                const uint64_t pos = h.imageDataPos + (uint64_t(c) * H + y0) * rb;
                if (!f.read_exact(pos, buf.data() + uint64_t(c) * strip * rb, n * rb))
                    return {PP_FILE_TRUNCATED, std::format(L"file truncated inside composite channel {}", c)};
            }
            for (uint32_t y = 0; y < n; ++y) {
                for (int c = 0; c < need; ++c) {
                    rows[c] = buf.data() + (uint64_t(c) * strip + y) * rb;
                    to8(rows[c]);
                }
                emit(rows, y0 + y);
            }
        }
        return {};
    }

    // RLE: per band of rows, read each channel's compressed bytes (contiguous within
    // the channel), unpack row by row, narrow, merge.
    std::vector<uint32_t> counts;
    FileStatus st = read_rle_counts(f, h, counts);
    if (!st.ok()) return st;
    if (h.mergedFlag == -1 && rle_is_blank(h, counts))
        return {PP_FILE_BLANK_COMPOSITE, L"composite is blank (written without a real flattened image)"};

    Bytes rowBuf(size_t(need) * rb);
    for (int c = 0; c < need; ++c) rows[c] = rowBuf.data() + uint64_t(c) * rb;
    uint64_t cursor[4] = {};  // file offset of each channel's next unread row
    uint64_t pos = h.imageDataPos + uint64_t(h.channels) * H * rle_count_bytes(h);
    for (int c = 0; c < need; ++c) {
        cursor[c] = pos;
        for (uint32_t y = 0; y < H; ++y) pos += counts[uint64_t(c) * H + y];
    }
    Bytes band[4];
    for (uint32_t y = 0; y < H;) {
        uint64_t len[4] = {}, total = 0;
        uint32_t y1 = y;
        do {
            for (int c = 0; c < need; ++c) {
                len[c] += counts[uint64_t(c) * H + y1];
                total += counts[uint64_t(c) * H + y1];
            }
            ++y1;
        } while (y1 < H && total < kBandBytes);

        const uint8_t* src[4] = {};
        for (int c = 0; c < need; ++c) {
            band[c].resize(len[c]);
            if (!f.read_exact(cursor[c], band[c].data(), len[c]))
                return {PP_FILE_TRUNCATED, std::format(L"file truncated inside composite channel {}", c)};
            cursor[c] += len[c];
            src[c] = band[c].data();
        }
        for (; y < y1; ++y) {
            for (int c = 0; c < need; ++c) {
                const uint32_t n = counts[uint64_t(c) * H + y];
                if (!psd::imageUtil::DecompressRle(src[c], n, rows[c], rb))
                    return {PP_FILE_CORRUPT, std::format(L"corrupt RLE data in composite channel {} row {}", c, y)};
                src[c] += n;
                to8(rows[c]);
            }
            emit(rows, y);
        }
    }
    return {};
}

}  // namespace pp
