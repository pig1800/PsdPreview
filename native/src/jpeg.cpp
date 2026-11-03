#include "jpeg.h"

#include <csetjmp>
#include <cstdio>

#include <jpeglib.h>

// libjpeg reports errors via longjmp; nothing with a destructor lives between
// setjmp and the libjpeg calls below, so these warnings don't apply.
#pragma warning(disable : 4324 4611)

namespace pp {
namespace {

struct ErrorMgr {
    jpeg_error_mgr pub;
    jmp_buf jump;
    char message[JMSG_LENGTH_MAX];
};

void on_error(j_common_ptr cinfo) {
    auto* e = reinterpret_cast<ErrorMgr*>(cinfo->err);
    e->pub.format_message(cinfo, e->message);
    longjmp(e->jump, 1);
}
void on_message(j_common_ptr) {}  // keep warnings off stdout

struct VectorDest {
    jpeg_destination_mgr pub;
    Bytes* out;
};

void dest_init(j_compress_ptr cinfo) {
    auto* d = reinterpret_cast<VectorDest*>(cinfo->dest);
    d->pub.next_output_byte = d->out->data();
    d->pub.free_in_buffer = d->out->size();
}
boolean dest_grow(j_compress_ptr cinfo) {
    auto* d = reinterpret_cast<VectorDest*>(cinfo->dest);
    size_t used = d->out->size();
    d->out->resize(used * 2);
    d->pub.next_output_byte = d->out->data() + used;
    d->pub.free_in_buffer = d->out->size() - used;
    return TRUE;
}
void dest_term(j_compress_ptr cinfo) {
    auto* d = reinterpret_cast<VectorDest*>(cinfo->dest);
    d->out->resize(d->out->size() - d->pub.free_in_buffer);
}

}  // namespace

bool encode_jpeg(const uint8_t* rgba, int w, int h, int quality, bool maxCompression, Bytes& out, std::wstring& error) {
    if (w <= 0 || h <= 0 || w > PP_MAX_JPEG_DIMENSION || h > PP_MAX_JPEG_DIMENSION) {
        error = std::format(L"{}x{} exceeds the JPEG size limit", w, h);
        return false;
    }
    out.clear();
    out.resize(std::max<size_t>(64 * 1024, size_t(jpeg_estimated_bytes(w, h))));

    jpeg_compress_struct cinfo{};
    ErrorMgr err{};
    VectorDest dest{};
    cinfo.err = jpeg_std_error(&err.pub);
    err.pub.error_exit = on_error;
    err.pub.output_message = on_message;
    if (setjmp(err.jump)) {
        jpeg_destroy_compress(&cinfo);
        error = L"mozjpeg: " + widen(err.message);
        out.clear();
        return false;
    }
    jpeg_create_compress(&cinfo);
    dest.pub.init_destination = dest_init;
    dest.pub.empty_output_buffer = dest_grow;
    dest.pub.term_destination = dest_term;
    dest.out = &out;
    cinfo.dest = &dest.pub;

    cinfo.image_width = JDIMENSION(w);
    cinfo.image_height = JDIMENSION(h);
    cinfo.input_components = kPixelBytes;
    cinfo.in_color_space = JCS_EXT_RGBX;  // RGBA8888, alpha byte skipped
    jpeg_set_defaults(&cinfo);  // mozjpeg: JCP_MAX_COMPRESSION profile
    if (!maxCompression) {
        // The progressive scan search is ~1/3 of the max-compression encode time and trellis
        // quantisation ~3/4 of the rest (see jpeg.h). The search only changes the scan layout
        // (decoded pixels are identical).
        jpeg_c_set_bool_param(&cinfo, JBOOLEAN_TRELLIS_QUANT, FALSE);
        jpeg_c_set_bool_param(&cinfo, JBOOLEAN_TRELLIS_QUANT_DC, FALSE);
        jpeg_c_set_bool_param(&cinfo, JBOOLEAN_OPTIMIZE_SCANS, FALSE);
        jpeg_simple_progression(&cinfo);  // libjpeg's standard script instead of the search's candidates
    }
    jpeg_set_quality(&cinfo, quality, TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    const size_t stride = size_t(w) * kPixelBytes;
    JSAMPROW rows[16];
    while (cinfo.next_scanline < cinfo.image_height) {
        JDIMENSION n = std::min<JDIMENSION>(16, cinfo.image_height - cinfo.next_scanline);
        for (JDIMENSION i = 0; i < n; ++i)
            rows[i] = const_cast<JSAMPROW>(rgba + size_t(cinfo.next_scanline + i) * stride);
        jpeg_write_scanlines(&cinfo, rows, n);
    }
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    out.shrink_to_fit();
    return true;
}

}  // namespace pp
