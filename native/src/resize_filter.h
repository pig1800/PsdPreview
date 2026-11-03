#pragma once

namespace pp {

// The one resampling filter of both backends (cpu_resize.cpp via IPP, gpu_resize.cu on CUDA):
// a Mitchell-Netravali cubic, widened by the reduction factor when shrinking (antialiased),
// pixel centres at (i + 0.5) * src / dst - 0.5, border pixels replicated.
//   shrinking  B = 0.3, C = 2.2: fitted to Photoshop's "Bicubic Sharper (reduction)". On a
//              1280 -> 450 px page it is within 47.3 dB PSNR of Photoshop's output
//              (antialiased Catmull-Rom: 36 dB; the old non-antialiased cubic, which skipped
//              most source pixels: 25-30 dB). See README "Resize filter".
//   enlarging  Catmull-Rom (B = 0, C = 0.5), which interpolates: same size = unchanged.
struct CubicFilter {
    float B, C;
};

inline CubicFilter resize_filter(int srcWidth, int dstWidth) {
    return dstWidth < srcWidth ? CubicFilter{0.3f, 2.2f} : CubicFilter{0.0f, 0.5f};
}

}  // namespace pp
