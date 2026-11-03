// CUDA resize kernel. Not compiled by the CMake build: scripts\build-ptx.cmd turns it into
// gpu_resize.ptx (checked in), which CMake embeds and gpu.cpp JIT-loads with
// cudaLibraryLoadData. Run the script after editing this file.
//
// Same filter as the CPU path (see resize_filter.h): Mitchell-Netravali cubic (B, C), support
// widened by the reduction factor when shrinking, pixel centres at (i + 0.5) * scale - 0.5,
// border pixels replicated, weights normalised. One thread per output pixel, all four 8-bit
// channels (RGBA or CMYK); the two 1-D weight sets are computed on the fly, so there are no
// tables to upload and no intermediate image.

// 6 x the Mitchell-Netravali kernel (the normalisation divides the 6 out).
__device__ __forceinline__ float cubic6(float x, float B, float C) {
    x = fabsf(x);
    if (x < 1.f) return ((12.f - 9.f * B - 6.f * C) * x + (-18.f + 12.f * B + 6.f * C)) * x * x + (6.f - 2.f * B);
    if (x < 2.f) return (((-B - 6.f * C) * x + (6.f * B + 30.f * C)) * x + (-12.f * B - 48.f * C)) * x + (8.f * B + 24.f * C);
    return 0.f;
}

__device__ __forceinline__ unsigned char to_u8(float v) { return (unsigned char)fminf(fmaxf(rintf(v), 0.f), 255.f); }

extern "C" __global__ void pp_resize_cubic(const uchar4* __restrict__ src, unsigned long long srcPitch, int w, int h,
                                           uchar4* __restrict__ dst, unsigned long long dstPitch, int dw, int dh,
                                           float B, float C) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;

    const float sx = float(w) / float(dw), sy = float(h) / float(dh);
    const float fx = fmaxf(sx, 1.f), fy = fmaxf(sy, 1.f);  // kernel width (in source pixels) / 2
    const float ix = 1.f / fx, iy = 1.f / fy;
    const float cx = (x + 0.5f) * sx - 0.5f, cy = (y + 0.5f) * sy - 0.5f;
    const int x0 = int(ceilf(cx - 2.f * fx)), x1 = int(floorf(cx + 2.f * fx));
    const int y0 = int(ceilf(cy - 2.f * fy)), y1 = int(floorf(cy + 2.f * fy));

    float wsx = 0.f;
    for (int i = x0; i <= x1; ++i) wsx += cubic6((i - cx) * ix, B, C);

    float ax = 0.f, ay = 0.f, az = 0.f, aw = 0.f, wsy = 0.f;
    for (int j = y0; j <= y1; ++j) {
        const float wy = cubic6((j - cy) * iy, B, C);
        if (wy == 0.f) continue;
        wsy += wy;
        const uchar4* row = reinterpret_cast<const uchar4*>(reinterpret_cast<const unsigned char*>(src) +
                                                            size_t(min(max(j, 0), h - 1)) * srcPitch);
        float rx = 0.f, ry = 0.f, rz = 0.f, rw = 0.f;
        for (int i = x0; i <= x1; ++i) {
            const float wx = cubic6((i - cx) * ix, B, C);
            const uchar4 p = row[min(max(i, 0), w - 1)];
            rx += wx * p.x;
            ry += wx * p.y;
            rz += wx * p.z;
            rw += wx * p.w;
        }
        ax += wy * rx;
        ay += wy * ry;
        az += wy * rz;
        aw += wy * rw;
    }
    const float n = 1.f / (wsx * wsy);
    uchar4* out = reinterpret_cast<uchar4*>(reinterpret_cast<unsigned char*>(dst) + size_t(y) * dstPitch) + x;
    *out = make_uchar4(to_u8(ax * n), to_u8(ay * n), to_u8(az * n), to_u8(aw * n));
}
