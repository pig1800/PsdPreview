// CUDA kernels of the GPU path. Not compiled by the CMake build: scripts\build-ptx.cmd turns
// this file into gpu_cuda.ptx (checked in), which CMake embeds and gpu_cuda.cpp JIT-loads with
// cudaLibraryLoadData. Run the script after editing this file.
//
// The GPU path uploads the composite's colour channels still PackBits-compressed (~1.35 bytes
// per pixel instead of 4 decoded), then
//   pp_unpack_rows    PackBits -> 8-bit planes, one warp per row
//   pp_resize_cubicN  planes (N = 1 gray, 3 RGB, 4 CMYK) -> resized 4-byte pixels

// ---- PackBits ---------------------------------------------------------------------------------

// Row r (channel-major: channel r / height, row r % height) of the colour channels: compressed
// bytes comp[offsets[r] .. offsets[r + 1]), rowBytes bytes unpacked, written to planes + r *
// width. 16-bit samples (shift = 1) keep their first, i.e. high, byte, like the CPU path's
// narrowing. Same semantics as unpack_row in psd.cpp: a run past the row end is cut; a short or
// malformed row is zero-filled and the lowest such row index is left in *firstBad.
//
// One warp per row: all 32 lanes read each run header (one broadcast load, so the control flow
// is uniform) and copy the run's <= 128 bytes together, with coalesced stores.
extern "C" __global__ void pp_unpack_rows(const unsigned char* __restrict__ comp, const unsigned int* __restrict__ offsets,
                                          int rows, int rowBytes, int shift, unsigned char* __restrict__ planes, int width,
                                          unsigned int* __restrict__ firstBad) {
    const int lane = threadIdx.x & 31;
    const int r = int((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    if (r >= rows) return;  // whole warps
    const unsigned char* src = comp + offsets[r];
    const unsigned char* const srcEnd = comp + offsets[r + 1];
    unsigned char* const dst = planes + size_t(r) * width;
    const int keep = (1 << shift) - 1;  // byte k of the unpacked row is kept if (k & keep) == 0
    int k = 0;
    while (k < rowBytes && src < srcEnd) {
        const int b = *src++;
        if (b == 0x80) continue;  // no-op
        if (b < 0x80) {           // literal: the next b + 1 bytes
            const int n = b + 1;
            if (n > srcEnd - src) break;
            const int m = min(n, rowBytes - k);
            for (int i = lane; i < m; i += 32)
                if (((k + i) & keep) == 0) dst[(k + i) >> shift] = src[i];
            src += n;
            k += m;
        } else {  // repeat: the next byte 257 - b times
            if (src == srcEnd) break;
            const unsigned char v = *src++;
            const int m = min(257 - b, rowBytes - k);
            for (int i = lane; i < m; i += 32)
                if (((k + i) & keep) == 0) dst[(k + i) >> shift] = v;
            k += m;
        }
    }
    if (k < rowBytes) {
        for (int i = ((k + keep) >> shift) + lane; i < width; i += 32) dst[i] = 0;
        if (lane == 0) atomicMin(firstBad, unsigned(r));
    }
}

// ---- Resize -----------------------------------------------------------------------------------
//
// Same filter as the CPU path (see resize_filter.h): Mitchell-Netravali cubic (B, C), support
// widened by the reduction factor when shrinking, pixel centres at (i + 0.5) * scale - 0.5,
// border pixels replicated, weights normalised. Output: RGBA (gray expanded, A = 255) or CMYK.
//
// Separable, in one kernel and without an intermediate image: a block of 32 x 8 threads makes
// 32 x 8 output pixels. It streams the source rows those 8 output rows need through shared
// memory in chunks: each source row is first filtered horizontally for the block's 32 output
// columns, then every thread adds its output row's vertical taps from shared memory. Each
// source row is filtered once per block instead of once per output row, which cuts the loads
// ~3x at a 2.8x reduction compared with a direct 2-D gather. Weights are computed on the fly:
// no tables to upload.

// 6 x the Mitchell-Netravali kernel (the normalisation divides the 6 out).
__device__ __forceinline__ float cubic6(float x, float B, float C) {
    x = fabsf(x);
    if (x < 1.f) return ((12.f - 9.f * B - 6.f * C) * x + (-18.f + 12.f * B + 6.f * C)) * x * x + (6.f - 2.f * B);
    if (x < 2.f) return (((-B - 6.f * C) * x + (6.f * B + 30.f * C)) * x + (-12.f * B - 48.f * C)) * x + (8.f * B + 24.f * C);
    return 0.f;
}

__device__ __forceinline__ unsigned char to_u8(float v) { return (unsigned char)fminf(fmaxf(rintf(v), 0.f), 255.f); }

constexpr int kTileW = 32, kTileH = 8;  // block = output tile; gpu_cuda.cpp launches 32 x 8 threads
constexpr int kChunk = 64;              // source rows per pass through shared memory (32 KB)

// planes: NC planes of w x h bytes, `plane` bytes apart.
template <int NC>
__device__ __forceinline__ void resize_cubic(const unsigned char* __restrict__ planes, unsigned long long plane, int w, int h,
                                             uchar4* __restrict__ dst, unsigned long long dstPitch, int dw, int dh, float B,
                                             float C) {
    __shared__ float4 H[kChunk][kTileW];  // horizontally filtered source rows
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int x = blockIdx.x * kTileW + tx, y = blockIdx.y * kTileH + ty;
    const int yLast = min(int(blockIdx.y) * kTileH + kTileH, dh) - 1;  // the block's last output row

    const float sx = float(w) / float(dw), sy = float(h) / float(dh);
    const float fx = fmaxf(sx, 1.f), fy = fmaxf(sy, 1.f);  // kernel half-width / 2 in source pixels
    const float ix = 1.f / fx, iy = 1.f / fy;
    const float cx = (x + 0.5f) * sx - 0.5f, cy = (y + 0.5f) * sy - 0.5f;
    const int x0 = int(ceilf(cx - 2.f * fx)), x1 = int(floorf(cx + 2.f * fx));
    const int y0 = int(ceilf(cy - 2.f * fy)), y1 = int(floorf(cy + 2.f * fy));
    // Source rows the whole block needs (one spare row each side against rounding differences).
    const int r0 = int(ceilf((blockIdx.y * kTileH + 0.5f) * sy - 0.5f - 2.f * fy)) - 1;
    const int r1 = int(floorf((yLast + 0.5f) * sy - 0.5f + 2.f * fy)) + 1;

    float wsx = 0.f;
    for (int i = x0; i <= x1; ++i) wsx += cubic6((i - cx) * ix, B, C);

    float ax = 0.f, ay = 0.f, az = 0.f, aw = 0.f, wsy = 0.f;
    for (int c0 = r0; c0 <= r1; c0 += kChunk) {  // every thread runs the same chunks (barriers)
        const int n = min(kChunk, r1 - c0 + 1);
        for (int r = ty; r < n && x < dw; r += kTileH) {  // columns past dw have nothing to filter
            const unsigned char* row = planes + size_t(min(max(c0 + r, 0), h - 1)) * w;
            float rx = 0.f, ry = 0.f, rz = 0.f, rw = 0.f;
            for (int i = x0; i <= x1; ++i) {
                const float wx = cubic6((i - cx) * ix, B, C);
                const int xi = min(max(i, 0), w - 1);
                rx += wx * row[xi];
                if (NC >= 3) {
                    ry += wx * row[plane + xi];
                    rz += wx * row[2 * plane + xi];
                }
                if (NC == 4) rw += wx * row[3 * plane + xi];
            }
            H[r][tx] = make_float4(rx, ry, rz, rw);
        }
        __syncthreads();
        const int jEnd = min(y1, c0 + n - 1);
        for (int j = max(y0, c0); j <= jEnd; ++j) {
            const float wy = cubic6((j - cy) * iy, B, C);
            if (wy == 0.f) continue;
            wsy += wy;
            const float4 r = H[j - c0][tx];
            ax += wy * r.x;
            if (NC >= 3) {
                ay += wy * r.y;
                az += wy * r.z;
            }
            if (NC == 4) aw += wy * r.w;
        }
        __syncthreads();
    }
    if (x >= dw || y >= dh) return;
    const float n = 1.f / (wsx * wsy);
    const unsigned char v0 = to_u8(ax * n);
    uchar4 p;
    if (NC == 1) p = make_uchar4(v0, v0, v0, 255);
    else if (NC == 3) p = make_uchar4(v0, to_u8(ay * n), to_u8(az * n), 255);
    else p = make_uchar4(v0, to_u8(ay * n), to_u8(az * n), to_u8(aw * n));
    *(reinterpret_cast<uchar4*>(reinterpret_cast<unsigned char*>(dst) + size_t(y) * dstPitch) + x) = p;
}

#define PP_RESIZE_KERNEL(NC)                                                                                        \
    extern "C" __global__ void __launch_bounds__(kTileW* kTileH)                                                    \
        pp_resize_cubic##NC(const unsigned char* __restrict__ planes, unsigned long long plane, int w, int h,       \
                            uchar4* __restrict__ dst, unsigned long long dstPitch, int dw, int dh, float B, float C) { \
        resize_cubic<NC>(planes, plane, w, h, dst, dstPitch, dw, dh, B, C);                                         \
    }
PP_RESIZE_KERNEL(1)
PP_RESIZE_KERNEL(3)
PP_RESIZE_KERNEL(4)
