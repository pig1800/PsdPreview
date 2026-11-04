// D3D12 compute kernels of the GPU path (gpu_d3d12.cpp). The build compiles them with the
// Windows SDK's dxc into DXIL headers, one per entry point:
//   dxc -T cs_6_0 -E <entry> -D PP_<KERNEL> [-D NC=n] -Fh <header> -Vn <array>
// Same algorithms and results as the CUDA kernels (gpu_cuda.cu), rewritten for HLSL:
//   - HLSL can't store single bytes, so threads assemble whole 32-bit words;
//   - the wave width differs between GPUs (4 lanes on WARP, 32 on NVIDIA, 32/64 on AMD, wider
//     on Adreno), so nothing here depends on it;
//   - integrated GPUs have a fraction of a desktop GPU's lanes, bandwidth and caches, so the
//     unpack is a thread per row (see below) rather than the CUDA kernel's warp per row.
// All buffers are raw (ByteAddressBuffer), bound as root descriptors: no descriptor heaps.

#define PP_RS "RootConstants(num32BitConstants=12, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1)"

// ---- status ----------------------------------------------------------------------------------
// u1 holds the first corrupt row (0xFFFFFFFF = none); pp_clear resets it before each image.

#if defined(PP_CLEAR)
RWByteAddressBuffer status : register(u1);

[RootSignature(PP_RS)]
[numthreads(1, 1, 1)]
void pp_clear() { status.Store(0, 0xFFFFFFFFu); }
#endif

// ---- PackBits ---------------------------------------------------------------------------------
//
// Row r (channel-major: channel r / height, row r % height) of the colour channels: compressed
// bytes comp[offsets[r] .. offsets[r + 1]), rowBytes bytes unpacked, written as `width` bytes to
// planes + r * pitch (pitch: a multiple of 16; the padding is zeroed). 16-bit samples (shift 1)
// keep their first, i.e. high, byte. Same semantics as unpack_row in psd.cpp: a run past the row
// end is cut; a short or malformed row is zero-filled and the lowest such row is left in status.
//
// One thread per row, decoding it front to back as the CPU does; a page has tens of thousands of
// rows, plenty to fill any GPU. Each thread reads and writes 16 bytes at a time (its rows are
// far apart from its neighbours', so every access is a cache line of its own). Measured
// alternatives, all slower on an integrated GPU: the CUDA kernel's warp per row (every lane
// walks every run), and two variants that stage rows through groupshared memory (too few waves
// in flight to hide its memory latency).

#if defined(PP_UNPACK)
ByteAddressBuffer comp : register(t0);
ByteAddressBuffer offsets : register(t1);
RWByteAddressBuffer planes : register(u0);
RWByteAddressBuffer status : register(u1);
cbuffer Unpack : register(b0) {
    uint rows, rowBytes, shift, width, pitch;
    uint groupsPerX;  // a dispatch has at most 65535 groups per axis
};

#define GROUP 64

// Compressed bytes, through a 16-byte block cache. (The buffer has 16 bytes of slack.)
struct Reader {
    uint4 block;
    uint at;  // address of `block`
    uint get(uint a) {
        if ((a & ~15u) != at) {
            at = a & ~15u;
            block = comp.Load4(at);
        }
        const uint i = (a >> 2) & 3u;
        const uint w = i == 0 ? block.x : i == 1 ? block.y : i == 2 ? block.z : block.w;
        return (w >> ((a & 3u) * 8u)) & 0xFFu;
    }
};

// Output bytes, assembled into words and stored 16 bytes at a time.
struct Writer {
    uint4 block;
    uint words, word, n;  // complete words in `block`; bytes in `word`
    uint addr;            // where `block` goes
    void put_word(uint w) {
        if (words == 0) block.x = w;
        else if (words == 1) block.y = w;
        else if (words == 2) block.z = w;
        else block.w = w;
        if (++words == 4) {
            planes.Store4(addr, block);
            addr += 16;
            words = 0;
        }
    }
    void put(uint b) {
        word |= b << (8u * n);
        if (++n == 4) {
            put_word(word);
            word = 0;
            n = 0;
        }
    }
    // The last, partial word and block; then zeros up to `end`.
    void finish(uint end) {
        if (n != 0) put_word(word);
        if (words != 0) {
            if (words < 2) block.y = 0;
            if (words < 3) block.z = 0;
            block.w = 0;
            planes.Store4(addr, block);
            addr += 16;
        }
        for (; addr < end; addr += 16) planes.Store4(addr, uint4(0, 0, 0, 0));
    }
};

[RootSignature(PP_RS)]
[numthreads(GROUP, 1, 1)]
void pp_unpack(uint3 group : SV_GroupID, uint lane : SV_GroupIndex) {
    const uint r = (group.y * groupsPerX + group.x) * GROUP + lane;
    if (r >= rows) return;
    uint src = offsets.Load(r * 4);
    const uint end = offsets.Load(r * 4 + 4);
    Reader rd;
    rd.at = 0xFFFFFFFFu;
    rd.block = uint4(0, 0, 0, 0);
    Writer wr;
    wr.block = uint4(0, 0, 0, 0);
    wr.words = 0;
    wr.word = 0;
    wr.n = 0;
    wr.addr = r * pitch;
    const uint keep = (1u << shift) - 1u;  // unpacked byte k is kept if (k & keep) == 0
    uint k = 0;                            // unpacked bytes done
    bool bad = false;
    [loop] while (k < rowBytes) {
        if (src >= end) { bad = true; break; }
        const uint b = rd.get(src++);
        if (b == 0x80) continue;  // no-op
        if (b < 0x80) {           // literal: the next b + 1 bytes
            const uint len = b + 1;
            if (len > end - src) { bad = true; break; }
            const uint m = min(len, rowBytes - k);
            [loop] for (uint i = 0; i < m; ++i)
                if (((k + i) & keep) == 0) wr.put(rd.get(src + i));
            src += len;
            k += m;
        } else {  // repeat: the next byte 257 - b times
            if (src >= end) { bad = true; break; }
            const uint v = rd.get(src++);
            const uint m = min(257 - b, rowBytes - k);
            uint count = ((k + m + keep) >> shift) - ((k + keep) >> shift);  // kept bytes among them
            k += m;
            [loop] for (; count > 0 && wr.n != 0; --count) wr.put(v);
            const uint vvvv = v * 0x01010101u;
            [loop] for (; count >= 4; count -= 4) wr.put_word(vvvv);
            [loop] for (; count > 0; --count) wr.put(v);
        }
    }
    wr.finish(r * pitch + pitch);  // short/malformed row: zeros; and the padding
    if (bad) {
        uint previous;
        status.InterlockedMin(0, r, previous);
    }
}
#endif

// ---- Resize -----------------------------------------------------------------------------------
//
// Same filter and the same separable single-kernel scheme as pp_resize_cubicN in gpu_cuda.cu
// (see the comments there): Mitchell-Netravali cubic (B, C), support widened by the reduction
// factor when shrinking, pixel centres at (i + 0.5) * scale - 0.5, border pixels replicated,
// weights normalised. A 32 x 8 group makes 32 x 8 output pixels, streaming the source rows they
// need through groupshared memory, filtered horizontally once per group.
// NC planes (1 gray, 3 RGB, 4 CMYK) of w x h bytes, rows `pitch` apart, planes `plane` apart
// (or with RGBA defined: w x h 4-byte pixels, as the CPU decoder makes them, rows `pitch` apart) ->
// 4-byte pixels (RGBA with gray expanded and A = 255, or CMYK). Group (0, 0) also copies the
// status word behind the result, so one read brings both back.
// Tried on an Adreno X1 and no faster there: staging each row's span in groupshared memory as
// floats with weight tables for both passes (40 instead of 31 ms per page), smaller chunks,
// forcing wave 64/128, a vertical weight table, unrolling the row loop. What mattered was the
// fixed tap count (FIXED, below).

#if defined(PP_RESIZE)
ByteAddressBuffer planes : register(t0);
RWByteAddressBuffer dst : register(u0);
RWByteAddressBuffer status : register(u1);
cbuffer Resize : register(b0) {
    uint w, h, pitch, plane, dw, dh, dstPitch;
    float B, C;
    uint statusOffset;
};

#define TILE_W 32
#define TILE_H 8
#define CHUNK 40  // source rows per pass through groupshared memory (enough for 3x at once)
#define TAPS 64   // horizontal taps per column the weight table holds (shrinking up to ~15x)

groupshared float4 H[CHUNK][TILE_W];      // horizontally filtered source rows (20 KB)
groupshared float WX[TILE_W][TAPS + 1];   // each column's horizontal weights (+ 1: no bank conflicts)

uint plane_byte(uint a) { return (planes.Load(a & ~3u) >> ((a & 3u) * 8u)) & 0xFFu; }

// 6 x the Mitchell-Netravali kernel (the normalisation divides the 6 out).
float cubic6(float x) {
    x = abs(x);
    if (x < 1.0f) return ((12.0f - 9.0f * B - 6.0f * C) * x + (-18.0f + 12.0f * B + 6.0f * C)) * x * x + (6.0f - 2.0f * B);
    if (x < 2.0f) return (((-B - 6.0f * C) * x + (6.0f * B + 30.0f * C)) * x + (-12.0f * B - 48.0f * C)) * x + (8.0f * B + 24.0f * C);
    return 0.0f;
}

uint to_u8(float v) { return uint(clamp(round(v), 0.0f, 255.0f)); }

[RootSignature(PP_RS)]
[numthreads(TILE_W, TILE_H, 1)]
void pp_resize(uint3 group : SV_GroupID, uint3 t : SV_GroupThreadID) {
    const int tx = int(t.x), ty = int(t.y);
    const int iw = int(w), ih = int(h), idw = int(dw), idh = int(dh);
    const int x = int(group.x) * TILE_W + tx, y = int(group.y) * TILE_H + ty;
    const int yLast = min(int(group.y) * TILE_H + TILE_H, idh) - 1;  // the group's last output row

    const float sx = float(w) / float(dw), sy = float(h) / float(dh);
    const float fx = max(sx, 1.0f), fy = max(sy, 1.0f);  // kernel half-width / 2 in source pixels
    const float ix = 1.0f / fx, iy = 1.0f / fy;
    const float cx = (float(x) + 0.5f) * sx - 0.5f, cy = (float(y) + 0.5f) * sy - 0.5f;
    const int x0 = int(ceil(cx - 2.0f * fx)), x1 = int(floor(cx + 2.0f * fx));
    const int y0 = int(ceil(cy - 2.0f * fy)), y1 = int(floor(cy + 2.0f * fy));
    // Source rows the whole group needs (one spare row each side against rounding differences).
    const int r0 = int(ceil((float(group.y * TILE_H) + 0.5f) * sy - 0.5f - 2.0f * fy)) - 1;
    const int r1 = int(floor((float(yLast) + 0.5f) * sy - 0.5f + 2.0f * fy)) + 1;

    float wsx = 0.0f;
    for (int i = x0; i <= x1; ++i) wsx += cubic6((float(i) - cx) * ix);
    // The horizontal weights depend only on the column: computed once into the table instead of
    // for every source row. Wider kernels compute them per tap. With FIXED (the host picks the
    // variant whose tap count covers the scale) the tap loop has that many iterations, unrolled,
    // the table zero-padded: a loop with a run-time count made the whole kernel 2.2x slower on
    // an Adreno X1 (25.6 instead of 11.5 ms per page).
    const bool table = int(4.0f * fx) + 2 <= TAPS;
#ifdef FIXED
    for (int k = ty; k < FIXED; k += TILE_H) WX[tx][k] = k <= x1 - x0 ? cubic6((float(x0 + k) - cx) * ix) : 0.0f;
#else
    if (table)
        for (int k = ty; k <= x1 - x0; k += TILE_H) WX[tx][k] = cubic6((float(x0 + k) - cx) * ix);
#endif
    GroupMemoryBarrierWithGroupSync();

    float4 acc = 0.0f;
    float wsy = 0.0f;
    for (int c0 = r0; c0 <= r1; c0 += CHUNK) {  // every thread runs the same chunks (barriers)
        const int n = min(CHUNK, r1 - c0 + 1);
        for (int r = ty; r < n && x < idw; r += TILE_H) {  // columns past dw have nothing to filter
            const uint rowAddr = uint(clamp(c0 + r, 0, ih - 1)) * pitch;
            float4 s = 0.0f;
#ifdef FIXED
            [unroll] for (int k = 0; k < FIXED; ++k) {
                const int i = x0 + k;
                const float wx = WX[tx][k];
#else
            for (int i = x0; i <= x1; ++i) {
                const float wx = table ? WX[tx][i - x0] : cubic6((float(i) - cx) * ix);
#endif
#if defined(RGBA)  // 4-byte pixels (the hybrid path: decoded by the CPU), one load per tap
                const uint v = planes.Load(rowAddr + 4u * uint(clamp(i, 0, iw - 1)));
                s += wx * float4(float(v & 0xFFu), float((v >> 8) & 0xFFu), float((v >> 16) & 0xFFu), float(v >> 24));
#else
                const uint a = rowAddr + uint(clamp(i, 0, iw - 1));
                s.x += wx * float(plane_byte(a));
#if NC >= 3
                s.y += wx * float(plane_byte(a + plane));
                s.z += wx * float(plane_byte(a + 2 * plane));
#endif
#if NC == 4
                s.w += wx * float(plane_byte(a + 3 * plane));
#endif
#endif
            }
            H[r][tx] = s;
        }
        GroupMemoryBarrierWithGroupSync();
        const int jEnd = min(y1, c0 + n - 1);
        for (int j = max(y0, c0); j <= jEnd; ++j) {
            const float wy = cubic6((float(j) - cy) * iy);
            if (wy == 0.0f) continue;
            wsy += wy;
            acc += wy * H[j - c0][tx];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (group.x == 0 && group.y == 0 && tx == 0 && ty == 0) dst.Store(statusOffset, status.Load(0));
    if (x >= idw || y >= idh) return;
    const float norm = 1.0f / (wsx * wsy);
    const uint v0 = to_u8(acc.x * norm);
#if NC == 1
    const uint p = v0 | v0 << 8 | v0 << 16 | 0xFF000000u;
#elif NC == 3
    const uint p = v0 | to_u8(acc.y * norm) << 8 | to_u8(acc.z * norm) << 16 | 0xFF000000u;
#else
    const uint p = v0 | to_u8(acc.y * norm) << 8 | to_u8(acc.z * norm) << 16 | to_u8(acc.w * norm) << 24;
#endif
    dst.Store(uint(y) * dstPitch + uint(x) * 4, p);
}
#endif
