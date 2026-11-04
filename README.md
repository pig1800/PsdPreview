# PsdPreview

Extracts the flattened full-resolution composite stored in PSD/PSB files, resizes it to a
target width (antialiased bicubic fitted to Photoshop's *Bicubic Sharper*), converts embedded
colour profiles to sRGB (Little CMS 2), and encodes JPEG with mozjpeg (q82). Decoding and
resizing run on the GPU when there is one worth using — CUDA on NVIDIA, D3D12 compute on any
other GPU (AMD, Intel, Qualcomm Adreno) — else on the CPU: Intel IPP with AVX-512 on x64,
stb_image_resize2 with Neon on ARM64 (Windows on Snapdragon).
Native C++ DLL (x64 and ARM64) with a C# (.NET 8) wrapper.

## Build

```
git clone --recursive https://github.com/pig1800/PsdPreview.git
cd PsdPreview
build.cmd
```

Runs `scripts\setup-deps.cmd` (initialises the psd_sdk submodule if needed, downloads Intel IPP
static libs from NuGet, builds mozjpeg 4.1.5, fetches Little CMS 2.20 source into
`third_party\`; each skipped if present),
`scripts\build-native.cmd` (CMake + Ninja → `build\native\bin\`), then
`dotnet build PsdPreview.sln -c Release`.

Requirements: Git, VS 2026 C++ tools, CMake ≥ 3.24, NASM, a Windows 10/11 SDK (its `dxc`
compiles the D3D12 kernels), CUDA Toolkit 13 (headers/libs only — no nvcc needed), .NET 8 SDK.

**ARM64** (Windows on Arm), cross-compiled on x64 or built on an ARM64 machine — needs VS's
"C++ ARM64 build tools" component; no CUDA, no IPP:

```
scripts\setup-deps.cmd arm64      :: + mozjpeg for ARM64 (Neon) in third_party\mozjpeg-arm64
scripts\build-native.cmd arm64    :: -> build\native-arm64\bin\
```

The ARM64 mozjpeg gets `scripts\mozjpeg-neon-fdct.patch`: its Neon forward DCT overflowed 16 bits
on blocks that mozjpeg's overshoot deringing had pushed past 255 (black text on white, say), and
whole rows of such a block came out inverted — 221 of 20 million test blocks. Patched, it is
bit-exact with the C DCT, and ARM64 JPEGs are byte-identical to x64's.

The CUDA kernels (`native\src\gpu_cuda.cu`) are checked in compiled, as `gpu_cuda.ptx`, which
CMake embeds into the DLL. After editing them, run `scripts\build-ptx.cmd`: it needs nvcc
and VS 2019 or 2022 C++ tools (e.g. VS 2019 Build Tools) as nvcc's host compiler, because
CUDA 13.0 can't use VS 2026's (its STL requires CUDA 13.2+). The D3D12 kernels
(`native\src\gpu_d3d12.hlsl`) are compiled by the build (dxc → DXIL, embedded).

## Deployment

`psdpreview_native.dll` is the only file: IPP, mozjpeg, lcms2, psd_sdk and cudart are linked
statically against the static CRT (`/MT`), so it imports only Windows system DLLs (`KERNEL32`,
`d3d12`, `dxgi`, the `api-ms-win-core-synch` API set) — no VC++ redistributable, no CUDA DLLs.
The GPU code is our own kernels: PTX that the NVIDIA driver JIT-compiles (Turing / sm_75 or
newer, driver for CUDA 13+), and DXIL (shader model 6.0) for any D3D12 driver. Without a usable
GPU, everything works on the CPU path.

The C# library is architecture-neutral. Referencing `managed\PsdPreview\PsdPreview.csproj`
copies each built `psdpreview_native.dll` to `x64\` / `arm64\` in the app's output, and the
library loads the one matching the process (an x64 app under emulation on ARM64 gets the x64
one). For a **single-file** app (`win-arm64` for ARM64):

```
dotnet publish <YourApp> -c Release -r win-x64 --self-contained ^
    -p:PublishSingleFile=true -p:IncludeNativeLibrariesForSelfExtract=true ^
    -p:EnableCompressionInSingleFile=true
```

The native DLL is bundled into the exe and extracted on first run to
`%TEMP%\.net\<app>\<hash>\` (override with `DOTNET_BUNDLE_EXTRACT_BASE_DIR`). The demo
published this way is one 36 MB exe (70 MB uncompressed) that runs from an otherwise empty
folder on both the GPU and CPU paths.

## C# usage

```csharp
using PsdPreview;

// once, at program start (failure is fine: processing then uses the CPU)
GpuInitResult gpu = PsdPreviewer.InitializeGpu();   // Auto: CUDA, else D3D12 on the fastest GPU
Console.WriteLine(gpu);                              // e.g. "D3D12: Qualcomm(R) Adreno(TM) ..., UMA, ..."

// 1. can every file be processed?
IReadOnlyList<PsdFileProblem> problems = PsdPreviewer.ValidateFiles(files, targetWidth: 720);
foreach (var p in problems) Console.WriteLine($"{p.Path}: {p.Message}");   // p.Kind for logic

// 2. enough memory?
MemoryCheckResult mem = PsdPreviewer.CheckMemory(files, 720);
if (!mem.IsEnough) Console.WriteLine(mem);   // "RAM needed 8.57 GB, available 34.96 GB; VRAM ..."
                                             // mem.HostShortfallBytes / DeviceShortfallBytes

// 3. process (progress.Value goes 0 → 1, poll it from a UI timer)
using var progress = new ProgressCounter();
using PsdPreviewBatch batch = await PsdPreviewer.ProcessAsync(files, 720, progress);
foreach (PsdPreviewImage img in batch)            // same order as `files`
{
    if (img.Succeeded) Save(img.FileName, img.Jpeg!);   // img.SourcePath = input path as given
    else Console.WriteLine($"{img.SourcePath}: {img.ErrorMessage}");
    // img.ColorProfile: embedded ICC description (null if none); img.ColorConverted: converted to sRGB
}
Console.WriteLine(batch.Backend);                 // what actually ran
```

`InitializeGpu(backend, adapter)` can also pick `GpuBackend.Cuda`, `GpuBackend.D3D12` (adapter =
index in DXGI's high-performance order, 0 = the fastest) or `GpuBackend.D3D12Warp` (Windows'
software rasterizer, for testing).

`PsdPreviewOptions` (all optional): `JpegQuality` (82), `ForceCpu`, `ExtractThreads` (default:
logical cores; they share up to that many GPU streamers), `EncodeThreads` (default: logical cores),
`MaxInFlight` (default 2 × cores), `MaxCompression` (mozjpeg's max-compression profile: ~11%
smaller files, ~3.5× slower overall, see [Encode](#pipeline)). `batch.Stats` reports
thread-seconds per stage (decode, resize, encode, extract threads waiting for encoders, encoders
idle) and GPU fallbacks; on the GPU path "decode" is reading + uploading the compressed data and
"resize" is the GPU's unpack + resize + download.
`PsdPreviewer.GetCpuBackend()` reports the SIMD level and the CPU resizer (IPP dispatch, or
stb_image_resize2 on ARM64).

`samples\PsdPreview.Demo` is a runnable end-to-end example:
`PsdPreview.Demo <dir-or-files> [--width 720] [--gpu auto|cuda|d3d12|warp] [--adapter N] [--cpu] [--out DIR]
[--limit N] [--quality 82] [--max-compression] [--extract N] [--encode N] [--validate-only] [--skip-validate]`.

Environment variables (diagnostics): `PP_GPU_PROFILE=1` adds the D3D12 kernels' GPU time per
image to `batch.Backend`; `PP_D3D12_DEBUG=1` enables the D3D12 debug layer (needs the Graphics
Tools feature), its messages on stderr; `PP_GPU_BALANCE=0` sends every image the GPU takes to the
GPU (see [CPU/GPU split](#cpugpu-split)); `PP_CPU_RESIZE=stb` uses stb_image_resize2 on x64 too;
`PP_SIMD=sse2|avx2|avx512` lowers the SIMD level. `PP_GPU_HYBRID=0` takes the full GPU path on
UMA; `PP_D3D12_NONCOHERENT=1` makes the CPU-writable heaps write-combined on any UMA adapter and
`PP_D3D12_DIRECT=0` copies results instead of writing them in CPU-readable memory (to test the
non-cache-coherent handling).

The C ABI is in `native\include\psdpreview.h` (blittable structs, UTF-16 paths).

## What validation reports

| Kind | Meaning |
|---|---|
| `NoComposite` | Photoshop's `hasRealMergedData` flag is 0 — saved with *Maximize Compatibility* off |
| `BlankComposite` | no Photoshop flag (other writer) and every composite row is a single run |
| `Truncated` | file ends before / inside the composite (e.g. an incomplete copy) |
| `Corrupt` | inconsistent header or RLE row table |
| `UnsupportedColorMode/Depth/Compression` | Bitmap/Indexed/Lab/Duotone/Multichannel, 1-bit, ZIP composite |
| `OutputTooLarge` | resized height > 65500 (JPEG limit) — only checked when a target width is given |
| `NotPsd`, `OpenFailed`, `BadDimensions` | as named |

## Formats and colour

PSD and PSB; RGB, Grayscale, CMYK; 8/16/32-bit; RLE and raw composites. Only the header,
section lengths, image-resource headers (plus the ICC profile payload) and the composite's
colour channels are read — layers, masks, alpha channels and other metadata are skipped.

- **Bit depth:** 16-bit is truncated to 8 (high byte, SIMD). 32-bit (Photoshop's linear-light
  float) is clamped and sRGB-encoded through a 4096-entry table (scalar; rare format); its ICC
  conversion is skipped.
- **Colour management:** an embedded ICC profile other than sRGB (Display P3, Adobe RGB, monitor
  profiles, CMYK or gray profiles…) is converted to sRGB with Little CMS 2 — relative
  colorimetric + black point compensation, Photoshop's default — **after** the resize, so only
  the small image is transformed. "sRGB" in the profile description = no conversion. One
  transform per distinct profile per batch. CMYK stays CMYK through the resize; without a
  profile it gets a naive conversion. Unreadable profiles, or ones that don't match the colour
  mode, are ignored (reported in `ColorProfile`).

## Pipeline

One fixed slot per input file. Extract threads decode **and** resize; encode threads take
each image as soon as it is resized:

1. **Decode and resize** (each extract thread), to the target width (height keeps aspect ratio),
   4 channels, with the filter described [below](#resize-filter). Also reads the ICC profile and
   looks up its transform.
   - GPU (RLE 8/16-bit and raw 8-bit composites) — push/process/pull through a *streamer*, from a
     pool the extract threads share (as many streamers as GPU memory allows, up to one per
     thread). CUDA, each streamer with its own non-blocking CUDA stream and two 4 MB pinned band
     buffers:
     - push: the colour channels' data, still PackBits-compressed (~1.35 bytes/px for webtoon
       pages instead of 4 decoded), is read from the file straight into the current band; a full
       band is uploaded asynchronously while the next one is read into the other. The row
       offsets (prefix sums of the RLE table) go up the same way. The CPU does nothing but read.
     - process: `pp_unpack_rows` unpacks every row into 8-bit planes (one warp per row: the lanes
       share each run header and copy the run together; 16-bit keeps the high byte; a corrupt row
       is reported back as a `Corrupt` file), then `pp_resize_cubicN` resizes the planes.
     - pull: the result comes back through the pinned bands into the encoder's buffer. (A direct
       download into pageable memory is staged by the driver synchronously and stalled every
       other thread's CUDA calls: 17 thread-s of the 25 the uploads took before.)

     Waits use blocking-sync events, so waiting threads sleep instead of spinning. Device memory
     comes from a private stream-ordered pool (`cudaMallocFromPoolAsync`), so the 16 streams
     overlap freely. An image the GPU can't take (VRAM exhausted, 32-bit, raw 16-bit) goes the
     CPU way.

     D3D12 (`gpu_d3d12.cpp`, kernels in `gpu_d3d12.hlsl`): one compute and one copy queue
     shared by all streamers; per streamer command lists, fences waited on with events (sleeping),
     and grow-only buffers kept for the batch, bound as root descriptors (no descriptor heaps).
     - integrated GPU (UMA: Adreno, AMD/Intel APUs): the file is read straight into memory the
       GPU reads (a custom heap with the upload heap's CPU properties) — no bands, no copies;
     - discrete GPU: two 4 MB upload-heap bands and the copy queue, band k + 1 read while band k
       copies, as with CUDA;
     - both: the resize kernel writes its result straight into CPU-readable memory (a UAV in a
       custom heap with the readback heap's properties) when the driver allows it, else into
       VRAM followed by a copy to a readback buffer.

     HLSL has no byte stores and the wave width varies (4 lanes on WARP, 32 on NVIDIA, 32/64 on
     AMD, wider on Adreno), so the D3D12 unpack is one thread per row decoding front to back and
     storing 16 bytes at a time. On an integrated GPU (a 2-CU Radeon) that is ~50 ms per page
     where a port of CUDA's warp per row, which walks every run with every lane, took ~100;
     storing 4 bytes at a time took 72, and two variants staging rows through groupshared memory
     115–135 (too few waves in flight to hide memory latency). The resize is the CUDA kernel's scheme, with each column's horizontal weights
     computed once into groupshared memory, and its tap loop unrolled to a fixed count (8, 12,
     16 or 24 variants, the smallest that covers the scale): with a run-time count the RGBA
     kernel was 2.2× slower on the Adreno X1 and the Radeon iGPU alike (planar: 1.7× and 1.3×). At init a self-test checks an
     unpack byte for byte and a shrink, then a 768 × 1024 calibration image times the kernels.
   - Hybrid (integrated GPUs, the default there; `PP_GPU_HYBRID=0` for the path above): the CPU
     decodes the composite as on the CPU path, but straight into memory the GPU reads (the
     streamer's CPU-writable custom heap, mapped); the GPU resizes those 4-byte pixels (one
     32-bit load per tap) into a CPU-readable custom heap buffer from a pool, which the encoder
     then reads where the GPU wrote it. No copies either way, any format the CPU decodes.
     `GpuInitResult.Hybrid` says which path a GPU takes.
   - Non-cache-coherent UMA (`IsCacheCoherentUma` false; the 9800X3D's Radeon iGPU is one): the
     CPU-writable heaps are write-combined, i.e. uncached, so the CPU only ever writes them, and
     sequentially (file reads, the decoder's merged rows, RLE offsets); everything the CPU reads
     is in readback-type (write-back) memory, mapped after the GPU's fence so `Map` can
     invalidate stale cache lines. A driver may refuse a UAV in that memory, or get it wrong, so
     at init the hybrid path is checked byte for byte through a pooled result buffer the CPU has
     read and overwritten in between; on a failure results are written to VRAM and copied into a
     readback buffer by the GPU (no cost measured on the Radeon: 6.60 s vs 6.56 s for 200 pages
     with every page on the GPU), and if that fails too, the full GPU path is used.
   - CPU: per band of rows (~4 MB of compressed data), read each colour channel's bytes,
     PackBits-decode row by row (`unpack_row`), reduce >8-bit samples, and merge the planes into
     4-byte pixels with SIMD: RGBA8888 (A = 255; gray expanded) or CMYK, into a per-thread buffer
     reused across files; then `ippiResizeAntialiasing_8u_C4R` (`ippiResizeCubic_8u_C4R` when
     enlarging) on x64, stb_image_resize2 with the same kernel on ARM64. `unpack_row` has psd_sdk's `DecompressRle` semantics but copies each run in fixed
     32-byte chunks (plain vector moves) instead of a `memcpy`/`memset` call per run of ≤ 128
     bytes: 4× faster, and PackBits was ~90% of the decode time.
2. **Encode** (N threads): ICC → sRGB if needed, then mozjpeg reads the buffer as
   `JCS_EXT_RGBX`: progressive, optimised Huffman, 4:2:0, overshoot deringing, q82 — mozjpeg's
   max-compression profile minus trellis quantisation and the progressive scan search. On 36 real
   pages at q82 that is 5.2× faster to encode for 11% larger files, at better quality (SSIM 0.974
   vs 0.972, PSNR 36.4 vs 34.4 dB, since trellis trades detail for size); the scan search alone
   costs 1.5× for 0.4% and doesn't change a pixel. At equal file size trellis does look slightly
   better (q78 without it: SSIM 0.969), hence `MaxCompression` for when size matters more.
   Baseline instead of progressive would encode another 2× faster for 4% larger files (1.2
   instead of 1.4 s for 200 pages at 425 px). Resized buffers are recycled through a small pool.

A counting semaphore caps files "between decode start and encode end" (`MaxInFlight`), which
bounds peak memory; `CheckMemory` computes the worst case from the actual headers using the
same configuration.

Progress counts 3 steps per file (extract, resize, encode); failed files have their remaining
steps added immediately, so progress always reaches 1.

### CPU/GPU split

A GPU slower than the CPU path would make a batch slower, so the GPU gets an image only if the
work already queued there wouldn't keep it waiting longer than decoding and resizing it on the
CPU takes. Both costs are per source megapixel: the GPU's from its kernels' timestamps (the
calibration image first), the CPU's measured under load as images go that way. A fast GPU never
builds such a backlog (the RTX takes ~0.3 ms/Mpx against the CPU path's ~11 ms/Mpx per thread).
For the hybrid path the CPU work saved is the resize only, so that is what the backlog is
compared with (decode and resize costs are measured separately). An *integrated* GPU slower
than half the extract threads' combined CPU throughput at that work isn't used at all: it adds
little and takes memory bandwidth from the encoders. Both integrated GPUs measured fall under
that line, and the measurements agree: balanced, the hybrid path was within ±3% of the CPU
alone on the Radeon iGPU and on the Adreno X1-85 (see
[Measured](#measured-snapdragon-x-elite-x1e78100-adreno-x1-85)). `PP_GPU_BALANCE=1` balances
without that rule. `batch.Backend`
says how many images went to the CPU while the GPU was busy, or that the GPU wasn't used.

### Resize filter

Shrinking uses a Mitchell–Netravali cubic, B = 0.3 and C = 2.2, whose support is widened by the
reduction factor (antialiased: every source pixel contributes). Pixel centres map as
`(i + 0.5) · src/dst − 0.5`, border pixels are replicated, and the weights are normalised. The
parameters were fitted to Photoshop's *Bicubic Sharper (reduction)*. Enlarging uses Catmull-Rom
(B = 0, C = 0.5), which interpolates, so a same-size resize changes nothing. `resize_filter.h`
holds the parameters for every backend:

- GPU: `pp_resize_cubicN` in `gpu_cuda.cu`, separable in one kernel: a 32 × 8 block streams
  the source rows its output rows need through shared memory, filters each horizontally once,
  then each thread sums its vertical taps from there. That reads ~3× less than a direct 2-D
  gather (0.8 instead of 1.4 ms for 1280 × 19500 → 450) with identical results, and needs no
  intermediate image. `pp_resize` in `gpu_d3d12.hlsl` is the same in HLSL.
- CPU: IPP's `ippiResizeAntialiasing` with the same cubic (x64); stb_image_resize2 v2.50
  (`native\vendor`, public domain) with the kernel as a filter callback (ARM64). stb matches IPP
  to within 1 level (100.6 dB) and is as fast single-threaded; under 16 threads 11–24% slower.

The backends agree to within rounding: luma PSNR of each against Photoshop is the same 46.70 dB,
and 64–82 dB between any two of CUDA, D3D12 (NVIDIA, AMD, WARP), IPP and stb (JPEG q100). At the
source width every backend gives byte-identical JPEGs (fuzzed PackBits: odd widths, 8/16-bit,
gray/RGB/CMYK, no-ops, runs past the row end, malformed rows).

On a 1280 × 19500 webtoon page reduced to 450 px wide, luma PSNR against Photoshop's lossless
*Bicubic Sharper* output (JPEG chroma subsampling puts every JPEG's RGB PSNR near 37 dB, so luma
is the fair measure):

| | luma PSNR |
|---|---|
| this filter, unencoded | 47.8 dB |
| this filter, our JPEG at q100 (GPU and CPU alike) | 46.7 dB |
| Photoshop's own JPEG export of its result | 40.8 dB |
| antialiased Lanczos-3 | 38.8 dB |
| antialiased Catmull-Rom (B = 0, C = 0.5) | 36.5 dB |
| before: IPP cubic without antialiasing, JPEG q100 | 30.6 dB |
| before: NPP cubic, JPEG q100 (also offset by a fraction of a pixel) | 24.8 dB |

A cubic without antialiasing reads only a 4 × 4 neighbourhood whatever the scale, so at a 2–4×
reduction most source pixels are skipped. That causes aliasing: moiré, thin lines broken into
dots, and text strokes of uneven thickness. Photoshop's kernel is sharper than any common cubic.
B and C came from a grid search on that page, ending in a ridge of near-equal fits. A free-form
least-squares fit of Photoshop's 1-D kernel is ≈ 0.97 at 0, crosses zero at ±1.1 output pixels
and dips to −0.27 at ±1.5; this cubic has the same −0.27 dip. A cubic additionally stretched by
1.07 would gain 0.6 dB, but IPP can't stretch its kernels, so all backends share the plain one.

### SIMD (plane merge, 16 → 8-bit)

One translation unit per instruction set (`simd_sse2.cpp`, `simd_avx2.cpp` with `/arch:AVX2`,
`simd_avx512.cpp` with `/arch:AVX512`), dispatched at runtime via CPUID + XGETBV: AVX-512 (F+BW)
→ AVX2 → SSE2. AVX1 has no 256-bit integer unpacks/shuffles, so AVX1-only CPUs use SSE2. The
merge is the unpack8 → unpack16 scheme of PSDBatchExporter's `copyPanarPixels*`, with the AVX2
lane fix-up by `permute2x128` and, for AVX-512, a 4×4 transpose of 128-bit lanes
(`shuffle_i64x2`); narrowing uses `and` + `packus` (SSE2/AVX2) or `vpmovwb` (AVX-512). The
kernel header only declares functions, so no AVX-compiled inline code can leak into baseline
paths. `PP_SIMD=sse2|avx2|avx512` lowers the level (testing). ARM64 has `simd_neon.cpp` (Neon is
part of the ARMv8 baseline), where the interleaving `vld2q`/`vst4q` do the whole merge.

Single-row (cache-resident, 1500 px) throughput on the 9800X3D:

| | merge_rgb | narrow16 |
|---|---|---|
| scalar | 2.6 Gpx/s | – |
| SSE2 | 20.7 Gpx/s | 34 Gsamples/s |
| AVX2 | 33.0 Gpx/s | 80 Gsamples/s |
| AVX-512 | 39.7 Gpx/s | 95 Gsamples/s |

Whole-file decode (CPU path) runs at ~1380 Mpx/s per thread on every level (~450 with psd_sdk's
PackBits decoder); PackBits still dominates.

### IPP and AVX-512

IPP is linked statically and initialised once before any worker starts. On AMD, `ippInit()`
reports `ippStsNonIntelCpu`; if the CPU/OS support AVX-512 F/CD/BW/DQ/VL but IPP did not
dispatch to its AVX-512 code (`k0`), the library forces it with `ippSetCpuFeatures`.
(IPP 2026.1 already picks `k0` on a Ryzen 7 9800X3D; the override is for builds/CPUs where it
would fall back to `l9`/AVX2.)

## psd_sdk

Used for `CreateDocument` (header + section offsets, PSD/PSB) and `imageUtil::DecompressRle`,
compiled with `PSD_ENABLE_LOGGING=0`. `ParseImageDataSection` isn't used only because it
decodes every channel (incl. alpha) into full planes. It is the `external/psd_sdk` submodule,
[pig1800/psd_sdk](https://github.com/pig1800/psd_sdk) — Molecular Matters' psd_sdk plus these fixes:

- PSB RLE row counts are 4 bytes; the composite (`ParseImageDataSection`) and layer channel
  (`ReadChannelDataRLE`) readers read 2, so PSB data decoded as garbage.
- `DecompressRle` bounds-checks source and destination, takes `size_t`, returns `bool`, and
  zero-fills on malformed/truncated data (it could overflow both buffers).
- Read errors are no longer swallowed: `SyncFileReader::Read` returns `bool`, keeps a sticky
  `HasFailed()` and splits >4 GB reads; Windows/Linux `NativeFile` report short reads (EOF);
  `MemoryFile` no longer reads past its buffer. The composite and layer readers check all of
  this and return null (freeing partial buffers) instead of decoding garbage; failed ZIP
  channels no longer return uninitialised planes.
- `width*height*bytes` arithmetic widened to 64-bit in the composite and layer paths.
- `PSD_ENABLE_LOGGING` / `PSD_ENABLE_ASSERTIONS` can be set from the compiler command line.

## Tests

```
scripts\vsenv.cmd cmake --build build/native --target psdpreview_tests
build\native\bin\psdpreview_tests simd                # kernels at every level vs. scalar + throughput
build\native\bin\psdpreview_tests sdk <psd/psb...>    # patched psd_sdk vs. our decoder; truncated/corrupt copies
build\native\bin\psdpreview_tests bench <psd/psb...>  # decode speed per SIMD level
build\native\bin\psdpreview_tests rle                 # unpack_row vs. psd_sdk's DecompressRle, 200000 random/malformed rows
```

For ARM64: `set PP_ARCH=arm64 && scripts\vsenv.cmd cmake --build build/native-arm64 --target psdpreview_tests`.

`tests\make_test_psd.py` writes minimal RGB/gray, 8/32-bit PSDs with an optional embedded ICC
profile (ImageMagick can't write 32-bit PSDs, and its grayscale + ICC PSDs are malformed).

## Measured (Ryzen 7 9800X3D, RTX 4070 Ti SUPER, NVMe)

200 real webtoon pages (1280 px wide, 26 GB of PSDs), warm file cache, two runs each:

| target width | CPU (IPP) | GPU (CUDA) | JPEG total | before (either backend) |
|---|---|---|---|---|
| 720 px | 4.8–5.0 s, peak RAM 4.8 GB | 3.2–3.3 s, peak RAM 2.7 GB | 252 MiB | 13.4–13.7 s, 226 MiB |
| 425 px | 3.2 s | 1.4 s | 113 MiB | 6.8 s, 102 MiB |
| 360 px | 2.9–3.0 s | 1.2 s | 87 MiB | 5.2–5.9 s |

`MaxCompression` at 720 px: 11.4 s (GPU) / 12.5 s (CPU), 226 MiB.

GPU backends, same 200 pages (a later session with other apps running and a colder file cache,
so every column is slower than above; the GPU runs at 425 px were bound by disk reads):

| target width | CUDA | D3D12 (RTX 4070 Ti SUPER) | CPU (IPP) | CPU (stb, SSE2 build) |
|---|---|---|---|---|
| 720 px | 3.87 s | 3.92 s | 5.54 s | 5.54 s |
| 425 px | 2.31 s | 2.37 s | 3.68 s | 3.68 s |

D3D12 kernel time per page (`PP_GPU_PROFILE`): RTX 4070 Ti SUPER unpack 1.6–1.8 ms, resize
1.3–1.6 ms; the 2-CU Radeon iGPU unpack 43–48 ms, resize 38–50 ms — memory-bound at its 18 GB/s,
hence not used (see [CPU/GPU split](#cpugpu-split)).

## Measured (Snapdragon X Elite X1E78100, Adreno X1-85)

The first 100 of the same pages (13 GB), warm file cache, second of two runs, 12 cores:

| target width | ARM64, auto (= CPU) | Adreno forced (`PP_GPU_BALANCE=0`) | the x64 build under emulation |
|---|---|---|---|
| 720 px | 2.59 s | 5.86 s | 4.70 s |
| 425 px | 1.87 s | 5.12 s | 2.87 s |

With the CPU decoding and the Adreno resizing (hybrid, zero copy both ways; 16.5 / 22 ms of
GPU per page at 425 / 720 px):

| target width | hybrid, balanced (`PP_GPU_BALANCE=1`) | hybrid, every page to the GPU | full GPU path | CPU |
|---|---|---|---|---|
| 720 px | 2.56 s | 2.99 s | 4.96 s | 2.58 s |
| 425 px | 2.28 s | 2.28 s | 4.01 s | 1.95 s |

The same on the 9800X3D's 2-CU Radeon, 200 pages (25.3 / 43.6 ms of GPU per page): hybrid
balanced 3.95 / 5.83 s, every page to the GPU 6.27 / 9.94 s, full GPU 19.7 / 21.4 s, CPU 4.09 /
5.90 s at 425 / 720 px.

The ARM64 CPU path does a page in about the same time as the 9800X3D's (per thread: decode
121 ms, stb Neon resize 86 ms, encode 96 ms per page at 425 px); emulated x64 (IPP at AVX2,
mozjpeg's AVX2) is 1.5–1.8× slower. On the Adreno the full GPU path's kernels take ~15 ms
(unpack) + 20–28 ms (resize, 30–37 before the fixed tap counts) per page. In isolation it
measures 54 GB/s copy, 41 G 4-byte loads/s and 1 TFLOPS; group size, wave size (64/128) and
staging rows in groupshared memory changed little. Results there are byte-identical
to x64 all the same (Adreno, WARP and the Neon CPU path).

On the GPU path the encoders are the bottleneck (~48 thread-s at 720 px, idle ~2 s) while the
extract threads mostly wait for them: reading + uploading takes ~4.5 thread-s and the GPU work
~4.5. On the CPU path decoding (~25 thread-s) and resizing (~30) compete with encoding for the
cores.
