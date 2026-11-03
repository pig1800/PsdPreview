# PsdPreview

Extracts the flattened full-resolution composite stored in PSD/PSB files, resizes it to a
target width (bicubic — CUDA NPP when available, Intel IPP with AVX-512 otherwise), converts
embedded colour profiles to sRGB (Little CMS 2), and encodes JPEG with mozjpeg (q82).
Native C++ DLL with a C# (.NET 8) wrapper.

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

Requirements: Git, VS 2026 C++ tools, CMake ≥ 3.24, NASM, CUDA Toolkit 13 (headers/libs only —
no nvcc needed), .NET 8 SDK.

## Deployment

`psdpreview_native.dll` is standalone: IPP, mozjpeg, lcms2, psd_sdk and cudart are linked
statically against the static CRT (`/MT`), so it imports only Windows system DLLs (`KERNEL32`,
the `api-ms-win-core-synch` API set) — no VC++ redistributable. The only other files are the
two CUDA NPP DLLs, `nppc64_13.dll` and `nppig64_13.dll`: NVIDIA ships static NPP
(`libnpp*_static`) for Linux only; on Windows the toolkit has just DLL import stubs. They are delay-loaded, so without them, or without an NVIDIA driver/GPU, everything
still works on the CPU path — drop them for a CPU-only distribution.

Referencing `managed\PsdPreview\PsdPreview.csproj` copies `psdpreview_native.dll` and the NPP
DLLs to the app's output. For a **single-file** app:

```
dotnet publish <YourApp> -c Release -r win-x64 --self-contained ^
    -p:PublishSingleFile=true -p:IncludeNativeLibrariesForSelfExtract=true ^
    -p:EnableCompressionInSingleFile=true
```

The native DLLs are bundled into the exe and extracted on first run to
`%TEMP%\.net\<app>\<hash>\` (override with `DOTNET_BUNDLE_EXTRACT_BASE_DIR`); the library finds
NPP next to itself there. The demo published this way is one 60 MB exe (95 MB uncompressed,
~25 MB of it NPP) that runs from an otherwise empty folder on both the GPU and CPU paths.

## C# usage

```csharp
using PsdPreview;

// once, at program start (failure is fine: processing then uses IPP on the CPU)
CudaInitResult cuda = PsdPreviewer.InitializeCuda();

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

`PsdPreviewOptions` (all optional): `JpegQuality` (82), `ForceCpu`, `ExtractThreads` (default:
logical cores; on the GPU path one CUDA stream each), `EncodeThreads` (default: logical cores),
`MaxInFlight` (default 2 × cores). `batch.Stats` reports thread-seconds per stage (decode, resize,
encode, extract threads waiting for encoders, encoders idle) and GPU fallbacks.
`PsdPreviewer.GetCpuBackend()` reports the SIMD level and IPP dispatch.

`samples\PsdPreview.Demo` is a runnable end-to-end example:
`PsdPreview.Demo <dir-or-files> [--width 720] [--cpu] [--out DIR] [--limit N] [--validate-only] [--skip-validate]`.

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

1. **Decode** (each extract thread): per band of rows (~4 MB of compressed data), read each
   colour channel's bytes, PackBits-decode row by row (psd_sdk's `DecompressRle`), reduce >8-bit
   samples, and merge the planes into 4-byte pixels with SIMD: RGBA8888 (A = 255; gray expanded)
   or CMYK. Also reads the ICC profile and looks up its transform.
2. **Resize** to the target width (height keeps aspect ratio), bicubic, 4 channels:
   - GPU — streamed push/process/pull per extract thread: each thread has its own non-blocking
     CUDA stream and two 4 MB pinned band buffers. The decoder writes rows straight into the
     current band; a full band is uploaded asynchronously (`cudaMemcpy2DAsync` + event) while the
     next band is decoded into the other buffer. As soon as the last band is queued,
     `nppiResize_8u_C4R` runs on the same stream and the result is downloaded into the encoder's
     buffer. The full-size image only ever exists in VRAM, allocated from a private
     stream-ordered pool (`cudaMallocFromPoolAsync`), so the 16 streams' uploads and kernels
     overlap with no host staging copies and no dispatcher thread. An image the GPU can't take
     (e.g. VRAM exhausted) is redone on the CPU path.
   - CPU: decode into a per-thread buffer that is reused across files, then
     `ippiResizeCubic_8u_C4R` (Catmull-Rom).
3. **Encode** (N threads): ICC → sRGB if needed, then mozjpeg reads the buffer as
   `JCS_EXT_RGBX` with its default max-compression profile (progressive, trellis, optimised
   Huffman, 4:2:0) at q82. Resized buffers are recycled through a small pool.

A counting semaphore caps files "between decode start and encode end" (`MaxInFlight`), which
bounds peak memory; `CheckMemory` computes the worst case from the actual headers using the
same configuration.

Progress counts 3 steps per file (extract, resize, encode); failed files have their remaining
steps added immediately, so progress always reaches 1.

NPP and IPP outputs are not pixel-identical (NPP maps pixel coordinates slightly differently)
but are visually the same.

### SIMD (plane merge, 16 → 8-bit)

One translation unit per instruction set (`simd_sse2.cpp`, `simd_avx2.cpp` with `/arch:AVX2`,
`simd_avx512.cpp` with `/arch:AVX512`), dispatched at runtime via CPUID + XGETBV: AVX-512 (F+BW)
→ AVX2 → SSE2. AVX1 has no 256-bit integer unpacks/shuffles, so AVX1-only CPUs use SSE2. The
merge is the unpack8 → unpack16 scheme of PSDBatchExporter's `copyPanarPixels*`, with the AVX2
lane fix-up by `permute2x128` and, for AVX-512, a 4×4 transpose of 128-bit lanes
(`shuffle_i64x2`); narrowing uses `and` + `packus` (SSE2/AVX2) or `vpmovwb` (AVX-512). The
kernel header only declares functions, so no AVX-compiled inline code can leak into baseline
paths. `PP_SIMD=sse2|avx2|avx512` lowers the level (testing).

Single-row (cache-resident, 1500 px) throughput on the 9800X3D:

| | merge_rgb | narrow16 |
|---|---|---|
| scalar | 2.6 Gpx/s | – |
| SSE2 | 20.7 Gpx/s | 34 Gsamples/s |
| AVX2 | 33.0 Gpx/s | 80 Gsamples/s |
| AVX-512 | 39.7 Gpx/s | 95 Gsamples/s |

Whole-file decode runs at ~750 Mpx/s per thread on every level — PackBits decoding dominates.

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
```

`tests\make_test_psd.py` writes minimal RGB/gray, 8/32-bit PSDs with an optional embedded ICC
profile (ImageMagick can't write 32-bit PSDs, and its grayscale + ICC PSDs are malformed).

## Measured (Ryzen 7 9800X3D, RTX 4070 Ti SUPER, NVMe)

200 real webtoon pages (1280 px wide, 26 GB of PSDs), warm file cache, two runs each:

| target width | CPU (IPP) | GPU (NPP, streamed) |
|---|---|---|
| 720 px | 13.3–13.4 s, peak RAM 4.9 GB | 12.9–14.0 s, peak RAM 2.8 GB |
| 360 px | 6.1–6.3 s | 5.8 s |

Both paths are bound by the mozjpeg max-compression encode: the 16 encoders are busy for the
whole run (~200 thread-s at 720 px) while extract threads mostly wait for them, and resizing
is only ~9 thread-s on the CPU (~1 on the GPU). The GPU path's gain is therefore the resize time
plus less memory traffic and ~40% less RAM; going substantially faster would need cheaper JPEG
settings (e.g. without trellis quantisation / scan optimisation).
