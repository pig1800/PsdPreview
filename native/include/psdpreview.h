// psdpreview — extract the flattened composite of PSD/PSB files, resize to a target
// width (antialiased bicubic matching Photoshop's "Bicubic Sharper"; on the GPU with CUDA or
// D3D12 compute, else on the CPU with Intel IPP on x64 / stb_image_resize2 on ARM64), and
// encode as JPEG with mozjpeg.
//
// Plain C ABI so it can be P/Invoked from C#. All strings are UTF-16 (wchar_t).
// Every struct is blittable; fixed-size message buffers avoid ownership questions.
#pragma once
#include <stdint.h>
#include <wchar.h>

#ifdef PSDPREVIEW_BUILD
#define PP_API __declspec(dllexport)
#else
#define PP_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define PP_MESSAGE_CAP 260
#define PP_MAX_JPEG_DIMENSION 65500

/* Return codes of the API functions. */
typedef enum PP_Status {
    PP_OK = 0,
    PP_ERR_INVALID_ARGUMENT = 1,
    PP_ERR_GPU_UNAVAILABLE = 2,  /* no driver / no such device */
    PP_ERR_GPU_FAILED = 3,       /* CUDA / D3D12 error during initialisation */
    PP_ERR_KERNEL_UNAVAILABLE = 4, /* the GPU kernels failed to load (driver too old?) or their self-test failed */
    PP_ERR_OUT_OF_MEMORY = 5,
    PP_ERR_INTERNAL = 6,
} PP_Status;

/* Per-file problem codes (validation and processing). */
typedef enum PP_FileCode {
    PP_FILE_OK = 0,
    PP_FILE_OPEN_FAILED = 1,
    PP_FILE_NOT_PSD = 2,
    PP_FILE_UNSUPPORTED_VERSION = 3,
    PP_FILE_UNSUPPORTED_COLOR_MODE = 4,
    PP_FILE_UNSUPPORTED_DEPTH = 5,
    PP_FILE_UNSUPPORTED_COMPRESSION = 6,
    PP_FILE_NO_COMPOSITE = 7,     /* saved with "Maximize Compatibility" off */
    PP_FILE_BLANK_COMPOSITE = 8,  /* non-Photoshop writer stored an empty composite */
    PP_FILE_TRUNCATED = 9,
    PP_FILE_CORRUPT = 10,
    PP_FILE_BAD_DIMENSIONS = 11,
    PP_FILE_OUTPUT_TOO_LARGE = 12, /* resized height would exceed the JPEG limit */
    PP_FILE_OUT_OF_MEMORY = 13,
    PP_FILE_RESIZE_FAILED = 14,
    PP_FILE_ENCODE_FAILED = 15,
} PP_FileCode;

/* GPU APIs. */
typedef enum PP_GpuBackend {
    PP_GPU_AUTO = 0,       /* CUDA if it works (NVIDIA, x64), else D3D12 on the high-performance adapter */
    PP_GPU_CUDA = 1,       /* x64 builds only */
    PP_GPU_D3D12 = 2,      /* any GPU with a D3D12 driver (shader model 6.0) */
    PP_GPU_D3D12_WARP = 3, /* D3D12 on WARP, Windows' software rasterizer (testing) */
} PP_GpuBackend;

typedef struct PP_DeviceInfo {
    uint64_t totalMemory;     /* VRAM; D3D12: the OS budget for the GPU (system memory on UMA) */
    uint64_t freeMemory;
    int32_t backend;          /* PP_GpuBackend actually initialised */
    int32_t adapter;          /* CUDA device / D3D12 adapter index */
    int32_t computeMajor;     /* CUDA compute capability, or D3D12 shader model (6.x) */
    int32_t computeMinor;
    int32_t multiProcessors;  /* CUDA SMs; 0 on D3D12 */
    int32_t waveLanesMin;     /* SIMD width: CUDA warp size, D3D12 wave lane count range */
    int32_t waveLanesMax;
    int32_t uma;              /* integrated GPU sharing system memory: uploads are zero-copy */
    int32_t cacheCoherentUma; /* 0 on UMA: CPU-written GPU memory is write-combined (uncached) */
    int32_t directResult;     /* D3D12: the result is written straight into CPU-readable memory */
    int32_t hybrid;           /* D3D12 on UMA: the CPU decodes into GPU memory, the GPU resizes;
                                 0: the GPU unpacks and resizes (discrete GPUs, CUDA) */
    int32_t cudaRuntimeVersion; /* e.g. 13000; 0 on D3D12 */
    int32_t cudaDriverVersion;
    wchar_t name[256];
    wchar_t message[PP_MESSAGE_CAP]; /* failure reason when init fails (auto: why CUDA was skipped) */
} PP_DeviceInfo;

typedef struct PP_CpuInfo {
    int32_t logicalCores;
    int32_t avx512Supported; /* CPU + OS support AVX-512 F/CD/BW/DQ/VL */
    int32_t avx512Active;    /* IPP is dispatched to its AVX-512 code path */
    int32_t forcedDispatch;  /* we had to override IPP's default dispatch */
    wchar_t resizeTarget[8]; /* IPP dispatch: "k0" = AVX-512, "l9" = AVX2...; stb_image_resize2: "NEON"/"SSE2" */
    wchar_t resizeName[128]; /* CPU resizer: IPP library name or "stb_image_resize2" */
    wchar_t resizeVersion[64];
    wchar_t simdLevel[16];   /* kernels used for plane merge / bit-depth reduction: "AVX-512", "AVX2", "SSE2", "NEON" */
    int32_t arm64;           /* ARM64 build */
} PP_CpuInfo;

/* Optional tuning; pass NULL for defaults. Zero/negative fields mean "default". */
typedef struct PP_Options {
    int32_t jpegQuality;    /* default 82 */
    int32_t forceCpu;       /* non-zero: use the CPU even if a GPU is initialised */
    int32_t extractThreads; /* default: logical cores (they share up to that many GPU streamers) */
    int32_t encodeThreads;  /* default: logical cores */
    int32_t maxInFlight;    /* files decoded but not yet encoded; default 2 x cores */
    int32_t maxCompression; /* non-zero: mozjpeg max compression (trellis, scan search): ~11% smaller
                               files, ~5x slower encoding (the pipeline's bottleneck) */
} PP_Options;

typedef struct PP_Issue {
    int32_t index; /* index into the input list */
    int32_t code;  /* PP_FileCode */
    wchar_t message[PP_MESSAGE_CAP];
} PP_Issue;

typedef struct PP_MemoryReport {
    uint64_t hostRequired;
    uint64_t hostAvailable;
    uint64_t deviceRequired; /* 0 when the CPU path will be used */
    uint64_t deviceAvailable;
    int32_t usesGpu;
    int32_t hostOk;
    int32_t deviceOk;
    int32_t unreadableFiles; /* files whose header could not be read (not counted) */
} PP_MemoryReport;

/* Timings are thread-seconds summed over a stage's threads (diagnostics). */
typedef struct PP_BatchInfo {
    double elapsedSeconds;
    double decodeSeconds;      /* composite decode (GPU path: reading + uploading the still compressed data) */
    double resizeSeconds;      /* CPU resize, or GPU unpack + resize + download */
    double encodeSeconds;      /* colour conversion + JPEG encoding */
    double extractWaitSeconds; /* extract threads blocked by maxInFlight: encoding is the bottleneck */
    double encodeIdleSeconds;  /* encode threads waiting for work: decoding/resizing is the bottleneck */
    int32_t count;
    int32_t succeeded;
    int32_t usedGpu;
    int32_t gpuFallbacks;      /* images redone on the CPU after a GPU problem (e.g. out of VRAM) */
    int32_t extractThreads;
    int32_t encodeThreads;
    int32_t gpuStreamers;      /* GPU streamers the extract threads shared (fewer if GPU memory is short) */
    wchar_t backend[PP_MESSAGE_CAP]; /* human-readable description of what ran */
} PP_BatchInfo;

typedef struct PP_Item {
    const wchar_t* sourcePath; /* input path this item came from (owned by the batch) */
    const uint8_t* data;       /* JPEG bytes, NULL on failure or after release */
    uint64_t size;
    int32_t width;
    int32_t height;
    int32_t code;           /* PP_FileCode */
    int32_t colorConverted; /* 1 if the embedded profile was converted to sRGB with lcms2 */
    wchar_t message[PP_MESSAGE_CAP];
    wchar_t colorProfile[128]; /* embedded ICC profile description, "" if none */
} PP_Item;

typedef struct PP_Batch PP_Batch;

/* ---- GPU ----------------------------------------------------------------
 * Call once at program start. backend: PP_GpuBackend; adapter: CUDA device, or D3D12
 * adapter index in DXGI's high-performance order (0 = the fastest GPU). Creates the
 * device, loads the kernels (JIT-compiled by the driver) and runs a self-test. Safe to
 * call again: a no-op for the same backend + adapter, else replaces the GPU in use
 * (don't call while a batch runs).
 * If it fails, processing silently uses the CPU path. */
PP_API int32_t pp_gpu_init(int32_t backend, int32_t adapter, PP_DeviceInfo* info);
PP_API int32_t pp_gpu_ready(void);
PP_API void pp_gpu_shutdown(void);

/* ---- CPU backend (resizer and SIMD info; initialises IPP on first call) -- */
PP_API int32_t pp_cpu_info(PP_CpuInfo* info);

/* ---- Validation ----------------------------------------------------------
 * Checks every file can be processed. `issues` must have room for `count`
 * entries; *issueCount receives how many were written. targetWidth may be 0
 * (skip the output-size check). */
PP_API int32_t pp_validate(const wchar_t* const* paths, int32_t count, int32_t targetWidth,
                           PP_Issue* issues, int32_t* issueCount);

/* ---- Memory check ------------------------------------------------------- */
PP_API int32_t pp_check_memory(const wchar_t* const* paths, int32_t count, int32_t targetWidth,
                               const PP_Options* options, PP_MemoryReport* report);

/* ---- Processing ----------------------------------------------------------
 * Blocking. `progress` (optional) is updated atomically from 0 to 1; steps are
 * count x 3 (extract, resize, encode). Returns a batch handle even when some
 * files fail; inspect each item's code. */
PP_API int32_t pp_process(const wchar_t* const* paths, int32_t count, int32_t targetWidth,
                          const PP_Options* options, double* progress, PP_Batch** batch);
PP_API int32_t pp_batch_info(const PP_Batch* batch, PP_BatchInfo* info);
PP_API int32_t pp_batch_item(const PP_Batch* batch, int32_t index, PP_Item* item);
PP_API void pp_batch_release_item(PP_Batch* batch, int32_t index); /* frees one JPEG */
PP_API void pp_batch_free(PP_Batch* batch);

#ifdef __cplusplus
}
#endif
