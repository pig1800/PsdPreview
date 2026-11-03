// psdpreview — extract the flattened composite of PSD/PSB files, resize to a target
// width (antialiased bicubic matching Photoshop's "Bicubic Sharper", CUDA or Intel IPP),
// and encode as JPEG with mozjpeg.
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
    PP_ERR_CUDA_UNAVAILABLE = 2, /* no driver / no CUDA device */
    PP_ERR_CUDA_FAILED = 3,      /* CUDA runtime error during initialisation */
    PP_ERR_KERNEL_UNAVAILABLE = 4, /* the resize kernel failed to load (driver too old?) or its self-test failed */
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

typedef struct PP_DeviceInfo {
    uint64_t totalMemory;
    uint64_t freeMemory;
    int32_t deviceId;
    int32_t computeMajor;
    int32_t computeMinor;
    int32_t multiProcessors;
    int32_t cudaRuntimeVersion; /* e.g. 13000 */
    int32_t cudaDriverVersion;
    wchar_t name[256];
    wchar_t message[PP_MESSAGE_CAP]; /* failure reason when init fails */
} PP_DeviceInfo;

typedef struct PP_CpuInfo {
    int32_t logicalCores;
    int32_t avx512Supported; /* CPU + OS support AVX-512 F/CD/BW/DQ/VL */
    int32_t avx512Active;    /* IPP is dispatched to its AVX-512 code path */
    int32_t forcedDispatch;  /* we had to override IPP's default dispatch */
    wchar_t ippTarget[8];    /* "k0" = AVX-512, "l9" = AVX2, ... */
    wchar_t ippName[128];
    wchar_t ippVersion[64];
    wchar_t simdLevel[16];   /* kernels used for plane merge / bit-depth reduction: "AVX-512", "AVX2", "SSE2" */
} PP_CpuInfo;

/* Optional tuning; pass NULL for defaults. Zero/negative fields mean "default". */
typedef struct PP_Options {
    int32_t jpegQuality;    /* default 82 */
    int32_t forceCpu;       /* non-zero: use IPP even if CUDA is initialised */
    int32_t extractThreads; /* default: logical cores (on the GPU path: one CUDA stream each) */
    int32_t encodeThreads;  /* default: logical cores */
    int32_t maxInFlight;    /* files decoded but not yet encoded; default 2 x cores */
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
    double decodeSeconds;      /* composite decode (GPU path: incl. streaming the bands up) */
    double resizeSeconds;      /* IPP resize, or GPU tail upload + resize + download */
    double encodeSeconds;      /* colour conversion + JPEG encoding */
    double extractWaitSeconds; /* extract threads blocked by maxInFlight: encoding is the bottleneck */
    double encodeIdleSeconds;  /* encode threads waiting for work: decoding/resizing is the bottleneck */
    int32_t count;
    int32_t succeeded;
    int32_t usedGpu;
    int32_t gpuFallbacks;      /* images redone on the CPU after a GPU problem (e.g. out of VRAM) */
    int32_t extractThreads;
    int32_t encodeThreads;
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

/* ---- CUDA ---------------------------------------------------------------
 * Call once at program start. Creates the CUDA context, loads the resize kernel
 * (JIT-compiled by the driver) and runs a tiny resize to check it. Safe to call
 * again (no-op if initialised).
 * If it fails, processing silently uses the CPU (IPP) path. */
PP_API int32_t pp_cuda_init(int32_t deviceId, PP_DeviceInfo* info);
PP_API int32_t pp_cuda_ready(void);
PP_API void pp_cuda_shutdown(void);

/* ---- CPU backend (IPP dispatch info; initialises IPP on first call) ------ */
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
