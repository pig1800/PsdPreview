// Batch pipeline + exported C API.
//
// One fixed Slot per input file. Two thread groups, each fires as soon as its input is ready:
//   extract (N threads) decode the composite and resize it:
//     GPU: rows go straight into pinned band buffers of the thread's GpuStreamer; each band is
//          uploaded on the thread's own CUDA stream while the next one is decoded, NPP resizes
//          as soon as the last band is queued, the result is downloaded (push -> process -> pull)
//     CPU: decode into a reused per-thread buffer, resize with IPP
//   encode  (N threads) embedded profile -> sRGB (lcms2, on the small image), mozjpeg
// Extract hands slot indices to encode through an IndexQueue. A counting semaphore caps how
// many files are between "decode started" and "encoded", which bounds peak memory and makes
// pp_check_memory's estimate an upper bound rather than a guess.
#include <chrono>
#include <semaphore>

#include "color.h"
#include "cpu_resize.h"
#include "gpu.h"
#include "jpeg.h"
#include "psd.h"
#include "simd.h"

namespace pp {

struct Slot {
    std::wstring path;
    PsdHeader hdr;
    int dw = 0, dh = 0;  // output size; pixel buffers are 4 bytes/pixel (RGBA, or CMYK before conversion)
    Bytes resized, jpeg;
    const ColorTransform* color = nullptr;  // embedded profile -> sRGB, applied after the resize
    std::wstring profile;                   // embedded profile description ("" = none)
    int code = PP_FILE_OK;
    std::wstring message;

    void fail(int c, std::wstring m) {
        code = c;
        message = std::move(m);
        Bytes().swap(resized);
        Bytes().swap(jpeg);
    }
};

// Thread-seconds spent per stage (diagnostics: shows where a batch's time goes).
struct StageTime {
    std::atomic<int64_t> ns{0};
    double seconds() const { return double(ns.load()) / 1e9; }
};

class Timed {
public:
    explicit Timed(StageTime& t) : t_(t), t0_(std::chrono::steady_clock::now()) {}
    ~Timed() { t_.ns += (std::chrono::steady_clock::now() - t0_).count(); }

private:
    StageTime& t_;
    std::chrono::steady_clock::time_point t0_;
};

struct Stats {
    StageTime decode, resize, encode, extractWait, encodeIdle;
    std::atomic<int> gpuFallbacks{0};
};

}  // namespace pp

struct PP_Batch {
    std::vector<pp::Slot> slots;
    bool usedGpu = false;
    std::wstring backend;
    double elapsed = 0;
    double decode = 0, resize = 0, encode = 0, extractWait = 0, encodeIdle = 0;
    int gpuFallbacks = 0, extractThreads = 0, encodeThreads = 0;
};

namespace pp {
namespace {

struct Config {
    int extract, encode, inFlight, quality;
    bool gpu;
};

Config resolve(const PP_Options* o, int count) {
    const int n = logical_cores(), maxThreads = std::max(1, count);
    auto pick = [](int v, int def) { return v > 0 ? v : def; };
    Config c{};
    c.extract = std::clamp(pick(o ? o->extractThreads : 0, n), 1, maxThreads);
    c.encode = std::clamp(pick(o ? o->encodeThreads : 0, n), 1, maxThreads);
    c.inFlight = std::max(1, pick(o ? o->maxInFlight : 0, 2 * n));
    c.quality = (o && o->jpegQuality >= 1 && o->jpegQuality <= 100) ? o->jpegQuality : 82;
    c.gpu = gpu_ready() && !(o && o->forceCpu);
    return c;
}

// Recycles resized-image buffers, so the encoders' input isn't re-allocated (and its pages
// re-faulted) per image. Holds at most maxInFlight buffers.
class BufferPool {
public:
    Bytes take(size_t n) {
        Bytes b;
        {
            std::lock_guard lk(m_);
            // smallest buffer that fits; if none fits, the largest one (it grows once)
            size_t best = free_.size();
            for (size_t i = 0; i < free_.size(); ++i) {
                if (best == free_.size()) { best = i; continue; }
                const size_t ci = free_[i].capacity(), cb = free_[best].capacity();
                const bool fi = ci >= n, fb = cb >= n;
                if ((fi && (!fb || ci < cb)) || (!fi && !fb && ci > cb)) best = i;
            }
            if (best < free_.size()) {
                b = std::move(free_[best]);
                free_.erase(free_.begin() + ptrdiff_t(best));
            }
        }
        b.clear();  // a growing resize then doesn't copy old contents
        b.resize(n);
        return b;
    }
    void give(Bytes&& b) {
        std::lock_guard lk(m_);
        free_.push_back(std::move(b));
    }

private:
    std::mutex m_;
    std::vector<Bytes> free_;
};

FileStatus output_size(const PsdHeader& h, int targetWidth, int& dw, int& dh) {
    const int64_t height = std::max<int64_t>(1, std::llround(double(h.height) * targetWidth / h.width));
    dw = targetWidth;
    dh = int(std::min<int64_t>(height, INT32_MAX));
    if (height > PP_MAX_JPEG_DIMENSION)
        return {PP_FILE_OUTPUT_TOO_LARGE, std::format(L"{}x{} resized to width {} is {} px tall, over the JPEG limit of {}",
                                                      h.width, h.height, targetWidth, height, PP_MAX_JPEG_DIMENSION)};
    return {};
}

// Opens the file, reads the header and works out the output size.
FileStatus prepare_slot(Slot& s, int targetWidth) {
    if (s.path.empty()) return {PP_FILE_OPEN_FAILED, L"empty path"};
    FileReader f;
    std::wstring err;
    if (!f.open(s.path, err)) return {PP_FILE_OPEN_FAILED, err};
    FileStatus st = read_psd_header(f, s.hdr);
    if (!st.ok()) return st;
    return output_size(s.hdr, targetWidth, s.dw, s.dh);
}

void prepare_slots(std::vector<Slot>& slots, const wchar_t* const* paths, int targetWidth, int threads,
                   const std::function<void(Slot&, const FileStatus&)>& onFail) {
    parallel_for(int(slots.size()), threads, [&](int i) {
        Slot& s = slots[i];
        s.path = paths[i] ? paths[i] : L"";
        FileStatus st;
        try {
            st = prepare_slot(s, targetWidth);
        } catch (const std::bad_alloc&) {
            st = {PP_FILE_OUT_OF_MEMORY, L"out of memory"};
        }
        if (!st.ok()) {
            s.fail(st.code, st.message);
            onFail(s, st);
        }
    });
}

struct Plan {
    uint64_t host = 0, device = 0;
};

Plan plan_memory(const std::vector<Slot>& slots, const Config& c) {
    Plan p;
    std::vector<uint64_t> resized;
    uint64_t maxSrc = 0, maxScratch = 0, maxEncode = 0, results = 0, maxSrcDev = 0, maxDstDev = 0;
    for (const Slot& s : slots) {
        if (s.code) continue;
        resized.push_back(uint64_t(s.dw) * s.dh * kPixelBytes);
        maxSrc = std::max(maxSrc, s.hdr.decodedBytes());
        maxScratch = std::max(maxScratch, decode_scratch_bytes(s.hdr));
        maxEncode = std::max(maxEncode, jpeg_working_bytes(s.dw, s.dh));
        results += jpeg_estimated_bytes(s.dw, s.dh);
        maxSrcDev = std::max(maxSrcDev, gpu_pitch(uint64_t(s.hdr.width) * kPixelBytes) * s.hdr.height);
        maxDstDev = std::max(maxDstDev, gpu_pitch(uint64_t(s.dw) * kPixelBytes) * s.dh);
    }
    // Worst case: the largest `inFlight` resized images are all waiting for the encoders.
    std::sort(resized.begin(), resized.end(), std::greater<>());
    uint64_t inFlight = 0;
    for (size_t i = 0; i < resized.size() && i < size_t(c.inFlight); ++i) inFlight += resized[i];
    const uint64_t extractors = uint64_t(std::min<int>(c.extract, int(resized.size())));
    const uint64_t encoders = uint64_t(std::min<int>(c.encode, int(resized.size())));
    p.host = inFlight + extractors * maxScratch + encoders * maxEncode + results + (64ull << 20);
    if (c.gpu && extractors) {
        // decoded rows only pass through two pinned bands per thread; images live in VRAM
        p.host += extractors * 2 * kGpuBandBytes;
        p.device = extractors * (maxSrcDev + maxDstDev) + (64ull << 20);
    } else {
        p.host += extractors * maxSrc;  // one reused whole-image decode buffer per thread
    }
    return p;
}

std::unique_ptr<PP_Batch> run_batch(const wchar_t* const* paths, int count, int targetWidth, const PP_Options* opt,
                                    double* progressTarget) {
    const auto t0 = std::chrono::steady_clock::now();
    auto batch = std::make_unique<PP_Batch>();
    std::vector<Slot>& slots = batch->slots;
    slots.resize(size_t(count));
    const Config cfg = resolve(opt, count);
    Progress progress(progressTarget, int64_t(count) * 3);
    const CpuBackend& cpu = cpu_backend();  // IPP dispatch must be settled before worker threads start

    prepare_slots(slots, paths, targetWidth, cfg.extract, [&](Slot&, const FileStatus&) { progress.step(3); });

    Stats stats;
    ColorCache colors;
    BufferPool pool;
    std::counting_semaphore<> window(cfg.inFlight);
    IndexQueue toJpeg;
    std::atomic<int> next{0}, extractorsLeft{cfg.extract};
    std::mutex noteMutex;
    std::wstring gpuNote;  // first GPU problem, for the backend description

    auto fail = [&](Slot& s, int code, std::wstring msg, int stepsLeft) {
        s.fail(code, std::move(msg));
        progress.step(stepsLeft);
        window.release();
    };
    auto note = [&](const std::wstring& what) {
        std::lock_guard lk(noteMutex);
        if (gpuNote.empty()) gpuNote = what;
    };

    auto extractWorker = [&] {
        std::unique_ptr<GpuStreamer> gs;
        if (cfg.gpu) {
            gs = std::make_unique<GpuStreamer>();
            std::wstring err;
            if (!gs->init(err)) {
                gs.reset();
                note(err);
            }
        }
        Bytes decodeBuf;  // CPU path: whole decoded image, capacity reused across this thread's files

        for (int i; (i = next.fetch_add(1)) < count;) {
            Slot& s = slots[i];
            if (s.code) continue;
            {
                Timed t(stats.extractWait);
                window.acquire();
            }
            bool extracted = false;  // has the extract progress step been counted?
            try {
                FileReader f;
                std::wstring err;
                if (!f.open(s.path, err)) { fail(s, PP_FILE_OPEN_FAILED, err, 3); continue; }
                // 32-bit documents are linear light and already sRGB-encoded by the decoder.
                std::vector<uint8_t> icc;
                if (s.hdr.depth != 32 && read_icc_profile(f, s.hdr, icc)) s.color = colors.get(icc, s.hdr.colorMode, s.profile);
                s.resized = pool.take(size_t(s.dw) * s.dh * kPixelBytes);

                bool resized = false;
                if (gs) {
                    std::wstring gerr;
                    if (gs->begin(s.hdr.width, s.hdr.height, gerr)) {
                        FileStatus st;
                        {
                            Timed t(stats.decode);
                            st = decode_composite(f, s.hdr, *gs);  // bands stream to the GPU while decoding
                        }
                        if (!st.ok()) {
                            gs->abort();
                            fail(s, st.code, st.message, 3);
                            continue;
                        }
                        progress.step();
                        extracted = true;
                        Timed t(stats.resize);
                        resized = gs->finish(s.dw, s.dh, s.resized.data(), gerr);
                    }
                    if (!resized) {  // e.g. out of GPU memory: redo this one on the CPU
                        ++stats.gpuFallbacks;
                        note(gerr);
                    }
                }
                if (!resized) {
                    FileStatus st;
                    {
                        Timed t(stats.decode);
                        st = decode_composite(f, s.hdr, decodeBuf);
                    }
                    if (!st.ok()) { fail(s, st.code, st.message, extracted ? 2 : 3); continue; }
                    if (!extracted) {
                        progress.step();
                        extracted = true;
                    }
                    Timed t(stats.resize);
                    if (!cpu_resize(decodeBuf.data(), int(s.hdr.width), int(s.hdr.height), s.resized.data(), s.dw, s.dh, err)) {
                        fail(s, PP_FILE_RESIZE_FAILED, L"IPP: " + err, 2);
                        continue;
                    }
                }
            } catch (const std::bad_alloc&) {
                fail(s, PP_FILE_OUT_OF_MEMORY, L"out of memory while decoding/resizing", extracted ? 2 : 3);
                continue;
            } catch (const std::exception& e) {
                fail(s, PP_FILE_CORRUPT, L"decode: " + widen(e.what()), extracted ? 2 : 3);
                continue;
            }
            progress.step();
            toJpeg.push(i);
        }
        if (--extractorsLeft == 0) toJpeg.close();
    };

    auto encodeWorker = [&] {
        for (int i;;) {
            bool got;
            {
                Timed t(stats.encodeIdle);
                got = toJpeg.pop(i);
            }
            if (!got) break;
            Slot& s = slots[i];
            std::wstring err;
            bool ok = false;
            int code = PP_FILE_ENCODE_FAILED;
            try {
                Timed t(stats.encode);
                to_srgb(s.color, s.hdr.colorMode, s.resized.data(), size_t(s.dw) * s.dh);
                ok = encode_jpeg(s.resized.data(), s.dw, s.dh, cfg.quality, s.jpeg, err);
            } catch (const std::bad_alloc&) {
                code = PP_FILE_OUT_OF_MEMORY;
                err = L"out of memory while encoding";
            }
            pool.give(std::move(s.resized));
            s.resized = Bytes();
            if (!ok) { fail(s, code, err, 1); continue; }
            progress.step();
            window.release();
        }
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < cfg.extract; ++t) threads.emplace_back(extractWorker);
    for (int t = 0; t < cfg.encode; ++t) threads.emplace_back(encodeWorker);
    for (auto& t : threads) t.join();
    if (cfg.gpu) gpu_trim();
    progress.finish();

    batch->usedGpu = cfg.gpu;
    batch->backend = cfg.gpu ? std::format(L"CUDA NPP bicubic on {}, streamed ({} streams, 2x{} MB pinned bands each)",
                                           gpu_description(), cfg.extract, kGpuBandBytes >> 20)
                             : std::format(L"Intel IPP {} bicubic, {} dispatch{}", cpu.version, cpu.target,
                                           cpu.avx512Active ? L" (AVX-512)" : L"");
    batch->backend += std::format(L"; {} extract / {} encode threads; mozjpeg q{}", cfg.extract, cfg.encode, cfg.quality);
    if (!gpuNote.empty())
        batch->backend += std::format(L"; GPU problem ({} image(s) redone on the CPU): {}", stats.gpuFallbacks.load(), gpuNote);
    batch->elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    batch->decode = stats.decode.seconds();
    batch->resize = stats.resize.seconds();
    batch->encode = stats.encode.seconds();
    batch->extractWait = stats.extractWait.seconds();
    batch->encodeIdle = stats.encodeIdle.seconds();
    batch->gpuFallbacks = stats.gpuFallbacks;
    batch->extractThreads = cfg.extract;
    batch->encodeThreads = cfg.encode;
    return batch;
}

FileStatus validate_one(Slot& s, int targetWidth) {
    if (s.path.empty()) return {PP_FILE_OPEN_FAILED, L"empty path"};
    FileReader f;
    std::wstring err;
    if (!f.open(s.path, err)) return {PP_FILE_OPEN_FAILED, err};
    FileStatus st = read_psd_header(f, s.hdr);
    if (st.ok()) st = check_composite(f, s.hdr);
    if (st.ok() && targetWidth > 0) st = output_size(s.hdr, targetWidth, s.dw, s.dh);
    return st;
}

}  // namespace
}  // namespace pp

using namespace pp;

#define PP_GUARD_BEGIN try {
#define PP_GUARD_END                                     \
    }                                                    \
    catch (const std::bad_alloc&) { return PP_ERR_OUT_OF_MEMORY; } \
    catch (...) { return PP_ERR_INTERNAL; }

extern "C" {

PP_API int32_t pp_cuda_init(int32_t deviceId, PP_DeviceInfo* info) {
    PP_GUARD_BEGIN
    return gpu_init(deviceId, info);
    PP_GUARD_END
}

PP_API int32_t pp_cuda_ready(void) { return gpu_ready() ? 1 : 0; }

PP_API void pp_cuda_shutdown(void) { gpu_shutdown(); }

PP_API int32_t pp_cpu_info(PP_CpuInfo* info) {
    if (!info) return PP_ERR_INVALID_ARGUMENT;
    PP_GUARD_BEGIN
    const CpuBackend& b = cpu_backend();
    *info = {};
    info->logicalCores = logical_cores();
    info->avx512Supported = b.avx512Supported;
    info->avx512Active = b.avx512Active;
    info->forcedDispatch = b.forcedDispatch;
    copy_str(info->ippTarget, b.target);
    copy_str(info->ippName, b.name);
    copy_str(info->ippVersion, b.version);
    copy_str(info->simdLevel, simd::kernels().name);
    return PP_OK;
    PP_GUARD_END
}

PP_API int32_t pp_validate(const wchar_t* const* paths, int32_t count, int32_t targetWidth, PP_Issue* issues,
                           int32_t* issueCount) {
    if (!issueCount || count < 0 || (count > 0 && (!paths || !issues)) || targetWidth < 0) return PP_ERR_INVALID_ARGUMENT;
    PP_GUARD_BEGIN
    *issueCount = 0;
    std::vector<Slot> slots(static_cast<size_t>(count));
    std::vector<FileStatus> status(static_cast<size_t>(count));
    parallel_for(count, logical_cores(), [&](int i) {
        slots[i].path = paths[i] ? paths[i] : L"";
        try {
            status[i] = validate_one(slots[i], targetWidth);
        } catch (const std::bad_alloc&) {
            status[i] = {PP_FILE_OUT_OF_MEMORY, L"out of memory"};
        }
    });
    int k = 0;
    for (int i = 0; i < count; ++i) {
        if (status[i].ok()) continue;
        issues[k].index = i;
        issues[k].code = status[i].code;
        copy_str(issues[k].message, status[i].message);
        ++k;
    }
    *issueCount = k;
    return PP_OK;
    PP_GUARD_END
}

PP_API int32_t pp_check_memory(const wchar_t* const* paths, int32_t count, int32_t targetWidth,
                               const PP_Options* options, PP_MemoryReport* report) {
    if (!report || count < 0 || (count > 0 && !paths) || targetWidth <= 0) return PP_ERR_INVALID_ARGUMENT;
    PP_GUARD_BEGIN
    *report = {};
    std::vector<Slot> slots(static_cast<size_t>(count));
    std::atomic<int> unreadable{0};
    const Config cfg = resolve(options, count);
    prepare_slots(slots, paths, targetWidth, logical_cores(), [&](Slot&, const FileStatus& st) {
        if (st.code != PP_FILE_OUTPUT_TOO_LARGE) ++unreadable;
    });
    const Plan p = plan_memory(slots, cfg);

    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    GlobalMemoryStatusEx(&ms);
    report->hostRequired = p.host;
    report->hostAvailable = ms.ullAvailPhys;
    report->hostOk = p.host <= ms.ullAvailPhys;
    report->usesGpu = cfg.gpu;
    report->deviceRequired = p.device;
    uint64_t freeB = 0, totalB = 0;
    if (cfg.gpu && gpu_memory(freeB, totalB)) report->deviceAvailable = freeB;
    report->deviceOk = !cfg.gpu || p.device <= report->deviceAvailable;
    report->unreadableFiles = unreadable;
    return PP_OK;
    PP_GUARD_END
}

PP_API int32_t pp_process(const wchar_t* const* paths, int32_t count, int32_t targetWidth, const PP_Options* options,
                          double* progress, PP_Batch** batch) {
    if (!batch || count < 0 || (count > 0 && !paths) || targetWidth <= 0 || targetWidth > PP_MAX_JPEG_DIMENSION)
        return PP_ERR_INVALID_ARGUMENT;
    *batch = nullptr;
    PP_GUARD_BEGIN
    *batch = run_batch(paths, count, targetWidth, options, progress).release();
    return PP_OK;
    PP_GUARD_END
}

PP_API int32_t pp_batch_info(const PP_Batch* batch, PP_BatchInfo* info) {
    if (!batch || !info) return PP_ERR_INVALID_ARGUMENT;
    *info = {};
    info->count = int32_t(batch->slots.size());
    for (const Slot& s : batch->slots) info->succeeded += s.code == PP_FILE_OK;
    info->usedGpu = batch->usedGpu;
    info->elapsedSeconds = batch->elapsed;
    info->decodeSeconds = batch->decode;
    info->resizeSeconds = batch->resize;
    info->encodeSeconds = batch->encode;
    info->extractWaitSeconds = batch->extractWait;
    info->encodeIdleSeconds = batch->encodeIdle;
    info->gpuFallbacks = batch->gpuFallbacks;
    info->extractThreads = batch->extractThreads;
    info->encodeThreads = batch->encodeThreads;
    copy_str(info->backend, batch->backend);
    return PP_OK;
}

PP_API int32_t pp_batch_item(const PP_Batch* batch, int32_t index, PP_Item* item) {
    if (!batch || !item || index < 0 || size_t(index) >= batch->slots.size()) return PP_ERR_INVALID_ARGUMENT;
    const Slot& s = batch->slots[size_t(index)];
    *item = {};
    item->sourcePath = s.path.c_str();
    item->data = s.jpeg.empty() ? nullptr : s.jpeg.data();
    item->size = s.jpeg.size();
    item->width = s.dw;
    item->height = s.dh;
    item->code = s.code;
    item->colorConverted = s.color != nullptr;
    copy_str(item->message, s.message);
    copy_str(item->colorProfile, s.profile);
    return PP_OK;
}

PP_API void pp_batch_release_item(PP_Batch* batch, int32_t index) {
    if (batch && index >= 0 && size_t(index) < batch->slots.size()) Bytes().swap(batch->slots[size_t(index)].jpeg);
}

PP_API void pp_batch_free(PP_Batch* batch) { delete batch; }

}  // extern "C"
