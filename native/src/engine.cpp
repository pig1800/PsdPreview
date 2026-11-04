// Batch pipeline + exported C API.
//
// One fixed Slot per input file. Two thread groups, each fires as soon as its input is ready:
//   extract (N threads) decode the composite and resize it:
//     GPU: a GpuStreamer from a shared pool (CUDA or D3D12, see gpu.h) takes the composite's
//          still PackBits-compressed colour channels from the file to the GPU, where our kernels
//          unpack the rows and resize, and brings the result back (push -> process -> pull). The
//          pool holds as many streamers as GPU memory allows, up to N. Formats the kernels don't
//          take (32-bit, raw 16-bit) go the CPU way.
//     CPU: decode into a reused per-thread buffer, resize with IPP (x64) or stb_image_resize2
//          (ARM64); same filter as the GPU (resize_filter.h)
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
    std::unique_ptr<GpuImage> image;  // hybrid path: the resized image where the GPU wrote it (instead of `resized`)
    const ColorTransform* color = nullptr;  // embedded profile -> sRGB, applied after the resize
    std::wstring profile;                   // embedded profile description ("" = none)
    int code = PP_FILE_OK;
    std::wstring message;

    void fail(int c, std::wstring m) {
        code = c;
        message = std::move(m);
        Bytes().swap(resized);
        Bytes().swap(jpeg);
        image.reset();
    }
    uint8_t* pixels() { return image ? image->data : resized.data(); }
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
    int gpuFallbacks = 0, extractThreads = 0, encodeThreads = 0, gpuStreamers = 0;
};

namespace pp {
namespace {

struct Config {
    int extract, encode, inFlight, quality;
    bool maxCompression;
    std::shared_ptr<GpuBackend> gpu;  // null: CPU path
    bool hybrid = false;  // integrated GPU: CPU decode, GPU resize (zero copy both ways)
};

Config resolve(const PP_Options* o, int count) {
    const int n = logical_cores(), maxThreads = std::max(1, count);
    auto pick = [](int v, int def) { return v > 0 ? v : def; };
    Config c{};
    c.extract = std::clamp(pick(o ? o->extractThreads : 0, n), 1, maxThreads);
    c.encode = std::clamp(pick(o ? o->encodeThreads : 0, n), 1, maxThreads);
    c.inFlight = std::max(1, pick(o ? o->maxInFlight : 0, 2 * n));
    c.quality = (o && o->jpegQuality >= 1 && o->jpegQuality <= 100) ? o->jpegQuality : 82;
    c.maxCompression = o && o->maxCompression;
    if (!(o && o->forceCpu)) c.gpu = gpu_backend();
    // Integrated GPUs take the hybrid path (faster than the full GPU path on both measured:
    // Adreno X1 and a 2-CU Radeon); PP_GPU_HYBRID=0 forces the full GPU path.
    char hy[4] = {};
    size_t hyLen = 0;
    const bool noHybrid = getenv_s(&hyLen, hy, sizeof hy, "PP_GPU_HYBRID") == 0 && hyLen > 0 && hy[0] == '0';
    c.hybrid = c.gpu && c.gpu->hybrid() && !noHybrid;
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
    uint64_t maxDevice = 0;  // GPU memory of the largest image (one streamer)
    int streamers = 0;       // GPU streamers that fit
};

// GPU streamers for the batch: one per extract thread, fewer if they don't all fit in the GPU
// memory that is free now (each keeps the memory of the largest image it has had).
int gpu_streamers(const Config& c, uint64_t maxDevice) {
    if (!c.gpu) return 0;
    uint64_t freeB = 0, totalB = 0;
    if (!maxDevice || !c.gpu->memory(freeB, totalB)) return c.extract;
    const uint64_t reserve = 256ull << 20;  // for everyone else
    const uint64_t usable = freeB > reserve ? freeB - reserve : 0;
    return int(std::clamp<uint64_t>(usable / maxDevice, 1, uint64_t(c.extract)));
}

Plan plan_memory(const std::vector<Slot>& slots, const Config& c) {
    Plan p;
    std::vector<uint64_t> resized;
    uint64_t maxSrc = 0, maxScratch = 0, maxEncode = 0, results = 0, maxSrcCpuOnly = 0, maxDev = 0;
    for (const Slot& s : slots) {
        if (s.code) continue;
        resized.push_back(uint64_t(s.dw) * s.dh * kPixelBytes);
        maxSrc = std::max(maxSrc, s.hdr.decodedBytes());
        maxScratch = std::max(maxScratch, decode_scratch_bytes(s.hdr));
        maxEncode = std::max(maxEncode, jpeg_working_bytes(s.dw, s.dh));
        results += jpeg_estimated_bytes(s.dw, s.dh);
        if (c.gpu && c.gpu->supports(s.hdr)) maxDev = std::max(maxDev, c.gpu->device_bytes(s.hdr, s.dw, s.dh));
        else maxSrcCpuOnly = std::max(maxSrcCpuOnly, s.hdr.decodedBytes());
    }
    // Worst case: the largest `inFlight` resized images are all waiting for the encoders.
    std::sort(resized.begin(), resized.end(), std::greater<>());
    uint64_t inFlight = 0;
    for (size_t i = 0; i < resized.size() && i < size_t(c.inFlight); ++i) inFlight += resized[i];
    const uint64_t extractors = uint64_t(std::min<int>(c.extract, int(resized.size())));
    const uint64_t encoders = uint64_t(std::min<int>(c.encode, int(resized.size())));
    p.host = inFlight + extractors * maxScratch + encoders * maxEncode + results + (64ull << 20);
    if (c.gpu && extractors) {
        // compressed data only passes through two band buffers per streamer; images live in GPU
        // memory (formats the GPU path doesn't take are decoded on the CPU, into a per-thread buffer)
        p.maxDevice = maxDev;
        p.streamers = std::min<int>(gpu_streamers(c, maxDev), int(extractors));
        p.host += uint64_t(p.streamers) * 2 * kGpuBandBytes + extractors * maxSrcCpuOnly;
        p.device = uint64_t(p.streamers) * maxDev + (64ull << 20);
    } else {
        p.host += extractors * maxSrc;  // one reused whole-image decode buffer per thread
    }
    return p;
}

// The GPU streamers, shared by the extract threads. At most `limit` exist; a thread that finds
// none free waits for one (GPU work is short next to everything else a page takes).
class StreamerPool {
public:
    StreamerPool(GpuBackend& gpu, int limit) : gpu_(gpu), limit_(limit) {}

    // Null when no streamer can be had at all (`error` says why): do the image on the CPU.
    std::unique_ptr<GpuStreamer> take(std::wstring& error) {
        std::unique_lock lk(m_);
        for (;;) {
            if (!free_.empty()) {
                auto s = std::move(free_.back());
                free_.pop_back();
                return s;
            }
            if (created_ < limit_) {
                ++created_;
                peak_ = std::max(peak_, created_);
                lk.unlock();
                auto s = gpu_.make_streamer(error);
                lk.lock();
                if (s) return s;
                limit_ = --created_;  // no more of them: make do with those that exist
            }
            if (created_ == 0) return nullptr;
            cv_.wait(lk);
        }
    }
    void give(std::unique_ptr<GpuStreamer> s) {
        {
            std::lock_guard lk(m_);
            free_.push_back(std::move(s));
        }
        cv_.notify_one();
    }
    int peak() {
        std::lock_guard lk(m_);
        return peak_;
    }

private:
    GpuBackend& gpu_;
    std::mutex m_;
    std::condition_variable cv_;
    std::vector<std::unique_ptr<GpuStreamer>> free_;
    int limit_, created_ = 0, peak_ = 0;
};

// Splits the images between the GPU and the CPU path when the GPU is the slower one: an image
// takes the GPU unless the work already queued there would keep it waiting longer than the CPU
// work it saves takes - decoding + resizing for the full GPU path, resizing alone for the
// hybrid path (the CPU decodes either way). A fast GPU never has such a backlog. Costs are per
// source megapixel: the GPU's from its timestamps, the CPU's (per extract thread, under load)
// measured here as images go that way, kept across batches.
// PP_GPU_BALANCE=0: everything the GPU takes goes to it; =1: balance, but use any GPU.
class Balancer {
public:
    Balancer(const GpuBackend& gpu, bool hybrid) : gpu_(gpu), hybrid_(hybrid) {}

    static int mode() {  // 0 off, 1 balance, 2 balance and skip a slow integrated GPU (default)
        char v[4] = {};
        size_t len = 0;
        if (getenv_s(&len, v, sizeof v, "PP_GPU_BALANCE") == 0 && len > 0) return v[0] == '0' ? 0 : 1;
        return 2;
    }
    // CPU work per source megapixel the GPU saves (seconds of one extract thread).
    static double cpu_saved(bool hybrid) { return hybrid ? cpuResize_.load() : cpuDecode_.load() + cpuResize_.load(); }

    // An integrated GPU much slower than the extract threads together at what it replaces isn't
    // worth it: it adds little throughput and takes memory bandwidth from the CPU (on a 2-CU
    // desktop iGPU that made batches 7% slower than the CPU alone, even balanced).
    static bool worth_it(const GpuBackend& gpu, bool hybrid, int extractThreads) {
        const double g = gpu.seconds_per_mpx(hybrid);
        return mode() < 2 || !gpu.integrated() || g <= 0 || g <= 2 * cpu_saved(hybrid) / extractThreads;
    }

    bool take_gpu(uint64_t pixels) {
        const double g = gpu_.seconds_per_mpx(hybrid_), mpx = double(pixels) / 1e6;
        if (on_ && g > 0) {
            const double wait = (double(queued_.load()) / 1e6 + mpx) * g;
            if (wait > mpx * cpu_saved(hybrid_)) return false;
        }
        queued_ += pixels;
        return true;
    }
    void gpu_done(uint64_t pixels) { queued_ -= pixels; }
    static void cpu_done(uint64_t pixels, double decodeSeconds, double resizeSeconds) {
        if (pixels < (1u << 18)) return;
        const double mpx = double(pixels) / 1e6;
        update(cpuDecode_, decodeSeconds / mpx);
        update(cpuResize_, resizeSeconds / mpx);
    }

private:
    static void update(std::atomic<double>& e, double v) {
        double old = e.load();
        while (!e.compare_exchange_weak(old, 0.8 * old + 0.2 * v)) {}
    }
    const GpuBackend& gpu_;
    const bool hybrid_;
    const bool on_ = mode() > 0;
    std::atomic<int64_t> queued_{0};  // source pixels sent to the GPU and not back yet
    // Defaults: a 16-thread desktop CPU (~12 ms per megapixel to decode + resize).
    static inline std::atomic<double> cpuDecode_{0.006}, cpuResize_{0.006};
};

std::unique_ptr<PP_Batch> run_batch(const wchar_t* const* paths, int count, int targetWidth, const PP_Options* opt,
                                    double* progressTarget) {
    const auto t0 = std::chrono::steady_clock::now();
    auto batch = std::make_unique<PP_Batch>();
    std::vector<Slot>& slots = batch->slots;
    slots.resize(size_t(count));
    Config cfg = resolve(opt, count);
    Progress progress(progressTarget, int64_t(count) * 3);
    const CpuBackend& cpu = cpu_backend();  // IPP dispatch must be settled before worker threads start
    std::wstring skipped;  // a GPU not used because it would slow the batch down
    if (cfg.gpu && !Balancer::worth_it(*cfg.gpu, cfg.hybrid, cfg.extract)) {
        skipped = cfg.gpu->description();
        cfg.gpu.reset();
        cfg.hybrid = false;
    }

    prepare_slots(slots, paths, targetWidth, cfg.extract, [&](Slot&, const FileStatus&) { progress.step(3); });
    std::unique_ptr<StreamerPool> gpuPool;
    std::unique_ptr<Balancer> balancer;
    if (cfg.gpu) {
        gpuPool = std::make_unique<StreamerPool>(*cfg.gpu, plan_memory(slots, cfg).streamers);
        balancer = std::make_unique<Balancer>(*cfg.gpu, cfg.hybrid);
    }
    std::atomic<int> balancedToCpu{0};  // images the GPU could have taken, done on the CPU while it was busy

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

    // An image in the balancer's GPU backlog, until released (or out of scope).
    struct Backlog {
        Balancer* balancer;
        uint64_t pixels;
        bool on = false;
        void release() {
            if (on) balancer->gpu_done(pixels);
            on = false;
        }
        ~Backlog() { release(); }
    };
    // Returns a pool streamer when it goes out of scope.
    struct Lease {
        StreamerPool* pool;
        std::unique_ptr<GpuStreamer> s;
        ~Lease() {
            if (s) pool->give(std::move(s));
        }
    };

    auto extractWorker = [&] {
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
                if (!cfg.hybrid) s.resized = pool.take(size_t(s.dw) * s.dh * kPixelBytes);

                bool resized = false;
                Lease gs{gpuPool.get(), nullptr};
                const uint64_t pixels = uint64_t(s.hdr.width) * s.hdr.height;
                Backlog queued{balancer.get(), pixels};
                if (gpuPool && (cfg.hybrid || cfg.gpu->supports(s.hdr)) && !(queued.on = balancer->take_gpu(pixels))) ++balancedToCpu;
                if (queued.on) {
                    std::wstring gerr;
                    {
                        Timed t(stats.extractWait);
                        gs.s = gpuPool->take(gerr);
                    }
                    if (!gs.s) {
                        ++stats.gpuFallbacks;
                        note(gerr);
                    }
                }
                if (gs.s && cfg.hybrid) {  // the CPU decodes into GPU memory, the GPU resizes
                    std::wstring gerr;
                    FileStatus st;
                    RowSink* target;
                    {
                        Timed t(stats.decode);
                        target = gs.s->decode_target(s.hdr, gerr);
                        if (target) st = decode_composite(f, s.hdr, *target);
                    }
                    if (!st.ok()) {
                        gs.s->abort();
                        fail(s, st.code, st.message, 3);
                        continue;
                    }
                    if (target) {
                        progress.step();
                        extracted = true;
                        Timed t(stats.resize);
                        s.image = gs.s->resize_decoded(s.dw, s.dh, gerr);
                        resized = s.image != nullptr;
                    }
                    if (!resized) {
                        ++stats.gpuFallbacks;
                        note(gerr);
                    }
                    gpuPool->give(std::move(gs.s));
                } else if (gs.s) {
                    std::wstring gerr;
                    FileStatus st;
                    bool uploaded;
                    {
                        Timed t(stats.decode);
                        uploaded = gs.s->upload(f, s.hdr, st, gerr);  // compressed data streams to the GPU
                    }
                    if (!st.ok()) { fail(s, st.code, st.message, 3); continue; }
                    if (uploaded) {
                        progress.step();
                        extracted = true;
                        Timed t(stats.resize);
                        resized = gs.s->finish(s.dw, s.dh, s.resized.data(), st, gerr);  // unpack + resize on the GPU
                        if (!st.ok()) { fail(s, st.code, st.message, 2); continue; }
                    }
                    if (!resized) {  // e.g. out of GPU memory: redo this one on the CPU
                        ++stats.gpuFallbacks;
                        note(gerr);
                    }
                    gpuPool->give(std::move(gs.s));  // before any CPU work: other threads may want it
                }
                queued.release();
                if (!resized) {
                    if (s.resized.empty()) s.resized = pool.take(size_t(s.dw) * s.dh * kPixelBytes);
                    const auto c0 = std::chrono::steady_clock::now();
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
                    const auto c1 = std::chrono::steady_clock::now();
                    Timed t(stats.resize);
                    if (!cpu_resize(decodeBuf.data(), int(s.hdr.width), int(s.hdr.height), s.resized.data(), s.dw, s.dh, err)) {
                        fail(s, PP_FILE_RESIZE_FAILED, err, 2);
                        continue;
                    }
                    using Seconds = std::chrono::duration<double>;
                    Balancer::cpu_done(pixels, Seconds(c1 - c0).count(), Seconds(std::chrono::steady_clock::now() - c1).count());
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
                to_srgb(s.color, s.hdr.colorMode, s.pixels(), size_t(s.dw) * s.dh);
                ok = encode_jpeg(s.pixels(), s.dw, s.dh, cfg.quality, cfg.maxCompression, s.jpeg, err);
            } catch (const std::bad_alloc&) {
                code = PP_FILE_OUT_OF_MEMORY;
                err = L"out of memory while encoding";
            }
            s.image.reset();  // back to the GPU backend's pool
            if (s.resized.capacity()) pool.give(std::move(s.resized));
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
    if (gpuPool) batch->gpuStreamers = gpuPool->peak();
    gpuPool.reset();  // frees the streamers' GPU memory
    if (cfg.gpu) cfg.gpu->trim();
    progress.finish();

    batch->usedGpu = cfg.gpu != nullptr;
    if (cfg.gpu) {
        batch->backend = std::format(L"{}, {} streamers{}", cfg.gpu->description(), batch->gpuStreamers,
                                     cfg.hybrid ? L", hybrid: CPU decode + GPU resize" : L"");
        if (balancedToCpu > 0) batch->backend += std::format(L", {} images on the CPU while the GPU was busy", balancedToCpu.load());
    } else {
        batch->backend = cpu.stb ? std::format(L"{} {} ({})", cpu.name, cpu.version, cpu.target)
                                 : std::format(L"Intel IPP {}, {} dispatch{}", cpu.version, cpu.target, cpu.avx512Active ? L" (AVX-512)" : L"");
        if (!skipped.empty()) batch->backend += L" (GPU not used: too slow to help)";
    }
    batch->backend += std::format(L"; {} extract / {} encode threads; mozjpeg q{}{}", cfg.extract, cfg.encode, cfg.quality,
                                  cfg.maxCompression ? L" max compression" : L"");
    if (cfg.gpu) {
        const std::wstring prof = cfg.gpu->profile();
        if (!prof.empty()) batch->backend += L"; " + prof;
    }
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

PP_API int32_t pp_gpu_init(int32_t backend, int32_t adapter, PP_DeviceInfo* info) {
    PP_GUARD_BEGIN
    return gpu_init(backend, adapter, info);
    PP_GUARD_END
}

PP_API int32_t pp_gpu_ready(void) { return gpu_backend() ? 1 : 0; }

PP_API void pp_gpu_shutdown(void) { gpu_shutdown(); }

PP_API int32_t pp_cpu_info(PP_CpuInfo* info) {
    if (!info) return PP_ERR_INVALID_ARGUMENT;
    PP_GUARD_BEGIN
    const CpuBackend& b = cpu_backend();
    *info = {};
    info->logicalCores = logical_cores();
    info->avx512Supported = b.avx512Supported;
    info->avx512Active = b.avx512Active;
    info->forcedDispatch = b.forcedDispatch;
    copy_str(info->resizeTarget, b.target);
    copy_str(info->resizeName, b.name);
    copy_str(info->resizeVersion, b.version);
    copy_str(info->simdLevel, simd::kernels().name);
#if defined(_M_ARM64)
    info->arm64 = 1;
#endif
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
    report->usesGpu = cfg.gpu != nullptr;
    report->deviceRequired = p.device;
    uint64_t freeB = 0, totalB = 0;
    if (cfg.gpu && cfg.gpu->memory(freeB, totalB)) report->deviceAvailable = freeB;
    // Fewer streamers are used when they don't all fit; one image at a time must.
    report->deviceOk = !cfg.gpu || p.maxDevice <= report->deviceAvailable;
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
    info->gpuStreamers = batch->gpuStreamers;
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
