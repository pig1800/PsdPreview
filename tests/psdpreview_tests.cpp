// psdpreview_tests simd              kernels at every supported level vs. scalar, plus throughput
// psdpreview_tests sdk <psd/psb>...  patched psd_sdk ParseImageDataSection vs. our decoder (pixel-exact),
//                                    then truncated / corrupted in-memory copies must fail cleanly
// psdpreview_tests bench <psd/psb>.. decode speed per SIMD level
// psdpreview_tests rle               our PackBits decoder vs. psd_sdk's on random, also malformed, rows
#include <chrono>
#include <cstdio>
#include <random>

#include "PsdPch.h"
#include "PsdDecompressRle.h"
#include "psd.h"
#include "simd.h"
#include "simd_kernels.h"

#include "PsdPch.h"
#include "PsdDocument.h"
#include "PsdImageDataSection.h"
#include "PsdMallocAllocator.h"
#include "PsdMemoryFile.h"
#include "PsdNativeFile.h"
#include "PsdParseDocument.h"
#include "PsdParseImageDataSection.h"
#include "PsdPlanarImage.h"

using namespace pp;
using Clock = std::chrono::steady_clock;

static int g_failures = 0;
#define CHECK(cond, ...)                \
    do {                                \
        if (!(cond)) {                  \
            ++g_failures;               \
            wprintf(L"  FAIL: " __VA_ARGS__); \
            wprintf(L"\n");             \
        }                               \
    } while (0)

static std::vector<simd::Level> levels() {
    std::vector<simd::Level> v;
    for (int l = 0; l <= int(simd::detected_level()); ++l) v.push_back(simd::Level(l));
    return v;
}

static int test_simd() {
    std::mt19937 rng(12345);
    auto fill = [&](std::vector<uint8_t>& v) { for (auto& x : v) x = uint8_t(rng()); };
    const size_t sizes[] = {0, 1, 2, 3, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 200, 1000, 4097, 100003};
    for (simd::Level level : levels()) {
        const simd::Kernels& k = simd::kernels_for(level);
        const int before = g_failures;
        for (size_t n : sizes) {
            std::vector<uint8_t> a(n), b(n), c(n), d(n), s16(2 * n);
            fill(a); fill(b); fill(c); fill(d); fill(s16);
            std::vector<uint8_t> ref(4 * n + 64, 0xCD), out(4 * n + 64, 0xCD);  // guard bytes catch overruns
            simd::scalar::merge_rgb(a.data(), b.data(), c.data(), ref.data(), n);
            k.merge_rgb(a.data(), b.data(), c.data(), out.data(), n);
            CHECK(ref == out, L"%s merge_rgb n=%zu", k.name, n);
            simd::scalar::merge_gray(a.data(), ref.data(), n);
            k.merge_gray(a.data(), out.data(), n);
            CHECK(ref == out, L"%s merge_gray n=%zu", k.name, n);
            simd::scalar::merge_4(a.data(), b.data(), c.data(), d.data(), ref.data(), n);
            k.merge_4(a.data(), b.data(), c.data(), d.data(), out.data(), n);
            CHECK(ref == out, L"%s merge_4 n=%zu", k.name, n);
            std::vector<uint8_t> r16(n + 64, 0xCD), o16(n + 64, 0xCD);
            simd::scalar::narrow16(s16.data(), r16.data(), n);
            k.narrow16(s16.data(), o16.data(), n);
            CHECK(r16 == o16, L"%s narrow16 n=%zu", k.name, n);
            std::vector<uint8_t> inplace = s16;
            k.narrow16(inplace.data(), inplace.data(), n);
            CHECK(std::equal(r16.begin(), r16.begin() + n, inplace.begin()), L"%s narrow16 in-place n=%zu", k.name, n);
        }
        // spot-check semantics, not just agreement with scalar
        uint8_t px[4], one = 7;
        k.merge_gray(&one, px, 1);
        CHECK(px[0] == 7 && px[1] == 7 && px[2] == 7 && px[3] == 255, L"%s merge_gray value", k.name);
        const uint8_t be[2] = {0xAB, 0xCD};
        uint8_t hi;
        k.narrow16(be, &hi, 1);
        CHECK(hi == 0xAB, L"%s narrow16 keeps the high byte", k.name);
        wprintf(L"%-8s correctness: %s\n", k.name, g_failures == before ? L"PASS" : L"FAIL");
    }

    const size_t N = 64u << 20;  // 64 Mpx
    std::vector<uint8_t> r(N), g(N), b(N), s16(2 * N), dst(4 * N);
    fill(r); fill(g); fill(b); fill(s16);
    for (simd::Level level : levels()) {
        const simd::Kernels& k = simd::kernels_for(level);
        double bestMerge = 1e9, bestNarrow = 1e9;
        for (int rep = 0; rep < 5; ++rep) {
            auto t0 = Clock::now();
            k.merge_rgb(r.data(), g.data(), b.data(), dst.data(), N);
            auto t1 = Clock::now();
            k.narrow16(s16.data(), dst.data(), N);
            auto t2 = Clock::now();
            bestMerge = std::min(bestMerge, std::chrono::duration<double>(t1 - t0).count());
            bestNarrow = std::min(bestNarrow, std::chrono::duration<double>(t2 - t1).count());
        }
        wprintf(L"%-8s 64 Mpx buffers (DRAM): merge_rgb %6.1f Gpx/s   narrow16 %6.1f GB/s in\n", k.name, N / bestMerge / 1e9,
                2.0 * N / bestNarrow / 1e9);
    }
    // How the decoder uses them: one cache-resident row at a time.
    const size_t row = 1500, reps = 200000;
    for (simd::Level level : levels()) {
        const simd::Kernels& k = simd::kernels_for(level);
        auto t0 = Clock::now();
        for (size_t i = 0; i < reps; ++i) k.merge_rgb(r.data(), g.data(), b.data(), dst.data(), row);
        auto t1 = Clock::now();
        for (size_t i = 0; i < reps; ++i) k.narrow16(s16.data(), dst.data(), row);
        auto t2 = Clock::now();
        auto t3 = Clock::now();
        for (size_t i = 0; i < reps; ++i) simd::scalar::merge_rgb(r.data(), g.data(), b.data(), dst.data(), row);
        auto t4 = Clock::now();
        const double m = std::chrono::duration<double>(t1 - t0).count(), n16 = std::chrono::duration<double>(t2 - t1).count();
        const double sc = std::chrono::duration<double>(t4 - t3).count();
        wprintf(L"%-8s 1500-px rows (cache): merge_rgb %6.1f Gpx/s (scalar %4.1f)   narrow16 %6.1f Gsamples/s\n", k.name,
                row * reps / m / 1e9, row * reps / sc / 1e9, row * reps / n16 / 1e9);
    }
    return 0;
}

static bool read_all(const std::wstring& path, std::vector<uint8_t>& bytes) {
    FileReader f;
    std::wstring err;
    if (!f.open(path, err) || f.size() > (1ull << 31)) return false;
    bytes.resize(size_t(f.size()));
    return f.read_exact(0, bytes.data(), bytes.size());
}

// Runs the SDK's composite parser on a memory image; returns whether it produced data.
static bool sdk_parse_memory(const std::vector<uint8_t>& bytes, size_t size) {
    psd::MallocAllocator a;
    psd::MemoryFile mf(&a);
    mf.Open(bytes.data(), size);
    psd::Document* doc = psd::CreateDocument(&mf, &a);
    if (!doc) return false;
    psd::ImageDataSection* img = psd::ParseImageDataSection(doc, &mf, &a);
    const bool ok = img != nullptr;
    if (img) psd::DestroyImageDataSection(img, &a);
    psd::DestroyDocument(doc, &a);
    return ok;
}

static int test_sdk(int argc, wchar_t** argv) {
    for (int i = 0; i < argc; ++i) {
        const std::wstring path = argv[i];
        wprintf(L"%s\n", path.c_str());
        FileReader f;
        std::wstring err;
        PsdHeader h;
        if (!f.open(path, err) || !read_psd_header(f, h).ok() || h.depth == 32) {
            wprintf(L"  skipped (unreadable / unsupported / 32-bit)\n");
            continue;
        }
        Bytes ours;
        FileStatus st = decode_composite(f, h, ours);
        CHECK(st.ok(), L"our decoder: %s", st.message.c_str());

        psd::MallocAllocator a;
        psd::NativeFile nf(&a);
        CHECK(nf.OpenRead(path.c_str()), L"NativeFile open");
        psd::Document* doc = psd::CreateDocument(&nf, &a);
        psd::ImageDataSection* img = doc ? psd::ParseImageDataSection(doc, &nf, &a) : nullptr;
        CHECK(img != nullptr, L"psd_sdk ParseImageDataSection returned null");
        if (img && st.ok()) {
            const size_t px = size_t(h.width) * h.height;
            const int need = h.sourceChannels();
            size_t mismatches = 0;
            for (int c = 0; c < need; ++c) {
                const void* plane = img->images[c].data;
                for (size_t p = 0; p < px; ++p) {
                    const uint8_t sdk = h.depth == 8 ? static_cast<const uint8_t*>(plane)[p]
                                                     : uint8_t(static_cast<const uint16_t*>(plane)[p] >> 8);
                    mismatches += sdk != ours[p * kPixelBytes + c];
                }
            }
            CHECK(mismatches == 0, L"%zu samples differ between psd_sdk and our decoder", mismatches);
            if (mismatches == 0)
                wprintf(L"  psd_sdk == ours (%ux%u, %s, %d-bit, %s)\n", h.width, h.height, h.version == 2 ? L"PSB" : L"PSD",
                        h.depth, h.compression == 1 ? L"RLE" : L"raw");
        }
        if (img) psd::DestroyImageDataSection(img, &a);
        if (doc) psd::DestroyDocument(doc, &a);
        nf.Close();

        std::vector<uint8_t> bytes;
        if (!read_all(path, bytes)) continue;
        // Truncated copy (inside the composite): must fail, not read past the buffer.
        const size_t cut = size_t(h.imageDataPos + (h.fileSize - h.imageDataPos) / 2);
        CHECK(!sdk_parse_memory(bytes, cut), L"truncated copy was accepted");
        // Corrupted copy: smash part of the RLE data; must not crash (result may be garbage or null).
        for (size_t p = cut - cut / 8; p < cut; ++p) bytes[p] = 0x7F;
        sdk_parse_memory(bytes, bytes.size());
        wprintf(L"  truncated + corrupted in-memory copies handled\n");
    }
    wprintf(L"sdk: %s\n", g_failures ? L"FAIL" : L"PASS");
    return 0;
}

static int bench(int argc, wchar_t** argv) {
    std::vector<std::pair<std::wstring, PsdHeader>> files;
    for (int i = 0; i < argc; ++i) {
        FileReader f;
        std::wstring err;
        PsdHeader h;
        if (f.open(argv[i], err) && read_psd_header(f, h).ok()) files.emplace_back(argv[i], h);
    }
    Bytes out;
    for (int pass = 0; pass < 2; ++pass) {  // pass 0 warms the file cache
        for (simd::Level level : levels()) {
            simd::force_level(level);
            double secs = 0, mpx = 0;
            for (auto& [path, h] : files) {
                FileReader f;
                std::wstring err;
                f.open(path, err);
                auto t0 = Clock::now();
                decode_composite(f, h, out);
                secs += std::chrono::duration<double>(Clock::now() - t0).count();
                mpx += double(h.width) * h.height / 1e6;
            }
            if (pass == 1) wprintf(L"%-8s decode %7.1f Mpx/s (%zu files, single thread)\n", simd::kernels().name, mpx / secs, files.size());
        }
    }
    return 0;
}

// Random rows: mostly valid PackBits of random runs (sometimes ending early, overrunning the row
// or with 0x80 no-ops), plus pure noise. Output and result must equal psd_sdk's DecompressRle.
static void test_rle() {
    std::mt19937 rng(7);
    std::vector<uint8_t> src, a, b;
    size_t mismatches = 0, malformed = 0, runs = 0;
    for (int iter = 0; iter < 200000; ++iter) {
        const size_t size = 1 + rng() % 600;
        src.clear();
        if (iter % 10 == 9) {
            const size_t n = rng() % 700;
            for (size_t i = 0; i < n; ++i) src.push_back(uint8_t(rng()));
        } else {
            size_t out = 0;
            const size_t target = size + (rng() % 4 == 0 ? rng() % 200 : 0) - (rng() % 8 == 0 ? std::min<size_t>(size, rng() % 50) : 0);
            while (out < target) {
                const uint32_t kind = rng() % 9;
                if (kind == 0) { src.push_back(0x80); continue; }
                const size_t n = 1 + rng() % (kind < 4 ? 128 : 127);
                if (kind < 4) {  // literal
                    src.push_back(uint8_t(n - 1));
                    for (size_t i = 0; i < n; ++i) src.push_back(uint8_t(rng()));
                    out += n;
                } else {  // repeat of n + 1 >= 2
                    src.push_back(uint8_t(257 - (n + 1)));
                    src.push_back(uint8_t(rng()));
                    out += n + 1;
                }
                ++runs;
            }
            if (rng() % 16 == 0 && !src.empty()) src.resize(rng() % src.size());  // cut short
        }
        const size_t srcSize = src.size();
        src.resize(srcSize + kRleSlack);
        a.assign(size + kRleSlack, 0xCD);
        b.assign(size + kRleSlack, 0xCD);
        const bool okSdk = psd::imageUtil::DecompressRle(src.data(), srcSize, a.data(), size);
        const bool okOurs = unpack_row(src.data(), srcSize, b.data(), size);
        malformed += !okSdk;
        if (okSdk != okOurs || memcmp(a.data(), b.data(), size) != 0) ++mismatches;
    }
    CHECK(mismatches == 0, L"%zu of 200000 rows differ from psd_sdk's DecompressRle", mismatches);
    wprintf(L"rle: 200000 rows (%zu runs, %zu malformed rows), %zu mismatches: %s\n", runs, malformed, mismatches,
            mismatches ? L"FAIL" : L"PASS");
}

int wmain(int argc, wchar_t** argv) {
    if (argc >= 2 && !wcscmp(argv[1], L"simd")) test_simd();
    else if (argc >= 2 && !wcscmp(argv[1], L"rle")) test_rle();
    else if (argc >= 3 && !wcscmp(argv[1], L"sdk")) test_sdk(argc - 2, argv + 2);
    else if (argc >= 3 && !wcscmp(argv[1], L"bench")) bench(argc - 2, argv + 2);
    else {
        wprintf(L"usage: psdpreview_tests simd | rle | sdk <files> | bench <files>\n");
        return 2;
    }
    return g_failures ? 1 : 0;
}
