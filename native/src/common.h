#pragma once
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <iterator>
#include <memory>
#include <cstdint>
#include <cstring>
#include <deque>
#include <format>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "psdpreview.h"

namespace pp {

// Every pixel buffer from extraction up to the JPEG encoder is 4 bytes per pixel:
// RGBA8888 (gray expanded), or CMYK until it is converted after the resize.
constexpr int kPixelBytes = 4;

// std::vector that doesn't zero-fill on resize (pixel buffers are fully overwritten).
template <class T>
struct DefaultInitAllocator : std::allocator<T> {
    template <class U>
    struct rebind { using other = DefaultInitAllocator<U>; };
    DefaultInitAllocator() = default;
    template <class U>
    DefaultInitAllocator(const DefaultInitAllocator<U>&) noexcept {}
    template <class U>
    void construct(U* p) noexcept { ::new (static_cast<void*>(p)) U; }
    template <class U, class... A>
    void construct(U* p, A&&... a) { ::new (static_cast<void*>(p)) U(std::forward<A>(a)...); }
};
using Bytes = std::vector<uint8_t, DefaultInitAllocator<uint8_t>>;

inline uint16_t be16(const uint8_t* p) { return uint16_t(p[0] << 8 | p[1]); }
inline uint32_t be32(const uint8_t* p) { return uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | p[3]; }
inline uint64_t be64(const uint8_t* p) { return uint64_t(be32(p)) << 32 | be32(p + 4); }

inline std::wstring widen(const char* s) {
    if (!s || !*s) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
    std::wstring w(n > 0 ? n - 1 : 0, L'\0');
    if (n > 1) MultiByteToWideChar(CP_UTF8, 0, s, -1, w.data(), n);
    return w;
}

inline std::wstring win32_message(DWORD err) {
    wchar_t* buf = nullptr;
    DWORD n = FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                             nullptr, err, 0, reinterpret_cast<wchar_t*>(&buf), 0, nullptr);
    std::wstring s = n ? std::wstring(buf, n) : std::format(L"Win32 error {}", err);
    if (buf) LocalFree(buf);
    while (!s.empty() && (s.back() == L'\n' || s.back() == L'\r' || s.back() == L' ' || s.back() == L'.')) s.pop_back();
    return s;
}

template <size_t N>
inline void copy_str(wchar_t (&dst)[N], const std::wstring& src) {
    size_t n = std::min(src.size(), N - 1);
    std::wmemcpy(dst, src.data(), n);
    dst[n] = L'\0';
}

// Read-only file with positional reads (no shared seek state).
class FileReader {
public:
    FileReader() = default;
    FileReader(const FileReader&) = delete;
    FileReader& operator=(const FileReader&) = delete;
    ~FileReader() { if (h_ != INVALID_HANDLE_VALUE) CloseHandle(h_); }

    bool open(const std::wstring& path, std::wstring& error) {
        std::wstring p = path;
        // Long paths need the \\?\ prefix unless the host process is long-path aware.
        if (p.size() >= MAX_PATH - 12 && p.rfind(L"\\\\?\\", 0) != 0) {
            if (p.rfind(L"\\\\", 0) == 0) p = L"\\\\?\\UNC\\" + p.substr(2);
            else if (p.size() > 2 && p[1] == L':') p = L"\\\\?\\" + p;
        }
        h_ = CreateFileW(p.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                         FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (h_ == INVALID_HANDLE_VALUE) { error = win32_message(GetLastError()); return false; }
        LARGE_INTEGER sz{};
        if (!GetFileSizeEx(h_, &sz)) { error = win32_message(GetLastError()); return false; }
        size_ = uint64_t(sz.QuadPart);
        return true;
    }

    uint64_t size() const { return size_; }

    // Returns the number of bytes actually read (short at EOF or on error).
    size_t read_at(uint64_t offset, void* dst, size_t n) const {
        size_t done = 0;
        auto* out = static_cast<uint8_t*>(dst);
        while (done < n) {
            DWORD chunk = DWORD(std::min<size_t>(n - done, 1u << 30));
            OVERLAPPED ov{};
            uint64_t off = offset + done;
            ov.Offset = DWORD(off);
            ov.OffsetHigh = DWORD(off >> 32);
            DWORD got = 0;
            if (!ReadFile(h_, out + done, chunk, &got, &ov) || got == 0) break;
            done += got;
        }
        return done;
    }
    bool read_exact(uint64_t offset, void* dst, size_t n) const { return read_at(offset, dst, n) == n; }

private:
    HANDLE h_ = INVALID_HANDLE_VALUE;
    uint64_t size_ = 0;
};

// Hand-off list of slot indices between pipeline stages.
class IndexQueue {
public:
    void push(int v) {
        { std::lock_guard lk(m_); q_.push_back(v); }
        cv_.notify_one();
    }
    bool try_pop(int& v) {
        std::lock_guard lk(m_);
        if (q_.empty()) return false;
        v = q_.front();
        q_.pop_front();
        return true;
    }
    // Blocks until an item is available; returns false once closed and drained.
    bool pop(int& v) {
        std::unique_lock lk(m_);
        cv_.wait(lk, [&] { return !q_.empty() || closed_; });
        if (q_.empty()) return false;
        v = q_.front();
        q_.pop_front();
        return true;
    }
    void close() {
        { std::lock_guard lk(m_); closed_ = true; }
        cv_.notify_all();
    }

private:
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<int> q_;
    bool closed_ = false;
};

// Monotonic 0..1 progress written to caller-owned memory.
class Progress {
public:
    Progress(double* target, int64_t totalSteps) : target_(target), total_(std::max<int64_t>(1, totalSteps)) {
        if (target_) std::atomic_ref<double>(*target_).store(0.0);
    }
    void step(int64_t n = 1) {
        int64_t d = done_.fetch_add(n, std::memory_order_relaxed) + n;
        if (!target_) return;
        double v = std::min(1.0, double(d) / double(total_));
        std::atomic_ref<double> ref(*target_);
        double cur = ref.load(std::memory_order_relaxed);
        while (v > cur && !ref.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {}
    }
    void finish() { if (target_) std::atomic_ref<double>(*target_).store(1.0); }

private:
    double* target_;
    int64_t total_;
    std::atomic<int64_t> done_{0};
};

inline int logical_cores() { return std::max(1, int(std::thread::hardware_concurrency())); }

inline void parallel_for(int count, int threads, const std::function<void(int)>& fn) {
    threads = std::clamp(threads, 1, std::max(1, count));
    std::atomic<int> next{0};
    auto worker = [&] {
        for (int i; (i = next.fetch_add(1)) < count;) fn(i);
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < threads; ++t) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();
}

}  // namespace pp
