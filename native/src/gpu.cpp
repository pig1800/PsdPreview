// Backend selection (pp_gpu_init).
#include "gpu.h"

namespace pp {
namespace {

// Never destroyed: releasing a D3D12 device from a static destructor, while the DLL unloads at
// process exit, can crash inside the driver (it may be torn down already). pp_gpu_shutdown
// releases it while the process is alive.
struct Globals {
    std::mutex mutex;
    std::shared_ptr<GpuBackend> backend;
    PP_DeviceInfo info{};
    int request = -1, adapter = -1;  // what `backend` was initialised for
};
Globals& globals() {
    static Globals* g = new Globals;
    return *g;
}

}  // namespace

int gpu_init(int backend, int adapter, PP_DeviceInfo* info) {
    Globals& G = globals();
    std::lock_guard lk(G.mutex);
    PP_DeviceInfo scratch{};
    PP_DeviceInfo& out = info ? *info : scratch;
    out = {};
    if (backend < PP_GPU_AUTO || backend > PP_GPU_D3D12_WARP || adapter < 0) {
        copy_str(out.message, L"unknown backend or negative adapter index");
        return PP_ERR_INVALID_ARGUMENT;
    }
    if (G.backend && G.request == backend && G.adapter == adapter) {
        out = G.info;
        uint64_t freeB = 0, totalB = 0;
        if (G.backend->memory(freeB, totalB)) out.freeMemory = freeB;
        copy_str(out.message, L"already initialised");
        return PP_OK;
    }
    G.backend.reset();  // batches already running keep their reference
    G.request = G.adapter = -1;

    std::shared_ptr<GpuBackend> b;
    int code = PP_ERR_GPU_UNAVAILABLE;
    switch (backend) {
    case PP_GPU_CUDA:
#if PP_HAVE_CUDA
        code = cuda_create(adapter, out, b);
#else
        copy_str(out.message, L"this build has no CUDA (ARM64)");
#endif
        break;
    case PP_GPU_D3D12: code = d3d12_create(adapter, false, out, b); break;
    case PP_GPU_D3D12_WARP: code = d3d12_create(0, true, out, b); break;
    default: {  // auto: CUDA on NVIDIA, else D3D12 on the high-performance adapter
        std::wstring why;
#if PP_HAVE_CUDA
        code = cuda_create(adapter, out, b);
        if (code != PP_OK) why = L"CUDA: " + std::wstring(out.message) + L"; ";
#endif
        if (!b) {
            out = {};
            code = d3d12_create(adapter, false, out, b);
            if (code != PP_OK) why += L"D3D12: ";
            copy_str(out.message, why + out.message);
        }
        break;
    }
    }
    if (code != PP_OK || !b) return code != PP_OK ? code : PP_ERR_INTERNAL;
    G.backend = std::move(b);
    G.info = out;
    G.request = backend;
    G.adapter = adapter;
    return PP_OK;
}

std::shared_ptr<GpuBackend> gpu_backend() {
    Globals& G = globals();
    std::lock_guard lk(G.mutex);
    return G.backend;
}

void gpu_shutdown() {
    Globals& G = globals();
    std::lock_guard lk(G.mutex);
    if (G.backend) G.backend->trim();
    G.backend.reset();
    G.request = G.adapter = -1;
}

}  // namespace pp
