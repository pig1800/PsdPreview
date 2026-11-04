using System.Runtime.InteropServices;

namespace PsdPreview.Interop;

// Mirrors native/include/psdpreview.h. All structs are blittable (fixed char buffers
// are UTF-16, matching wchar_t on Windows), so calls marshal nothing.

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDeviceInfo
{
    public ulong TotalMemory;
    public ulong FreeMemory;
    public int Backend;
    public int Adapter;
    public int ComputeMajor;
    public int ComputeMinor;
    public int MultiProcessors;
    public int WaveLanesMin;
    public int WaveLanesMax;
    public int Uma;
    public int CacheCoherentUma;
    public int DirectResult;
    public int Hybrid;
    public int CudaRuntimeVersion;
    public int CudaDriverVersion;
    public fixed char Name[256];
    public fixed char Message[Native.MessageCap];
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeCpuInfo
{
    public int LogicalCores;
    public int Avx512Supported;
    public int Avx512Active;
    public int ForcedDispatch;
    public fixed char ResizeTarget[8];
    public fixed char ResizeName[128];
    public fixed char ResizeVersion[64];
    public fixed char SimdLevel[16];
    public int Arm64;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeOptions
{
    public int JpegQuality;
    public int ForceCpu;
    public int ExtractThreads;
    public int EncodeThreads;
    public int MaxInFlight;
    public int MaxCompression;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeIssue
{
    public int Index;
    public int Code;
    public fixed char Message[Native.MessageCap];
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeMemoryReport
{
    public ulong HostRequired;
    public ulong HostAvailable;
    public ulong DeviceRequired;
    public ulong DeviceAvailable;
    public int UsesGpu;
    public int HostOk;
    public int DeviceOk;
    public int UnreadableFiles;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeBatchInfo
{
    public double ElapsedSeconds;
    public double DecodeSeconds;
    public double ResizeSeconds;
    public double EncodeSeconds;
    public double ExtractWaitSeconds;
    public double EncodeIdleSeconds;
    public int Count;
    public int Succeeded;
    public int UsedGpu;
    public int GpuFallbacks;
    public int ExtractThreads;
    public int EncodeThreads;
    public int GpuStreamers;
    public fixed char Backend[Native.MessageCap];
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeItem
{
    public char* SourcePath;
    public byte* Data;
    public ulong Size;
    public int Width;
    public int Height;
    public int Code;
    public int ColorConverted;
    public fixed char Message[Native.MessageCap];
    public fixed char ColorProfile[128];
}

internal static unsafe partial class Native
{
    internal const int MessageCap = 260;
    private const string Lib = "psdpreview_native";

    // The native DLL is built per architecture and sits in x64\ or arm64\ next to the app (or,
    // in a single-file app, where the bundle extracted its native libraries); load the one
    // matching this process (an x64 app under emulation on ARM64 gets the x64 one).
    // Falls back to the default search (e.g. the DLL right next to the app).
    static Native()
    {
        NativeLibrary.SetDllImportResolver(typeof(Native).Assembly, (name, assembly, path) =>
        {
            if (name != Lib) return IntPtr.Zero;
            string? arch = RuntimeInformation.ProcessArchitecture switch
            {
                Architecture.X64 => "x64",
                Architecture.Arm64 => "arm64",
                _ => null,
            };
            if (arch == null) return IntPtr.Zero;
            var dirs = new List<string> { AppContext.BaseDirectory };
            if (AppContext.GetData("NATIVE_DLL_SEARCH_DIRECTORIES") is string extra)
                dirs.AddRange(extra.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries));
            foreach (var dir in dirs)
                if (NativeLibrary.TryLoad(Path.Combine(dir, arch, Lib + ".dll"), out var handle))
                    return handle;
            return IntPtr.Zero;
        });
    }

    [LibraryImport(Lib)] internal static partial int pp_gpu_init(int backend, int adapter, NativeDeviceInfo* info);
    [LibraryImport(Lib)] internal static partial int pp_gpu_ready();
    [LibraryImport(Lib)] internal static partial void pp_gpu_shutdown();
    [LibraryImport(Lib)] internal static partial int pp_cpu_info(NativeCpuInfo* info);
    [LibraryImport(Lib)] internal static partial int pp_validate(char** paths, int count, int targetWidth, NativeIssue* issues, int* issueCount);
    [LibraryImport(Lib)] internal static partial int pp_check_memory(char** paths, int count, int targetWidth, NativeOptions* options, NativeMemoryReport* report);
    [LibraryImport(Lib)] internal static partial int pp_process(char** paths, int count, int targetWidth, NativeOptions* options, double* progress, nint* batch);
    [LibraryImport(Lib)] internal static partial int pp_batch_info(nint batch, NativeBatchInfo* info);
    [LibraryImport(Lib)] internal static partial int pp_batch_item(nint batch, int index, NativeItem* item);
    [LibraryImport(Lib)] internal static partial void pp_batch_release_item(nint batch, int index);
    [LibraryImport(Lib)] internal static partial void pp_batch_free(nint batch);

    internal static string Str(char* p) => p == null ? string.Empty : new string(p);
}

/// <summary>Packs a path list into one unmanaged block: a wchar_t* array followed by the strings.</summary>
internal sealed unsafe class NativePathList : IDisposable
{
    private void* _block;

    public NativePathList(IReadOnlyList<string> paths)
    {
        Count = paths.Count;
        nuint chars = 0;
        var normalized = new string[Count];
        for (int i = 0; i < Count; i++)
        {
            normalized[i] = Normalize(paths[i]);
            chars += (nuint)normalized[i].Length + 1;
        }
        nuint pointerBytes = (nuint)Count * (nuint)sizeof(char*);
        _block = NativeMemory.Alloc(pointerBytes + chars * sizeof(char) + 1);
        var table = (char**)_block;
        var text = (char*)((byte*)_block + pointerBytes);
        for (int i = 0; i < Count; i++)
        {
            table[i] = text;
            normalized[i].AsSpan().CopyTo(new Span<char>(text, normalized[i].Length));
            text[normalized[i].Length] = '\0';
            text += normalized[i].Length + 1;
        }
    }

    public int Count { get; }
    public char** Pointer => (char**)_block;

    private static string Normalize(string? path)
    {
        if (string.IsNullOrWhiteSpace(path)) return string.Empty;
        try { return Path.GetFullPath(path); }
        catch (Exception) { return path; } // native side reports the open failure
    }

    public void Dispose()
    {
        if (_block != null) { NativeMemory.Free(_block); _block = null; }
    }
}
