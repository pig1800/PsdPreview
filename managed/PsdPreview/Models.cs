namespace PsdPreview;

/// <summary>Return codes of the native API.</summary>
public enum PsdPreviewStatus
{
    Ok = 0,
    InvalidArgument = 1,
    /// <summary>No GPU driver, or no such device / adapter.</summary>
    GpuUnavailable = 2,
    /// <summary>CUDA or D3D12 error during initialisation.</summary>
    GpuFailed = 3,
    /// <summary>The GPU kernels failed to load (e.g. driver too old) or their self-test failed.</summary>
    KernelUnavailable = 4,
    OutOfMemory = 5,
    Internal = 6,
}

/// <summary>Why a file cannot be (or could not be) processed.</summary>
public enum PsdProblemKind
{
    None = 0,
    OpenFailed = 1,
    NotPsd = 2,
    UnsupportedVersion = 3,
    UnsupportedColorMode = 4,
    UnsupportedDepth = 5,
    UnsupportedCompression = 6,
    /// <summary>Saved with "Maximize PSD and PSB File Compatibility" off.</summary>
    NoComposite = 7,
    /// <summary>A non-Photoshop writer stored an empty placeholder composite.</summary>
    BlankComposite = 8,
    Truncated = 9,
    Corrupt = 10,
    BadDimensions = 11,
    /// <summary>The resized height would exceed the JPEG limit of 65500 px.</summary>
    OutputTooLarge = 12,
    OutOfMemory = 13,
    ResizeFailed = 14,
    EncodeFailed = 15,
}

public sealed record PsdFileProblem(int Index, string Path, PsdProblemKind Kind, string Message)
{
    public string FileName => System.IO.Path.GetFileName(Path);
    public override string ToString() => $"{FileName}: {Message}";
}

/// <summary>GPU APIs.</summary>
public enum GpuBackend
{
    /// <summary>CUDA if it works (NVIDIA, x64), else D3D12 on the high-performance adapter.</summary>
    Auto = 0,
    /// <summary>NVIDIA GPUs; x64 builds only.</summary>
    Cuda = 1,
    /// <summary>D3D12 compute: any GPU with a D3D12 driver (AMD, Intel, Qualcomm Adreno, NVIDIA).</summary>
    D3D12 = 2,
    /// <summary>D3D12 on WARP, the software rasterizer of Windows (testing).</summary>
    D3D12Warp = 3,
}

/// <param name="Status">Ok, or why no GPU is used.</param>
/// <param name="Message">Failure reason; with <see cref="GpuBackend.Auto"/> also why CUDA was skipped.</param>
/// <param name="Backend">The API that was initialised.</param>
/// <param name="Adapter">CUDA device / D3D12 adapter index (DXGI high-performance order).</param>
/// <param name="DeviceName">GPU name.</param>
/// <param name="ComputeCapability">CUDA compute capability, or D3D12 shader model (6.x).</param>
/// <param name="WaveLanesMin">SIMD width: CUDA warp size, or the smallest D3D12 wave.</param>
/// <param name="WaveLanesMax">The largest D3D12 wave.</param>
/// <param name="IsUma">Integrated GPU sharing system memory: the compressed data is read straight into memory it reads (no copy).</param>
/// <param name="IsCacheCoherentUma">UMA with CPU and GPU caches coherent. When false, memory the CPU writes for the GPU is
/// write-combined (uncached; the library only ever writes it sequentially).</param>
/// <param name="DirectResult">D3D12: the result is written straight into CPU-readable memory (else copied there by the GPU).</param>
/// <param name="Hybrid">The path batches take: true (D3D12 on UMA) = the CPU decodes into memory the GPU reads, the GPU
/// resizes into memory the encoder reads; false = the GPU unpacks and resizes (discrete GPUs, CUDA).</param>
/// <param name="TotalMemoryBytes">VRAM; for D3D12 the OS budget for the GPU (system memory on UMA).</param>
/// <param name="FreeMemoryBytes">What is left of it now.</param>
/// <param name="CudaRuntimeVersion">CUDA runtime version (e.g. 13000); 0 for D3D12.</param>
/// <param name="CudaDriverVersion">CUDA driver version; 0 for D3D12.</param>
public sealed record GpuInitResult(
    PsdPreviewStatus Status,
    string Message,
    GpuBackend Backend,
    int Adapter,
    string DeviceName,
    Version ComputeCapability,
    int WaveLanesMin,
    int WaveLanesMax,
    bool IsUma,
    bool IsCacheCoherentUma,
    bool DirectResult,
    bool Hybrid,
    long TotalMemoryBytes,
    long FreeMemoryBytes,
    int CudaRuntimeVersion,
    int CudaDriverVersion)
{
    public bool Success => Status == PsdPreviewStatus.Ok;

    public override string ToString() => !Success
        ? $"GPU unavailable ({Status}: {Message})"
        : Backend == GpuBackend.Cuda
            ? $"CUDA: {DeviceName} sm_{ComputeCapability.Major}{ComputeCapability.Minor}, {FreeMemoryBytes / 1048576} MiB free"
            : $"D3D12: {DeviceName}, SM {ComputeCapability.Major}.{ComputeCapability.Minor}, waves {WaveLanesMin}-{WaveLanesMax}" +
              $"{(IsUma ? IsCacheCoherentUma ? ", UMA" : ", UMA (not cache-coherent)" : "")}{(DirectResult ? ", direct result" : "")}" +
              $"{(Hybrid ? ", hybrid: CPU decode + GPU resize" : ", GPU decode + resize")}, {FreeMemoryBytes / 1048576} MiB budget free" +
              (Message.Length > 0 ? $" ({Message})" : "");
}

/// <param name="SimdLevel">Kernels used for plane merge / bit-depth reduction: "AVX-512", "AVX2", "SSE2" or "NEON".</param>
/// <param name="ResizeTarget">IPP dispatch target ("k0" = AVX-512, "l9" = AVX2), or stb_image_resize2's SIMD set.</param>
/// <param name="ResizeLibrary">CPU resizer: the IPP library name, or "stb_image_resize2" (ARM64).</param>
/// <param name="ResizeVersion">Its version.</param>
/// <param name="Avx512Supported">CPU and OS support AVX-512 F/CD/BW/DQ/VL.</param>
/// <param name="Avx512Active">IPP runs its AVX-512 code path.</param>
/// <param name="ForcedDispatch">IPP's default dispatch had to be overridden to get AVX-512.</param>
/// <param name="LogicalCores">Logical processors.</param>
/// <param name="Arm64">The native library is the ARM64 build.</param>
public sealed record CpuBackendInfo(
    string SimdLevel,
    string ResizeTarget,
    string ResizeLibrary,
    string ResizeVersion,
    bool Avx512Supported,
    bool Avx512Active,
    bool ForcedDispatch,
    int LogicalCores,
    bool Arm64);

public sealed record MemoryCheckResult(
    long HostRequiredBytes,
    long HostAvailableBytes,
    long DeviceRequiredBytes,
    long DeviceAvailableBytes,
    bool UsesGpu,
    bool HostOk,
    bool DeviceOk,
    int UnreadableFiles)
{
    public bool IsEnough => HostOk && DeviceOk;
    public long HostShortfallBytes => Math.Max(0, HostRequiredBytes - HostAvailableBytes);
    public long DeviceShortfallBytes => Math.Max(0, DeviceRequiredBytes - DeviceAvailableBytes);

    public override string ToString()
    {
        static string Gb(long b) => $"{b / 1073741824.0:0.00} GB";
        var s = $"RAM needed {Gb(HostRequiredBytes)}, available {Gb(HostAvailableBytes)}";
        if (UsesGpu) s += $"; VRAM needed {Gb(DeviceRequiredBytes)}, available {Gb(DeviceAvailableBytes)}";
        return s;
    }
}

/// <summary>Optional tuning. Zero means "use the default".</summary>
public sealed class PsdPreviewOptions
{
    public int JpegQuality { get; init; } = 82;
    /// <summary>Use the CPU even when a GPU is initialised.</summary>
    public bool ForceCpu { get; init; }
    public int ExtractThreads { get; init; }
    public int EncodeThreads { get; init; }
    /// <summary>Max files decoded but not yet encoded (bounds peak memory). Default 2 x cores.</summary>
    public int MaxInFlight { get; init; }
    /// <summary>
    /// mozjpeg's max-compression profile (trellis quantisation + progressive scan search): ~11%
    /// smaller files, but ~5x slower encoding, which is the pipeline's bottleneck. Off by default.
    /// </summary>
    public bool MaxCompression { get; init; }
}

/// <summary>
/// Where a batch's time went, in thread-seconds summed over each stage's threads. Divide by the
/// thread count for wall-clock utilisation. High <see cref="ExtractWaitSeconds"/> means encoding is
/// the bottleneck; high <see cref="EncodeIdleSeconds"/> means decoding/resizing is.
/// </summary>
public sealed record PsdPreviewStats(
    double DecodeSeconds,
    double ResizeSeconds,
    double EncodeSeconds,
    double ExtractWaitSeconds,
    double EncodeIdleSeconds,
    int ExtractThreads,
    int EncodeThreads,
    int GpuFallbacks,
    int GpuStreamers)
{
    public override string ToString() =>
        $"extract x{ExtractThreads}{(GpuStreamers > 0 ? $" ({GpuStreamers} GPU streamers)" : "")}: decode {DecodeSeconds:0.0}s, " +
        $"resize {ResizeSeconds:0.0}s, waiting {ExtractWaitSeconds:0.0}s | " +
        $"encode x{EncodeThreads}: encode {EncodeSeconds:0.0}s, idle {EncodeIdleSeconds:0.0}s" +
        (GpuFallbacks > 0 ? $" | {GpuFallbacks} GPU fallback(s)" : "");
}

/// <summary>One output of a batch, in the same order as the input list.</summary>
public sealed class PsdPreviewImage
{
    internal PsdPreviewImage(int index, string sourcePath, MemoryStream? jpeg, int width, int height,
                             PsdProblemKind problem, string? errorMessage, string? colorProfile, bool colorConverted)
    {
        ColorProfile = colorProfile;
        ColorConverted = colorConverted;
        Index = index;
        SourcePath = sourcePath;
        Jpeg = jpeg;
        Width = width;
        Height = height;
        Problem = problem;
        ErrorMessage = errorMessage;
    }

    public int Index { get; }
    /// <summary>The path exactly as passed in.</summary>
    public string SourcePath { get; }
    public string FileName => Path.GetFileName(SourcePath);
    /// <summary>JPEG data, positioned at 0. Null if this file failed.</summary>
    public MemoryStream? Jpeg { get; }
    public int Width { get; }
    public int Height { get; }
    public PsdProblemKind Problem { get; }
    public string? ErrorMessage { get; }
    /// <summary>Description of the embedded ICC profile, or null if the file has none.</summary>
    public string? ColorProfile { get; }
    /// <summary>True if the image was converted from its embedded profile to sRGB (lcms2).</summary>
    public bool ColorConverted { get; }
    public bool Succeeded => Jpeg != null;

    public override string ToString() =>
        Succeeded ? $"{FileName}: {Width}x{Height}, {Jpeg!.Length:N0} bytes" : $"{FileName}: {ErrorMessage}";
}

public sealed class PsdPreviewBatch : IReadOnlyList<PsdPreviewImage>, IDisposable
{
    private readonly PsdPreviewImage[] _images;

    internal PsdPreviewBatch(PsdPreviewImage[] images, string backend, bool usedGpu, TimeSpan elapsed, PsdPreviewStats stats)
    {
        _images = images;
        Stats = stats;
        Backend = backend;
        UsedGpu = usedGpu;
        Elapsed = elapsed;
        Failures = images.Where(i => !i.Succeeded)
                         .Select(i => new PsdFileProblem(i.Index, i.SourcePath, i.Problem, i.ErrorMessage ?? string.Empty))
                         .ToArray();
    }

    /// <summary>What actually ran (GPU/CPU path, threads, quality).</summary>
    public string Backend { get; }
    public bool UsedGpu { get; }
    public TimeSpan Elapsed { get; }
    /// <summary>Per-stage timings (thread-seconds).</summary>
    public PsdPreviewStats Stats { get; }
    public IReadOnlyList<PsdFileProblem> Failures { get; }
    public int SucceededCount => _images.Length - Failures.Count;

    public PsdPreviewImage this[int index] => _images[index];
    public int Count => _images.Length;
    public IEnumerator<PsdPreviewImage> GetEnumerator() => ((IEnumerable<PsdPreviewImage>)_images).GetEnumerator();
    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();

    public void Dispose()
    {
        foreach (var image in _images) image.Jpeg?.Dispose();
    }
}

public sealed class PsdPreviewException(PsdPreviewStatus status, string message) : Exception(message)
{
    public PsdPreviewStatus Status { get; } = status;
}
