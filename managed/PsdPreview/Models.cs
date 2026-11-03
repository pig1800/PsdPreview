namespace PsdPreview;

/// <summary>Return codes of the native API.</summary>
public enum PsdPreviewStatus
{
    Ok = 0,
    InvalidArgument = 1,
    CudaUnavailable = 2,
    CudaFailed = 3,
    NppUnavailable = 4,
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

public sealed record CudaInitResult(
    PsdPreviewStatus Status,
    string Message,
    int DeviceId,
    string DeviceName,
    Version ComputeCapability,
    long TotalMemoryBytes,
    long FreeMemoryBytes,
    Version NppVersion,
    int CudaRuntimeVersion,
    int CudaDriverVersion)
{
    public bool Success => Status == PsdPreviewStatus.Ok;
}

/// <param name="SimdLevel">Kernels used for plane merge / bit-depth reduction: "AVX-512", "AVX2" or "SSE2".</param>
/// <param name="IppTarget">IPP dispatch target: "k0" = AVX-512, "l9" = AVX2.</param>
/// <param name="IppName">IPP library name.</param>
/// <param name="IppVersion">IPP version string.</param>
/// <param name="Avx512Supported">CPU and OS support AVX-512 F/CD/BW/DQ/VL.</param>
/// <param name="Avx512Active">IPP runs its AVX-512 code path.</param>
/// <param name="ForcedDispatch">IPP's default dispatch had to be overridden to get AVX-512.</param>
/// <param name="LogicalCores">Logical processors.</param>
public sealed record CpuBackendInfo(
    string SimdLevel,
    string IppTarget,
    string IppName,
    string IppVersion,
    bool Avx512Supported,
    bool Avx512Active,
    bool ForcedDispatch,
    int LogicalCores);

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
    /// <summary>Use IPP even when CUDA is initialised.</summary>
    public bool ForceCpu { get; init; }
    public int ExtractThreads { get; init; }
    public int EncodeThreads { get; init; }
    /// <summary>Max files decoded but not yet encoded (bounds peak memory). Default 2 x cores.</summary>
    public int MaxInFlight { get; init; }
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
    int GpuFallbacks)
{
    public override string ToString() =>
        $"extract x{ExtractThreads}: decode {DecodeSeconds:0.0}s, resize {ResizeSeconds:0.0}s, waiting {ExtractWaitSeconds:0.0}s | " +
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
