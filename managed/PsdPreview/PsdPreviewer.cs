using PsdPreview.Interop;

namespace PsdPreview;

/// <summary>
/// Extracts the flattened composite of PSD/PSB files, resizes it to a target width
/// (antialiased bicubic matching Photoshop's "Bicubic Sharper"; CUDA when initialised, Intel IPP
/// otherwise) and encodes JPEG (mozjpeg).
/// </summary>
/// <remarks>
/// Typical use:
/// <code>
/// PsdPreviewer.InitializeCuda();                              // once, at startup
/// var problems = PsdPreviewer.ValidateFiles(files);          // show + stop if any
/// var mem = PsdPreviewer.CheckMemory(files, 720);            // show mem.ToString() if !mem.IsEnough
/// using var progress = new ProgressCounter();                // poll progress.Value from the UI
/// using var batch = await PsdPreviewer.ProcessAsync(files, 720, progress);
/// foreach (var img in batch) Save(img.FileName, img.Jpeg);   // img.SourcePath = input path
/// </code>
/// </remarks>
public static unsafe class PsdPreviewer
{
    /// <summary>
    /// Initialises CUDA and the resize kernel on the given device. Call once when the program starts.
    /// On failure the library keeps working on the CPU (IPP) path; the result says why.
    /// </summary>
    public static CudaInitResult InitializeCuda(int deviceId = 0)
    {
        NativeDeviceInfo info;
        var status = (PsdPreviewStatus)Native.pp_cuda_init(deviceId, &info);
        return new CudaInitResult(
            status,
            Native.Str(info.Message),
            info.DeviceId,
            Native.Str(info.Name),
            new Version(info.ComputeMajor, info.ComputeMinor),
            (long)info.TotalMemory,
            (long)info.FreeMemory,
            info.CudaRuntimeVersion,
            info.CudaDriverVersion);
    }

    public static bool IsCudaReady => Native.pp_cuda_ready() != 0;

    /// <summary>Stops using the GPU for subsequent batches (the CUDA context is left alone).</summary>
    public static void ShutdownCuda() => Native.pp_cuda_shutdown();

    /// <summary>IPP dispatch information (e.g. "k0" = AVX-512 code path).</summary>
    public static CpuBackendInfo GetCpuBackend()
    {
        NativeCpuInfo info;
        Check(Native.pp_cpu_info(&info), nameof(GetCpuBackend));
        return new CpuBackendInfo(Native.Str(info.SimdLevel), Native.Str(info.IppTarget), Native.Str(info.IppName),
                                  Native.Str(info.IppVersion),
                                  info.Avx512Supported != 0, info.Avx512Active != 0, info.ForcedDispatch != 0,
                                  info.LogicalCores);
    }

    /// <summary>
    /// Checks every file can be processed. Returns an empty list when all are fine,
    /// otherwise one entry per problematic file. Pass <paramref name="targetWidth"/> to
    /// also check that the resized image fits the JPEG size limit.
    /// </summary>
    public static IReadOnlyList<PsdFileProblem> ValidateFiles(IReadOnlyList<string> files, int targetWidth = 0)
    {
        ArgumentNullException.ThrowIfNull(files);
        if (files.Count == 0) return Array.Empty<PsdFileProblem>();
        using var paths = new NativePathList(files);
        var issues = new NativeIssue[files.Count];
        int found;
        fixed (NativeIssue* p = issues)
            Check(Native.pp_validate(paths.Pointer, files.Count, targetWidth, p, &found), nameof(ValidateFiles));
        var result = new PsdFileProblem[found];
        for (int i = 0; i < found; i++)
        {
            fixed (char* msg = issues[i].Message)
                result[i] = new PsdFileProblem(issues[i].Index, files[issues[i].Index], (PsdProblemKind)issues[i].Code,
                                               Native.Str(msg));
        }
        return result;
    }

    /// <summary>
    /// Estimates peak RAM (and VRAM when the GPU path will be used) for processing
    /// <paramref name="files"/> at <paramref name="targetWidth"/>, and compares with what is free now.
    /// </summary>
    public static MemoryCheckResult CheckMemory(IReadOnlyList<string> files, int targetWidth, PsdPreviewOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(files);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(targetWidth);
        using var paths = new NativePathList(files);
        var opt = ToNative(options);
        NativeMemoryReport r;
        Check(Native.pp_check_memory(paths.Pointer, files.Count, targetWidth, &opt, &r), nameof(CheckMemory));
        return new MemoryCheckResult((long)r.HostRequired, (long)r.HostAvailable, (long)r.DeviceRequired,
                                     (long)r.DeviceAvailable, r.UsesGpu != 0, r.HostOk != 0, r.DeviceOk != 0,
                                     r.UnreadableFiles);
    }

    /// <summary>
    /// Processes the batch (blocking). The result has one entry per input file, in input
    /// order, each carrying its source path and JPEG stream (null + error for failures).
    /// </summary>
    public static PsdPreviewBatch Process(IReadOnlyList<string> files, int targetWidth, ProgressCounter? progress = null,
                                          PsdPreviewOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(files);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(targetWidth);
        using var paths = new NativePathList(files);
        var opt = ToNative(options);
        nint batch = 0;
        double* progressCell = progress != null ? progress.Acquire() : null;
        try
        {
            Check(Native.pp_process(paths.Pointer, files.Count, targetWidth, &opt, progressCell, &batch), nameof(Process));
        }
        finally
        {
            progress?.Release();
        }

        try
        {
            NativeBatchInfo info;
            Check(Native.pp_batch_info(batch, &info), nameof(Process));
            var images = new PsdPreviewImage[files.Count];
            for (int i = 0; i < images.Length; i++)
            {
                NativeItem item;
                Check(Native.pp_batch_item(batch, i, &item), nameof(Process));
                MemoryStream? jpeg = null;
                if (item.Data != null && item.Size > 0)
                {
                    var bytes = GC.AllocateUninitializedArray<byte>(checked((int)item.Size));
                    new ReadOnlySpan<byte>(item.Data, bytes.Length).CopyTo(bytes);
                    jpeg = new MemoryStream(bytes, 0, bytes.Length, writable: false, publiclyVisible: true);
                }
                string? error = item.Code == 0 ? null : Native.Str(item.Message);
                string profile = Native.Str(item.ColorProfile);
                Native.pp_batch_release_item(batch, i); // free native copy right away: peak stays ~1x results
                images[i] = new PsdPreviewImage(i, files[i], jpeg, item.Width, item.Height, (PsdProblemKind)item.Code, error,
                                                profile.Length == 0 ? null : profile, item.ColorConverted != 0);
            }
            var stats = new PsdPreviewStats(info.DecodeSeconds, info.ResizeSeconds, info.EncodeSeconds,
                                            info.ExtractWaitSeconds, info.EncodeIdleSeconds, info.ExtractThreads,
                                            info.EncodeThreads, info.GpuFallbacks);
            return new PsdPreviewBatch(images, Native.Str(info.Backend), info.UsedGpu != 0,
                                       TimeSpan.FromSeconds(info.ElapsedSeconds), stats);
        }
        finally
        {
            Native.pp_batch_free(batch);
        }
    }

    /// <summary><see cref="Process"/> on a thread-pool thread.</summary>
    public static Task<PsdPreviewBatch> ProcessAsync(IReadOnlyList<string> files, int targetWidth,
                                                     ProgressCounter? progress = null, PsdPreviewOptions? options = null) =>
        Task.Factory.StartNew(() => Process(files, targetWidth, progress, options), CancellationToken.None,
                              TaskCreationOptions.LongRunning, TaskScheduler.Default);

    private static NativeOptions ToNative(PsdPreviewOptions? o) => o == null
        ? new NativeOptions { JpegQuality = 82 }
        : new NativeOptions
        {
            JpegQuality = o.JpegQuality,
            ForceCpu = o.ForceCpu ? 1 : 0,
            ExtractThreads = o.ExtractThreads,
            EncodeThreads = o.EncodeThreads,
            MaxInFlight = o.MaxInFlight,
        };

    private static void Check(int status, string operation)
    {
        if (status != 0)
            throw new PsdPreviewException((PsdPreviewStatus)status, $"{operation} failed: {(PsdPreviewStatus)status}");
    }
}
