// Demo / smoke test:
//   PsdPreview.Demo <dir-or-file>... [--width 720] [--cpu] [--out DIR] [--limit N] [--quality 82] [--validate-only] [--skip-validate]
using System.Diagnostics;
using PsdPreview;

Console.OutputEncoding = System.Text.Encoding.UTF8;
var inputs = new List<string>();
int width = 720, limit = int.MaxValue, quality = 82;
bool forceCpu = false, validateOnly = false, skipValidate = false;
string? outDir = null;
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--width": width = int.Parse(args[++i]); break;
        case "--limit": limit = int.Parse(args[++i]); break;
        case "--quality": quality = int.Parse(args[++i]); break;
        case "--out": outDir = args[++i]; break;
        case "--cpu": forceCpu = true; break;
        case "--validate-only": validateOnly = true; break;
        case "--skip-validate": skipValidate = true; break;
        default: inputs.Add(args[i]); break;
    }
}
if (inputs.Count == 0)
{
    Console.WriteLine("usage: PsdPreview.Demo <dir-or-file>... [--width N] [--cpu] [--out DIR] [--limit N] [--quality Q] [--validate-only]");
    return 1;
}

var files = inputs.SelectMany<string, string>(p => Directory.Exists(p)
        ? Directory.EnumerateFiles(p, "*.*", SearchOption.AllDirectories)
                   .Where(f => f.EndsWith(".psd", StringComparison.OrdinalIgnoreCase) || f.EndsWith(".psb", StringComparison.OrdinalIgnoreCase))
                   .Order(StringComparer.Ordinal)
        : [p])
    .Take(limit).ToList();

var sw = Stopwatch.StartNew();
var cuda = PsdPreviewer.InitializeCuda();
Console.WriteLine(cuda.Success
    ? $"CUDA: {cuda.DeviceName} sm_{cuda.ComputeCapability.Major}{cuda.ComputeCapability.Minor}, " +
      $"{cuda.FreeMemoryBytes / 1048576} MiB free  (init {sw.ElapsedMilliseconds} ms)"
    : $"CUDA unavailable ({cuda.Status}: {cuda.Message}) - using CPU");
var cpu = PsdPreviewer.GetCpuBackend();
Console.WriteLine($"SIMD: {cpu.SimdLevel} (plane merge / bit-depth reduction)");
Console.WriteLine($"IPP : {cpu.IppName} {cpu.IppVersion}, target {cpu.IppTarget}, AVX-512 supported={cpu.Avx512Supported} " +
                  $"active={cpu.Avx512Active} forced={cpu.ForcedDispatch}, {cpu.LogicalCores} threads");
Console.WriteLine($"{files.Count} files");

sw.Restart();
IReadOnlyList<PsdFileProblem> problems = skipValidate ? [] : PsdPreviewer.ValidateFiles(files, width);
Console.WriteLine($"Validate: {problems.Count} problem file(s) in {sw.ElapsedMilliseconds} ms");
foreach (var p in problems) Console.WriteLine($"  [{p.Kind}] {p.Path}\n      {p.Message}");
if (validateOnly) return 0;

var bad = problems.Select(p => p.Index).ToHashSet();
var good = files.Where((_, i) => !bad.Contains(i)).ToList();
var options = new PsdPreviewOptions { ForceCpu = forceCpu, JpegQuality = quality };

var mem = PsdPreviewer.CheckMemory(good, width, options);
Console.WriteLine($"Memory: {mem}  -> {(mem.IsEnough ? "OK" : "NOT ENOUGH")}");
if (!mem.IsEnough) return 2;

using var progress = new ProgressCounter();
sw.Restart();
var task = PsdPreviewer.ProcessAsync(good, width, progress, options);
while (!task.Wait(250)) Console.Write($"\r  progress {progress.Value:P1}   ");
using var batch = task.Result;
Console.WriteLine($"\r  progress {progress.Value:P1}");
Console.WriteLine($"Backend: {batch.Backend}");
Console.WriteLine($"Stages : {batch.Stats}");
Console.WriteLine($"Done: {batch.SucceededCount}/{batch.Count} ok in {batch.Elapsed.TotalSeconds:0.00} s " +
                  $"({batch.Count / batch.Elapsed.TotalSeconds:0.0} files/s), total JPEG {batch.Where(b => b.Succeeded).Sum(b => b.Jpeg!.Length) / 1048576.0:0.0} MiB");
foreach (var g in batch.Where(b => b.ColorProfile != null).GroupBy(b => (b.ColorProfile, b.ColorConverted)))
    Console.WriteLine($"  colour: {g.Count()} x \"{g.Key.ColorProfile}\" -> {(g.Key.ColorConverted ? "converted to sRGB" : "kept")}");
Console.WriteLine($"Peak working set: {Process.GetCurrentProcess().PeakWorkingSet64 / 1073741824.0:0.00} GB");
foreach (var f in batch.Failures) Console.WriteLine($"  FAILED [{f.Kind}] {f.Path}: {f.Message}");

if (outDir != null)
{
    Directory.CreateDirectory(outDir);
    foreach (var img in batch.Where(b => b.Succeeded))
    {
        var name = $"{img.Index:D4}_{Path.GetFileNameWithoutExtension(img.FileName)}.jpg";
        using var fs = File.Create(Path.Combine(outDir, name));
        img.Jpeg!.CopyTo(fs);
    }
    Console.WriteLine($"Wrote {batch.SucceededCount} JPEGs to {Path.GetFullPath(outDir)}");
}
return batch.Failures.Count == 0 ? 0 : 3;
