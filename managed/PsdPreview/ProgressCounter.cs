using System.Runtime.InteropServices;

namespace PsdPreview;

/// <summary>
/// Progress of a running batch, 0..1, written by the native pipeline. Poll
/// <see cref="Value"/> from any thread (e.g. a UI timer). Steps = files x 3
/// (extract, resize, encode). Reusable across batches.
/// </summary>
public sealed unsafe class ProgressCounter : IDisposable
{
    private readonly object _gate = new();
    private double* _value;
    private int _users;
    private bool _disposeRequested;

    public ProgressCounter()
    {
        _value = (double*)NativeMemory.AlignedAlloc(sizeof(double), sizeof(double));
        *_value = 0;
    }

    public double Value
    {
        get
        {
            lock (_gate) return _value == null ? 0 : Volatile.Read(ref *_value);
        }
    }

    /// <summary>Pins the native cell for the duration of a native call.</summary>
    internal double* Acquire()
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_value == null || _disposeRequested, this);
            _users++;
            return _value;
        }
    }

    internal void Release()
    {
        lock (_gate)
        {
            if (--_users == 0 && _disposeRequested) Free();
        }
    }

    public void Dispose()
    {
        lock (_gate)
        {
            _disposeRequested = true;
            if (_users == 0) Free();
        }
        GC.SuppressFinalize(this);
    }

    ~ProgressCounter() => Free();

    private void Free()
    {
        if (_value != null)
        {
            NativeMemory.AlignedFree(_value);
            _value = null;
        }
    }
}
