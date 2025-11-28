import os
import ctypes
import numpy as np
import subprocess
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
CPP_SOURCE = BASE_DIR / "native_fft.cpp"
# Use .dll on Windows, .so on Linux/Mac
SO_FILE = BASE_DIR / ("native_fft.dll" if os.name == 'nt' else "native_fft.so")

_fft_lib = None


def compile_cpp():
    """Attempts to compile the C++ FFT library."""
    global _fft_lib

    # Determine compiler and output extension based on OS
    if os.name == 'nt':
        # Windows: compile to DLL
        cmd = ["g++", "-shared", "-o", str(SO_FILE), str(CPP_SOURCE), "-O3", "-static-libgcc", "-static-libstdc++"]
    else:
        # Linux/Mac: compile to .so
        cmd = ["g++", "-shared", "-o", str(SO_FILE), str(CPP_SOURCE), "-fPIC", "-O3"]

    try:
        if not SO_FILE.exists() or SO_FILE.stat().st_mtime < CPP_SOURCE.stat().st_mtime:
            print(f"[FFT Bridge] Compiling C++ extension: {' '.join(cmd)}")
            subprocess.check_call(cmd)
            print("[FFT Bridge] Compilation successful.")
    except Exception as e:
        print(f"[FFT Bridge] Warning: Could not compile C++ FFT. Error: {e}")
        return None

    try:
        lib = ctypes.CDLL(str(SO_FILE))
        # void fft_c(const float* in_real, const float* in_imag, float* out_real, float* out_imag, int n, int inverse)
        lib.fft_c.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # in_real
            ctypes.POINTER(ctypes.c_float),  # in_imag
            ctypes.POINTER(ctypes.c_float),  # out_real
            ctypes.POINTER(ctypes.c_float),  # out_imag
            ctypes.c_int,  # n
            ctypes.c_int  # inverse (0 or 1)
        ]
        
        # void stft_spectrogram(const float* signal, int signal_len, int sr, int win_ms, int hop_ms,
        #                       float* out_spec, int* out_freq_bins, int* out_frames)
        lib.stft_spectrogram.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # signal
            ctypes.c_int,                     # signal_len
            ctypes.c_int,                     # sr
            ctypes.c_int,                     # win_ms
            ctypes.c_int,                     # hop_ms
            ctypes.POINTER(ctypes.c_float),  # out_spec
            ctypes.POINTER(ctypes.c_int),    # out_freq_bins
            ctypes.POINTER(ctypes.c_int)     # out_frames
        ]
        
        return lib
    except OSError as e:
        print(f"[FFT Bridge] Failed to load shared library: {e}")
        return None


def fft_cpp(x: np.ndarray) -> np.ndarray:
    """
    Computes FFT using the C++ backend.
    Input: 1D numpy array (float or complex)
    Output: 1D numpy array (complex)
    """
    global _fft_lib
    if _fft_lib is None:
        _fft_lib = compile_cpp()

    # Fallback to numpy if C++ failed to load
    if _fft_lib is None:
        return np.fft.fft(x)

    n = x.size

    # Prepare input buffers
    if np.iscomplexobj(x):
        in_real = np.ascontiguousarray(x.real, dtype=np.float32)
        in_imag = np.ascontiguousarray(x.imag, dtype=np.float32)
    else:
        in_real = np.ascontiguousarray(x, dtype=np.float32)
        in_imag = np.zeros(n, dtype=np.float32)

    # Prepare output buffers
    out_real = np.zeros(n, dtype=np.float32)
    out_imag = np.zeros(n, dtype=np.float32)

    # Call C++
    _fft_lib.fft_c(
        in_real.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        in_imag.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_real.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_imag.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(n),
        ctypes.c_int(0)  # 0 for Forward FFT
    )

    return out_real + 1j * out_imag


def ifft_cpp(x: np.ndarray) -> np.ndarray:
    """
    Computes IFFT using the C++ backend.
    Input: 1D numpy array (complex)
    Output: 1D numpy array (complex)
    """
    global _fft_lib
    if _fft_lib is None:
        _fft_lib = compile_cpp()

    # Fallback to numpy if C++ failed to load
    if _fft_lib is None:
        return np.fft.ifft(x)

    n = x.size

    # Prepare input buffers
    if np.iscomplexobj(x):
        in_real = np.ascontiguousarray(x.real, dtype=np.float32)
        in_imag = np.ascontiguousarray(x.imag, dtype=np.float32)
    else:
        in_real = np.ascontiguousarray(x, dtype=np.float32)
        in_imag = np.zeros(n, dtype=np.float32)

    # Prepare output buffers
    out_real = np.zeros(n, dtype=np.float32)
    out_imag = np.zeros(n, dtype=np.float32)

    # Call C++ with inverse=1
    _fft_lib.fft_c(
        in_real.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        in_imag.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_real.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_imag.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(n),
        ctypes.c_int(1)  # 1 for Inverse FFT
    )

    return out_real + 1j * out_imag


def fftfreq(n: int, d: float = 1.0) -> np.ndarray:
    """
    Return the Discrete Fourier Transform sample frequencies.
    
    Args:
        n: Window length
        d: Sample spacing (inverse of sampling rate)
    
    Returns:
        Array of length n containing sample frequencies
    """
    return np.fft.fftfreq(n, d)


def stft_spectrogram(x: np.ndarray, sr: int, win_ms: int = 25, hop_ms: int = 10) -> np.ndarray:
    """
    Compute STFT spectrogram using C++ backend.
    
    Args:
        x: Input signal (1D array)
        sr: Sample rate
        win_ms: Window size in milliseconds
        hop_ms: Hop size in milliseconds
    
    Returns:
        2D array (freq_bins, time_frames) containing magnitude spectrogram
    """
    global _fft_lib
    if _fft_lib is None:
        _fft_lib = compile_cpp()
    
    if _fft_lib is None:
        # Fallback to numpy
        win_samples = int((sr * win_ms) / 1000)
        hop_samples = int((sr * hop_ms) / 1000)
        nfft = 1
        while nfft < win_samples:
            nfft <<= 1
        
        window = np.hanning(win_samples)
        num_frames = (len(x) - win_samples) // hop_samples + 1
        freq_bins = nfft // 2 + 1
        
        spec = np.zeros((freq_bins, num_frames), dtype=np.float32)
        for i in range(num_frames):
            start = i * hop_samples
            frame = x[start:start + win_samples] * window
            frame_padded = np.zeros(nfft, dtype=np.float32)
            frame_padded[:len(frame)] = frame
            fft_result = np.fft.fft(frame_padded)
            spec[:, i] = np.abs(fft_result[:freq_bins])
        return spec
    
    signal = np.ascontiguousarray(x, dtype=np.float32)
    signal_len = len(signal)
    
    # Estimate output size
    win_samples = (sr * win_ms) // 1000
    hop_samples = (sr * hop_ms) // 1000
    nfft = 1
    while nfft < win_samples:
        nfft <<= 1
    
    num_frames = (signal_len - win_samples) // hop_samples + 1
    freq_bins = nfft // 2 + 1
    
    out_spec = np.zeros(freq_bins * num_frames, dtype=np.float32)
    out_freq_bins = ctypes.c_int()
    out_frames = ctypes.c_int()
    
    _fft_lib.stft_spectrogram(
        signal.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(signal_len),
        ctypes.c_int(sr),
        ctypes.c_int(win_ms),
        ctypes.c_int(hop_ms),
        out_spec.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(out_freq_bins),
        ctypes.byref(out_frames)
    )
    
    # Reshape to (freq_bins, time_frames)
    return out_spec.reshape(out_freq_bins.value, out_frames.value)


# Public API
fft = fft_cpp
ifft = ifft_cpp