import os
import ctypes
import numpy as np
import subprocess
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
CPP_SOURCE = BASE_DIR / "native_fft.cpp"
SO_FILE = BASE_DIR / "native_fft.so"  # .dll on Windows

_fft_lib = None


def compile_cpp():
    """Attempts to compile the C++ FFT library."""
    global _fft_lib

    # Determine compiler and flags based on OS
    if os.name == 'nt':
        # Windows instruction (requires MinGW or MSVC)
        # Simplified for Linux/Mac environment usually found in Python deployments
        cmd = ["g++", "-shared", "-o", str(SO_FILE), str(CPP_SOURCE), "-fPIC", "-O3"]
    else:
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