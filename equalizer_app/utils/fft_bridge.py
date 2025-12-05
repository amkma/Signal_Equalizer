import os
import ctypes
import subprocess
import sys
import shutil
from pathlib import Path
import numpy as np

# Paths
BASE_DIR = Path(__file__).resolve().parent
CPP_SOURCE = BASE_DIR / "native_fft.cpp"
SO_FILE = BASE_DIR / ("native_fft.dll" if os.name == 'nt' else "native_fft.so")

_fft_lib = None


def _add_compiler_to_path():
    """
    Locates g++ and adds its directory to DLL search path to resolve dependencies.
    """
    compiler_path = shutil.which("g++")
    if compiler_path:
        bin_dir = str(Path(compiler_path).parent.resolve())

        # Add to PATH environment variable
        if bin_dir not in os.environ["PATH"]:
            os.environ["PATH"] += os.pathsep + bin_dir
            print(f"[FFT Bridge] Added compiler bin to PATH: {bin_dir}")

        # For Python 3.8+ on Windows, explicitly add DLL directory
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(bin_dir)
            except Exception:
                pass


def _compile_dll():
    print(f"[FFT Bridge] Compiling C++ extension at {SO_FILE}...")

    if SO_FILE.exists():
        try:
            os.remove(SO_FILE)
        except PermissionError:
            print(f"[FFT Bridge] Warning: Could not delete old DLL. Trying to overwrite.")

    if os.name == 'nt':
        # Windows: attempt static linking to avoid dependency hell
        cmd = [
            "g++", "-shared", "-o", str(SO_FILE), str(CPP_SOURCE),
            "-O3", "-static", "-static-libgcc", "-static-libstdc++"
        ]
    else:
        cmd = ["g++", "-shared", "-o", str(SO_FILE), str(CPP_SOURCE), "-fPIC", "-O3"]

    try:
        subprocess.check_call(cmd)
        print("[FFT Bridge] Compilation successful.")
    except FileNotFoundError:
        raise RuntimeError(
            "CRITICAL: C++ Compiler (g++) not found.\n"
            "Please install MinGW-w64 and ensure 'g++' is in your system PATH."
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Compilation failed with error code {e.returncode}.")


def _load_dll():
    if not SO_FILE.exists():
        raise FileNotFoundError(f"DLL file missing at {SO_FILE}")

    try:
        return ctypes.CDLL(str(SO_FILE.resolve()))
    except FileNotFoundError:
        _add_compiler_to_path()
        try:
            return ctypes.CDLL(str(SO_FILE.resolve()))
        except Exception as e:
            raise RuntimeError(f"Failed to load DLL. Error: {e}")
    except OSError as e:
        raise RuntimeError(f"OS Error loading DLL: {e}")


def compile_and_load():
    global _fft_lib
    if _fft_lib is not None:
        return _fft_lib

    _add_compiler_to_path()

    if not SO_FILE.exists() or (SO_FILE.stat().st_mtime < CPP_SOURCE.stat().st_mtime):
        _compile_dll()

    lib = _load_dll()

    # --- Define Signatures ---
    lib.fft_c.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int
    ]

    lib.apply_equalizer_c.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.POINTER(ctypes.c_float)
    ]

    lib.compute_spectrum_c.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)
    ]

    lib.stft_spectrogram_c.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
    ]

    _fft_lib = lib
    return lib


def _get_ptr(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


# ================= PUBLIC API (Calls C++) =================

def process_equalizer(signal, sr, bands):
    lib = compile_and_load()
    n = signal.size

    bands_arr = []
    for b in bands:
        bands_arr.extend([float(b['fmin']), float(b['fmax']), float(b['gain'])])

    c_bands = np.array(bands_arr, dtype=np.float32)
    c_sig = np.ascontiguousarray(signal, dtype=np.float32)
    c_out = np.zeros(n, dtype=np.float32)

    lib.apply_equalizer_c(
        _get_ptr(c_sig), n, int(sr),
        _get_ptr(c_bands), len(bands),
        _get_ptr(c_out)
    )
    return c_out


def get_spectrum_data(signal, sr, scale_type="linear"):
    lib = compile_and_load()
    n = signal.size
    VISUAL_POINTS = 2000

    c_sig = np.ascontiguousarray(signal, dtype=np.float32)
    out_mags = np.zeros(VISUAL_POINTS, dtype=np.float32)
    out_fmax = ctypes.c_float()

    scale_mode = 1 if scale_type == "audiogram" else 0

    lib.compute_spectrum_c(
        _get_ptr(c_sig), n, int(sr), VISUAL_POINTS, scale_mode,
        _get_ptr(out_mags), ctypes.byref(out_fmax)
    )
    return out_mags.tolist(), float(out_fmax.value)


def get_spectrogram_matrix(signal, sr):
    lib = compile_and_load()

    win_ms = 25
    hop_ms = 10

    # --- Exact Memory Allocation Fix ---
    # 1. Calculate window/hop in samples
    win_size = int((sr * win_ms) / 1000)
    hop_size = int((sr * hop_ms) / 1000)

    # 2. Calculate Next Power of 2 (matches C++ logic)
    nfft = 1
    while nfft < win_size:
        nfft <<= 1

    # 3. Calculate exact dimensions
    freq_bins = nfft // 2 + 1

    if len(signal) < win_size:
        num_frames = 0
    else:
        num_frames = (len(signal) - win_size) // hop_size + 1

    # 4. Allocate buffer with exact size + small safety margin
    total_elements = freq_bins * (num_frames + 5)

    c_sig = np.ascontiguousarray(signal, dtype=np.float32)
    out_spec = np.zeros(total_elements, dtype=np.float32)
    out_bins = ctypes.c_int()
    out_frames = ctypes.c_int()

    lib.stft_spectrogram_c(
        _get_ptr(c_sig), len(signal), int(sr), win_ms, hop_ms,
        _get_ptr(out_spec), ctypes.byref(out_bins), ctypes.byref(out_frames)
    )

    f, t = out_bins.value, out_frames.value

    if f <= 0 or t <= 0:
        return np.array([]), np.array([]), np.zeros((0, 0))

    # Reshape the valid portion of the linear buffer to 2D
    spec = out_spec[:f * t].reshape((f, t))

    freqs = np.linspace(0, sr / 2, f)
    times = np.arange(t) * (hop_ms / 1000.0)

    return freqs, times, spec