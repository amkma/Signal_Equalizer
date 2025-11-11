import io
import json
import wave
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict

# ---------- Basic WAV IO ----------
def read_wav(path: str) -> Tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        fs = wf.getframerate()
        n = wf.getnframes()
        ch = wf.getnchannels()
        raw = wf.readframes(n)
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if ch == 2:
            data = data.reshape(-1, 2).mean(axis=1)  # mono mixdown
        return data.copy(), fs

def write_wav(path: str, x: np.ndarray, fs: int):
    x = np.clip(x, -1.0, 1.0)
    int16 = (x * 32767.0).astype(np.int16).tobytes()
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(int16)

# ---------- Cooley–Tukey FFT (no np.fft) ----------
def _fft_recursive(x: np.ndarray) -> np.ndarray:
    N = x.shape[0]
    if N <= 1:
        return x.astype(np.complex64)
    if N % 2 != 0:
        # fallback DFT (rare when N not power of 2)
        n = np.arange(N)
        k = n.reshape((N,1))
        M = np.exp(-2j*np.pi*k*n/N)
        return (M @ x.astype(np.complex64))
    X_even = _fft_recursive(x[::2])
    X_odd  = _fft_recursive(x[1::2])
    factor = np.exp(-2j*np.pi*np.arange(N)/N)
    first = X_even + factor[:N//2]*X_odd
    second = X_even - factor[:N//2]*X_odd
    return np.concatenate([first, second])

def fft_cooley_tukey(x: np.ndarray) -> np.ndarray:
    # zero pad to next power of 2
    N = x.shape[0]
    n2 = 1<<(N-1).bit_length()
    if n2 != N:
        x = np.pad(x, (0, n2-N))
    return _fft_recursive(x.astype(np.complex64))

def ifft_cooley_tukey(X: np.ndarray, target_len: int) -> np.ndarray:
    conj = np.conjugate(X)
    inv = _fft_recursive(conj)
    x = np.conjugate(inv)/X.shape[0]
    x = np.real(x)[:target_len]
    return x.astype(np.float32)

# ---------- Spectrum & Spectrogram ----------
def mag_spectrum(x: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray]:
    X = fft_cooley_tukey(x)
    N = X.shape[0]
    freqs = np.linspace(0, fs/2, N//2, endpoint=False)
    mag = np.abs(X[:N//2]) / (N/2)
    return freqs, mag.astype(np.float32)

def audiogram_scale(freqs: np.ndarray) -> np.ndarray:
    # Simple mapping to Bark-like bins for visualization (not clinical audiogram)
    return np.log10(1 + freqs/20.0)

def stft_spectrogram(x: np.ndarray, fs: int, win_ms=25, hop_ms=10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    win = int(fs*win_ms/1000)
    hop = int(fs*hop_ms/1000)
    if win % 2 == 1: win += 1
    n_frames = 1 + max(0, (len(x)-win)//hop)
    spec = []
    for i in range(n_frames):
        s = i*hop
        frame = x[s:s+win]
        if len(frame) < win:
            frame = np.pad(frame, (0, win-len(frame)))
        frame = frame * np.hanning(win)
        X = fft_cooley_tukey(frame)
        mag = np.abs(X[:win//2]).astype(np.float32)
        spec.append(mag)
    spec = np.stack(spec, axis=1) if len(spec)>0 else np.zeros((win//2,0),dtype=np.float32)
    freqs = np.linspace(0, fs/2, win//2, endpoint=False)
    times = np.arange(n_frames)*hop/fs
    return freqs, times, spec

# ---------- Equalizer ----------
@dataclass
class Band:
    fmin: float
    fmax: float
    scale: float  # 0..2

def apply_generic_eq(x: np.ndarray, fs: int, bands: List[Band]) -> np.ndarray:
    X = fft_cooley_tukey(x)
    N = X.shape[0]
    freqs = np.linspace(0, fs, N, endpoint=False)
    for b in bands:
        # positive + negative freq bins
        mask_pos = (freqs>=b.fmin) & (freqs<b.fmax)
        mask_neg = (freqs>=fs-b.fmax) & (freqs<fs-b.fmin)
        X[mask_pos] *= b.scale
        X[mask_neg] *= b.scale
    y = ifft_cooley_tukey(X, len(x))
    return y

# ---------- Helpers ----------
def serialize_settings(obj: Dict) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

def parse_settings(text: str) -> Dict:
    return json.loads(text)
