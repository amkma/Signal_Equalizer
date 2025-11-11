# equalizer_app/utils/fft.py
import numpy as np
import math

def _bit_reverse_indices(n):
    j = 0
    for i in range(n):
        yield j
        m = n >> 1
        while m and j & m:
            j ^= m
            m >>= 1
        j |= m

def fft_radix2(x: np.ndarray):
    """Cooley–Tukey radix-2 DIT FFT. x: real/complex, length power of 2."""
    n = int(len(x))
    X = np.array(x, dtype=np.complex64, copy=True)
    # bit-reversal
    idx = list(_bit_reverse_indices(n))
    X[:] = X[idx]

    m = 2
    while m <= n:
        half = m // 2
        theta = -2.0 * math.pi / m
        w_m = complex(math.cos(theta), math.sin(theta))
        for k in range(0, n, m):
            w = 1+0j
            for j in range(half):
                t = w * X[k + j + half]
                u = X[k + j]
                X[k + j] = u + t
                X[k + j + half] = u - t
                w *= w_m
        m <<= 1
    return X

def ifft_radix2(X: np.ndarray):
    """Inverse FFT via conjugate trick using our own fft."""
    Xc = np.conjugate(X)
    xc = fft_radix2(Xc)
    x = np.conjugate(xc) / len(X)
    return x

def stft_spectrogram(x: np.ndarray, sr: int, win_ms=25, hop_ms=10):
    """Return magnitude spectrogram (freq x time) as float array [0..1]."""
    win = int(sr * win_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    nfft = 1
    while nfft < win:
        nfft <<= 1

    # Hamming window
    w = 0.54 - 0.46 * np.cos(2*np.pi*np.arange(win)/(win-1))
    frames = []
    for start in range(0, len(x) - win, hop):
        seg = x[start:start+win] * w
        pad = np.zeros(nfft, dtype=np.float32)
        pad[:win] = seg
        X = fft_radix2(pad)
        mag = np.abs(X[:nfft//2])
        frames.append(mag)
    if not frames:
        return np.zeros((nfft//2, 1), dtype=np.float32)
    S = np.stack(frames, axis=1).astype(np.float32)
    # log scale for visibility
    S = np.log1p(S)
    # normalize per-whole
    S -= S.min()
    S /= (S.max() + 1e-12)
    return S
