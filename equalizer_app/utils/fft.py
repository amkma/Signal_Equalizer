# equalizer_app/utils/fft.py
import numpy as np
import math


def fft_radix2(x: np.ndarray):
    """
    Vectorized Recursive Cooley-Tukey Radix-2 FFT.
    Optimized for speed using NumPy broadcasting instead of Python loops.
    Supports 1D arrays (signal) or 2D batches (spectrogram frames).
    """
    # Ensure complex type
    x = np.asarray(x, dtype=np.complex64)
    n = x.shape[-1]

    # Base case
    if n <= 1:
        return x
    # Logic (Recursion): The "divide and conquer" strategy keeps splitting the array
    # until it has only 1 sample. The Fourier Transform of a
    # single point is just the point itself.


    # Handle cases where N is not a power of 2 (padding)
    if n & (n - 1) != 0: # n & (n - 1) != 0 is a bitwise trick to check if n is NOT a power of 2
        target = 1
        while target < n: target <<= 1
        pad_width = [(0, 0)] * (x.ndim - 1) + [(0, target - n)]
        x = np.pad(x, pad_width, mode='constant')
        n = target
    # If it isn't, the code calculates the next power of 2
    # (target) and pads the signal with zeros (silence) at the end.


    # Recursive divide (Vectorized)
    # Instead of looping, we split the array using slicing
    even = fft_radix2(x[..., ::2])
    odd = fft_radix2(x[..., 1::2])
    #Logic: This is the core of Cooley-Tukey.
    #Even: Takes indices 0, 2, 4...
    # Odd: Takes indices 1, 3, 5...
    # It recursively calls fft_radix2 on these halves.
    # This keeps happening until they hit the base case (N=1).


    # Vectorized Twiddle factors
    # factor = exp(-2j * pi * k / N)
    k = np.arange(n // 2)
    factor = np.exp(-2j * np.pi * k / n)
    #This calculates WkN=e−i2πk / N
    #This is a complex number of magnitude 1
    #It represents a rotation in the complex plane.
    #It aligns the "Odd" samples so they can be mathematically
    #combined with the "Even" samples to form the frequencies.

    # Reshape factor for broadcasting if input is 2D (e.g. spectrogram frames)
    if x.ndim > 1:
        # Reshape to (1, 1, ..., n//2) to broadcast across frames
        new_shape = [1] * (x.ndim - 1) + [n // 2]
        factor = factor.reshape(new_shape)
        #Logic: This allows the FFT to run on multiple audio frames
        # (Spectrograms) simultaneously for speed, rather than looping one by one.

    # Butterfly operation (Vectorized)
    return np.concatenate([even + factor * odd, even - factor * odd], axis=-1)
    #This combines the results of the recursive splits.
    #First half of frequencies (0 to N/2): Even + (Twiddle * Odd)
    #Second half of frequencies (N/2 to N): Even - (Twiddle * Odd)
    #The array returned is now in the Frequency Domain. Index 0 is 0Hz (DC offset),
    #and higher indices represent higher pitches.

def ifft_radix2(X: np.ndarray):
    """
    Inverse FFT via conjugate trick using our vectorized fft.
    x = conj(FFT(conj(X))) / N
    We need to get from Frequency Domain (Eq modified)
    back to Time Domain (Audio we can hear).
    """
    Xc = np.conjugate(X)
    xc = fft_radix2(Xc)
    x = np.conjugate(xc) / X.shape[-1]
    return x
#Instead of writing a separate Inverse algorithm, we use a mathematical property:
#IFFT(X)=(1/N)*conj(FFT(conj(X)))

def stft_spectrogram(x: np.ndarray, sr: int, win_ms=25, hop_ms=10):
    """
    High-performance Vectorized STFT spectrogram.
    Processes all frames in a single batch FFT call.
    """
    # 1. Calc parameters
    win = int(sr * win_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    nfft = 1
    while nfft < win:
        nfft <<= 1

    # 2. Vectorized Framing
    if len(x) < win:
        return np.zeros((nfft // 2, 1), dtype=np.float32)

    n_frames = (len(x) - win) // hop + 1

    # Create an index matrix to slice all frames at once (no loops)
    # shape: (n_frames, win)
    frame_indices = np.tile(np.arange(win), (n_frames, 1)) + np.arange(n_frames)[:, None] * hop

    try:
        frames = x[frame_indices]  # Shape: (n_frames, win)
    except IndexError:
        # Fallback for edge cases
        return np.zeros((nfft // 2, 1), dtype=np.float32)

    # 3. Apply Hamming Window (Vectorized)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(win) / (win - 1))
    frames = frames * w

    # 4. Zero-pad frames to nfft
    if nfft > win:
        frames_padded = np.zeros((n_frames, nfft), dtype=np.float32)
        frames_padded[:, :win] = frames
    else:
        frames_padded = frames

    # 5. Batch FFT
    # Pass the entire 2D matrix to our vectorized FFT
    # It will process the last axis (time samples) automatically
    X = fft_radix2(frames_padded)

    # 6. Magnitude & Log Scale
    # Take first half (Nyquist) and transpose to (freq, time)
    mag = np.abs(X[:, :nfft // 2]).T

    mag = np.log1p(mag)  # Log scale for visibility

    # Normalize [0..1]
    m_min = mag.min()
    m_max = mag.max()
    if m_max - m_min > 1e-12:
        mag = (mag - m_min) / (m_max - m_min)
    else:
        mag -= m_min

    return mag.astype(np.float32)