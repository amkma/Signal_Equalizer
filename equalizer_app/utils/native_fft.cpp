#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern "C" {

// --- Helper: Bit Reversal for FFT ---
void bit_reverse_copy(const std::complex<float>* src, std::complex<float>* dst, int n) {
    int bits = 0;
    while ((1 << bits) < n) bits++;

    for (int i = 0; i < n; i++) {
        int rev = 0;
        int val = i;
        for (int j = 0; j < bits; j++) {
            rev = (rev << 1) | (val & 1);
            val >>= 1;
        }
        if (rev < n) dst[rev] = src[i];
    }
}

// --- Core FFT (Iterative Cooley-Tukey) ---
// Used internally by all other functions
void fft_core(std::complex<float>* x, int n, bool inverse) {
    // Reorder array by bit-reversal
    std::vector<std::complex<float>> temp(n);
    bit_reverse_copy(x, temp.data(), n);
    for(int i=0; i<n; i++) x[i] = temp[i];

    for (int len = 2; len <= n; len <<= 1) {
        float ang = 2 * M_PI / len * (inverse ? 1 : -1);
        std::complex<float> wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            std::complex<float> w(1);
            for (int j = 0; j < len / 2; j++) {
                std::complex<float> u = x[i + j];
                std::complex<float> v = x[i + j + len / 2] * w;
                x[i + j] = u + v;
                x[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    if (inverse) {
        for (int i = 0; i < n; i++) x[i] /= n;
    }
}

// --- Exposed: Simple FFT/IFFT ---
void fft_c(const float* in_real, const float* in_imag, float* out_real, float* out_imag, int n, int inverse) {
    std::vector<std::complex<float>> data(n);
    for (int i = 0; i < n; i++) {
        data[i] = std::complex<float>(in_real[i], in_imag ? in_imag[i] : 0.0f);
    }

    fft_core(data.data(), n, inverse);

    for (int i = 0; i < n; i++) {
        out_real[i] = data[i].real();
        out_imag[i] = data[i].imag();
    }
}

// --- Helper: Hanning Window ---
void apply_hanning_window(float* data, int n) {
    for (int i = 0; i < n; i++) {
        float multiplier = 0.5f * (1.0f - cos(2.0f * M_PI * i / (n - 1)));
        data[i] *= multiplier;
    }
}

// --- Exposed: Full Equalizer Processing ---
// Performs: FFT -> Frequency Masking -> Gain Application -> IFFT
// bands_data: flattened array of [fmin, fmax, gain, fmin, fmax, gain, ...]
void apply_equalizer_c(const float* input_signal, int n, int sr,
                      const float* bands_data, int num_bands,
                      float* out_signal) {

    // 1. Prepare Complex Data (Zero Padding to Power of 2 usually handled in Python,
    //    but assuming n is valid power of 2 here for simplicity or exact size)
    std::vector<std::complex<float>> X(n);
    for(int i=0; i<n; i++) X[i] = std::complex<float>(input_signal[i], 0.0f);

    // 2. Forward FFT
    fft_core(X.data(), n, false);

    // 3. Apply Gains
    // FFT Frequencies: k * sr / n
    float freq_step = (float)sr / n;

    for (int i = 0; i < n; i++) {
        float freq = i * freq_step;
        // Handle symmetry for frequencies > sr/2
        if (freq > sr/2.0f) freq = sr - freq;

        // Check bands
        float gain = 1.0f;
        for (int b = 0; b < num_bands; b++) {
            float fmin = bands_data[b*3 + 0];
            float fmax = bands_data[b*3 + 1];
            float g    = bands_data[b*3 + 2];

            if (freq >= fmin && freq <= fmax) {
                // Multiplicative gain? Or replacement? Standard EQ multiplies.
                // If multiple bands overlap, gains multiply (series) or accumulate.
                // We'll assume simple multiplication for active bands.
                gain *= g;
            }
        }
        X[i] *= gain;
    }

    // 4. Inverse FFT
    fft_core(X.data(), n, true);

    // 5. Real Output
    for(int i=0; i<n; i++) {
        out_signal[i] = X[i].real();
    }
}

// --- Exposed: Compute Magnitude Spectrum (for Visualization) ---
// Performs: Window -> FFT -> Abs -> Normalize -> Downsample (simple binning)
void compute_spectrum_c(const float* input_signal, int n, int sr, int output_points,
                        int scale_type, float* out_mags, float* out_fmax) {

    // Copy and Window
    std::vector<std::complex<float>> X(n);
    for(int i=0; i<n; i++) {
        float val = input_signal[i];
        // Inline Hanning
        float win = 0.5f * (1.0f - cos(2.0f * M_PI * i / (n - 1)));
        X[i] = std::complex<float>(val * win, 0.0f);
    }

    // FFT
    fft_core(X.data(), n, false);

    // Compute Magnitudes (first half)
    int half_n = n / 2;
    std::vector<float> mags(half_n);
    float max_val = 0.0f;

    // Remove DC
    mags[0] = 0.0f;

    for(int i=1; i<half_n; i++) {
        float m = std::abs(X[i]);
        if (m > max_val) max_val = m;
        mags[i] = m;
    }

    // Normalize and Scale
    if (max_val < 1e-9) max_val = 1e-9;

    for(int i=0; i<half_n; i++) {
        float val = mags[i] / max_val;

        if (scale_type == 1) { // Audiogram / Log-ish
            // Simple simulation of dB-like scaling mapped 0-1
            val = 20.0f * log10(val + 1e-9); // dB
            float min_db = -80.0f;
            val = (val - min_db) / (0.0f - min_db);
            if (val < 0) val = 0;
            if (val > 1) val = 1;
        } else {
            // Square root compression (Linear visual preference)
            val = sqrt(val);
        }
        mags[i] = val;
    }

    // Downsample to output_points (simple averaging)
    int step = half_n / output_points;
    if (step < 1) step = 1;

    for(int i=0; i<output_points && (i*step < half_n); i++) {
        float sum = 0.0f;
        int count = 0;
        for(int j=0; j<step && (i*step + j < half_n); j++) {
            sum += mags[i*step + j];
            count++;
        }
        out_mags[i] = (count > 0) ? sum / count : 0.0f;
    }

    *out_fmax = (float)sr / 2.0f;
}

// --- Exposed: STFT Spectrogram ---
void stft_spectrogram_c(const float* signal, int signal_len, int sr, int win_ms, int hop_ms,
                        float* out_spec, int* out_freq_bins, int* out_frames) {

    int win_size = (sr * win_ms) / 1000;
    int hop_size = (sr * hop_ms) / 1000;

    // Determine FFT size (next power of 2)
    int nfft = 1;
    while (nfft < win_size) nfft <<= 1;

    int num_frames = (signal_len - win_size) / hop_size + 1;
    if (num_frames < 0) num_frames = 0;

    int freq_bins = nfft / 2 + 1;

    // Set output dimensions for Python
    *out_freq_bins = freq_bins;
    *out_frames = num_frames;

    if (num_frames == 0) return;

    // Precompute window
    std::vector<float> window(win_size);
    for(int i=0; i<win_size; i++) {
        window[i] = 0.5f * (1.0f - cos(2.0f * M_PI * i / (win_size - 1)));
    }

    std::vector<std::complex<float>> fft_buf(nfft);

    // Processing loop
    for (int t = 0; t < num_frames; t++) {
        int start = t * hop_size;

        // Windowing & Zero Pad
        for(int i=0; i<nfft; i++) {
            if (i < win_size && (start + i) < signal_len) {
                fft_buf[i] = std::complex<float>(signal[start + i] * window[i], 0.0f);
            } else {
                fft_buf[i] = std::complex<float>(0.0f, 0.0f);
            }
        }

        // FFT
        fft_core(fft_buf.data(), nfft, false);

        // Store Magnitude
        for (int f = 0; f < freq_bins; f++) {
            // output is flattened [freq_bins * num_frames]
            // We store column-major or row-major? Python expects 2D.
            // Let's fill linearly: frame 0 all freqs, frame 1 all freqs...
            // Python reshape logic depends on this.
            // Usually: out_spec[f * num_frames + t] -> (freq, time)
            out_spec[f * num_frames + t] = std::abs(fft_buf[f]);
        }
    }
}

}