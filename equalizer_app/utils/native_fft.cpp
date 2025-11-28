/* equalizer_app/utils/native_fft.cpp */
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif

#include <cmath>
#include <complex>
#include <vector>

const double PI = 3.141592653589793238460;

extern "C" {

    // Recursive Cooley-Tukey FFT
    void fft_recursive(std::complex<float>* data, int n, int inverse) {
        if (n <= 1) return;
        
        int half = n / 2;
        std::vector<std::complex<float>> even(half), odd(half);
        
        for (int i = 0; i < half; i++) {
            even[i] = data[i * 2];
            odd[i] = data[i * 2 + 1];
        }
        
        fft_recursive(even.data(), half, inverse);
        fft_recursive(odd.data(), half, inverse);
        
        for (int k = 0; k < half; k++) {
            float angle = (inverse ? 2 : -2) * PI * k / n;
            std::complex<float> t = std::polar(1.0f, angle) * odd[k];
            data[k] = even[k] + t;
            data[k + half] = even[k] - t;
        }
    }

    // FFT wrapper
    void fft_c(const float* in_real, const float* in_imag,
               float* out_real, float* out_imag,
               int n, int inverse) {
        
        std::vector<std::complex<float>> data(n);
        for (int i = 0; i < n; i++) {
            data[i] = std::complex<float>(in_real[i], in_imag ? in_imag[i] : 0.0f);
        }
        
        fft_recursive(data.data(), n, inverse);
        
        for (int i = 0; i < n; i++) {
            if (inverse) data[i] /= (float)n;
            out_real[i] = data[i].real();
            out_imag[i] = data[i].imag();
        }
    }

    // STFT Spectrogram: Returns magnitude spectrogram (freq_bins x time_frames)
    API void stft_spectrogram(const float* signal, int signal_len, int sr,
                              int win_ms, int hop_ms,
                              float* out_spec, int* out_freq_bins, int* out_frames) {
        
        int win_samples = (sr * win_ms) / 1000;
        int hop_samples = (sr * hop_ms) / 1000;
        
        // Round to next power of 2
        int nfft = 1;
        while (nfft < win_samples) nfft <<= 1;
        
        int num_frames = (signal_len - win_samples) / hop_samples + 1;
        int freq_bins = nfft / 2 + 1;
        
        *out_freq_bins = freq_bins;
        *out_frames = num_frames;
        
        std::vector<float> window(win_samples);
        std::vector<float> frame(nfft, 0.0f);
        std::vector<float> out_real(nfft);
        std::vector<float> out_imag(nfft);
        
        // Hanning window
        for (int i = 0; i < win_samples; i++) {
            window[i] = 0.5f * (1.0f - cos(2.0 * PI * i / (win_samples - 1)));
        }
        
        for (int f = 0; f < num_frames; f++) {
            int start = f * hop_samples;
            
            // Zero-pad and apply window
            for (int i = 0; i < nfft; i++) frame[i] = 0.0f;
            for (int i = 0; i < win_samples && (start + i) < signal_len; i++) {
                frame[i] = signal[start + i] * window[i];
            }
            
            // FFT
            fft_c(frame.data(), nullptr, out_real.data(), out_imag.data(), nfft, 0);
            
            // Magnitude of positive frequencies
            for (int k = 0; k < freq_bins; k++) {
                float mag = sqrt(out_real[k] * out_real[k] + out_imag[k] * out_imag[k]);
                out_spec[k * num_frames + f] = mag;
            }
        }
    }
}