/* equalizer_app/utils/native_fft.cpp */
#include <cmath>
#include <complex>
#include <vector>

const double PI = 3.141592653589793238460;

extern "C" {

    // Bit-reversal permutation
    void bit_reverse_copy(const float* src_real, const float* src_imag,
                          std::complex<float>* dst, int n) {
        int bits = 0;
        while ((1 << bits) < n) bits++;

        for (int i = 0; i < n; i++) {
            int rev = 0;
            int num = i;
            for (int j = 0; j < bits; j++) {
                rev = (rev << 1) | (num & 1);
                num >>= 1;
            }
            if (rev < n) {
                dst[rev] = std::complex<float>(src_real[i], src_imag ? src_imag[i] : 0.0f);
            }
        }
    }

    // Iterative Radix-2 FFT
    void fft_c(const float* in_real, const float* in_imag,
               float* out_real, float* out_imag,
               int n, int inverse) {

        std::vector<std::complex<float>> buffer(n);
        bit_reverse_copy(in_real, in_imag, buffer.data(), n);

        for (int s = 1; s <= log2(n); s++) {
            int m = 1 << s;
            int half_m = m >> 1;
            std::complex<float> wm = std::exp(std::complex<float>(0, (inverse ? 2 : -2) * PI / m));

            for (int k = 0; k < n; k += m) {
                std::complex<float> w = 1.0;
                for (int j = 0; j < half_m; j++) {
                    std::complex<float> t = w * buffer[k + j + half_m];
                    std::complex<float> u = buffer[k + j];
                    buffer[k + j] = u + t;
                    buffer[k + j + half_m] = u - t;
                    w *= wm;
                }
            }
        }

        for (int i = 0; i < n; i++) {
            if (inverse) {
                buffer[i] /= (float)n;
            }
            out_real[i] = buffer[i].real();
            out_imag[i] = buffer[i].imag();
        }
    }
}