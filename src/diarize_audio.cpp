#include "diarize_audio.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// Cooley-Tukey radix-2 in-place FFT (matches src/preprocessor.cpp).
// ---------------------------------------------------------------------------

namespace {

struct fft_tables {
    std::vector<float> sin_t;
    std::vector<float> cos_t;
    std::vector<size_t> bit_rev;
    int n_fft = 0;

    void init(int n) {
        if (n_fft == n) return;
        n_fft = n;
        sin_t.resize(n);
        cos_t.resize(n);
        for (int i = 0; i < n; i++) {
            float th = (2.0f * (float)M_PI * (float)i) / (float)n;
            sin_t[i] = std::sin(th);
            cos_t[i] = std::cos(th);
        }
        int log2n = 0;
        for (int t = n; t > 1; t >>= 1) log2n++;
        bit_rev.resize(n);
        for (int i = 0; i < n; i++) {
            int r = 0, x = i;
            for (int j = 0; j < log2n; j++) { r = (r << 1) | (x & 1); x >>= 1; }
            bit_rev[i] = (size_t)r;
        }
    }
};

// Forward FFT in-place. Input frame (already windowed) of length n; output
// real[i], imag[i] for i in [0, n) (we'll only use the lower half).
static void fft_forward(const fft_tables & T,
                        const float * frame,
                        float * real, float * imag) {
    const int n = T.n_fft;
    for (int i = 0; i < n; i++) {
        real[T.bit_rev[i]] = frame[i];
        imag[T.bit_rev[i]] = 0.0f;
    }
    for (int m = 2; m <= n; m <<= 1) {
        int m2   = m >> 1;
        int step = n / m;
        for (int k = 0; k < n; k += m) {
            for (int j = 0; j < m2; j++) {
                int idx = j * step;
                float wr =  T.cos_t[idx];
                float wi = -T.sin_t[idx]; // forward FFT
                int i1 = k + j;
                int i2 = k + j + m2;
                float tr = wr * real[i2] - wi * imag[i2];
                float ti = wr * imag[i2] + wi * real[i2];
                real[i2] = real[i1] - tr;
                imag[i2] = imag[i1] - ti;
                real[i1] = real[i1] + tr;
                imag[i1] = imag[i1] + ti;
            }
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Apply preemphasis: y[0] = x[0]; y[t] = x[t] - preemph * x[t-1].
// ---------------------------------------------------------------------------

static void preemph_in_place(std::vector<float> & buf, float preemph) {
    if (buf.empty()) return;
    float prev = buf[0];
    for (size_t i = 1; i < buf.size(); i++) {
        float curr = buf[i];
        buf[i] = curr - preemph * prev;
        prev = curr;
    }
    // buf[0] stays as x[0] (matches NeMo).
}

// ---------------------------------------------------------------------------
// STFT with center=True + constant (zero) padding, matching NeMo's torch.stft
// call (pad_mode="constant"). Returns power spectrogram (n_bins, n_frames)
// row-major, where n_bins = n_fft/2+1 and n_frames = 1 + n_samples / hop.
// ---------------------------------------------------------------------------

static void stft_power(
    const std::vector<float> & audio,
    const float * window_pad,    // length n_fft (with hann zero-padded)
    const fft_tables & T,
    int n_fft, int hop,
    std::vector<float> & power)  // (n_bins * n_frames)
{
    const int n = (int)audio.size();
    const int half = n_fft / 2;
    const int n_frames = 1 + n / hop;
    const int n_bins = n_fft / 2 + 1;

    power.assign((size_t)n_bins * (size_t)n_frames, 0.0f);

    std::vector<float> frame(n_fft);
    std::vector<float> real(n_fft), imag(n_fft);

    for (int t = 0; t < n_frames; t++) {
        const int start = t * hop - half;
        for (int k = 0; k < n_fft; k++) {
            const int idx = start + k;
            const float s = (idx < 0 || idx >= n) ? 0.0f : audio[(size_t)idx];
            frame[k] = s * window_pad[k];
        }
        fft_forward(T, frame.data(), real.data(), imag.data());
        for (int k = 0; k < n_bins; k++) {
            const float r = real[k];
            const float i = imag[k];
            power[(size_t)k * (size_t)n_frames + (size_t)t] = r * r + i * i;
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------------

size_t diarize_compute_logmel(
    const float * audio_f32, size_t n_samples,
    const diarize_audio_cfg & cfg,
    const float * fb,
    const float * window,
    std::vector<float> & mel_out,
    size_t * out_t_valid)
{
    // Build the n_fft-length window with the win_size hann centered inside.
    std::vector<float> win_pad(cfg.n_fft, 0.0f);
    int win_off = (cfg.n_fft - cfg.win_size) / 2;
    for (int i = 0; i < cfg.win_size; i++) win_pad[win_off + i] = window[i];

    // Copy audio to mutable buffer; apply preemphasis.
    std::vector<float> audio(audio_f32, audio_f32 + n_samples);
    preemph_in_place(audio, cfg.preemph);

    // STFT.
    fft_tables T;
    T.init(cfg.n_fft);
    std::vector<float> power; // (n_bins, n_frames) = (k, t) row-major
    stft_power(audio, win_pad.data(), T, cfg.n_fft, cfg.hop_size, power);

    const int n_bins   = cfg.n_fft / 2 + 1;
    const int n_frames = 1 + (int)n_samples / cfg.hop_size;

    // Apply mel filterbank: mel[m, t] = sum_k fb[m, k] * power[k, t].
    // Then log(mel + eps).
    std::vector<float> mel((size_t)cfg.n_mels * (size_t)n_frames, 0.0f);
    for (int t = 0; t < n_frames; t++) {
        for (int m = 0; m < cfg.n_mels; m++) {
            float s = 0.0f;
            const float * fb_row    = fb + (size_t)m * (size_t)n_bins;
            const float * power_col = power.data();
            for (int k = 0; k < n_bins; k++) {
                s += fb_row[k] * power_col[(size_t)k * (size_t)n_frames + (size_t)t];
            }
            mel[(size_t)m * (size_t)n_frames + (size_t)t] = std::log(s + cfg.log_zero_guard);
        }
    }

    // Per-feature normalize: subtract per-mel mean, divide by per-mel std (over t).
    if (cfg.per_feature_normalize) {
        for (int m = 0; m < cfg.n_mels; m++) {
            float * row = mel.data() + (size_t)m * (size_t)n_frames;
            double sum = 0.0;
            for (int t = 0; t < n_frames; t++) sum += row[t];
            float mean = (float)(sum / n_frames);
            double var = 0.0;
            for (int t = 0; t < n_frames; t++) {
                float d = row[t] - mean;
                var += (double)d * d;
            }
            // NeMo uses unbiased variance (Bessel's correction).
            float std_v = std::sqrt((float)(var / std::max(1, n_frames - 1)));
            float inv_std = 1.0f / (std_v + 1e-5f);
            for (int t = 0; t < n_frames; t++) row[t] = (row[t] - mean) * inv_std;
        }
    }

    // NeMo's effective valid-frame count is n_samples / hop (without the +1 that
    // torch.stft(center=True) adds). The trailing extra frame computed by the
    // FFT is zeroed out as part of the pad-to-16 step.
    int t_valid = (int)n_samples / cfg.hop_size;
    int t_padded = t_valid;
    if (cfg.pad_to > 1) {
        int rem = t_valid % cfg.pad_to;
        if (rem != 0) t_padded += cfg.pad_to - rem;
    }
    mel_out.assign((size_t)cfg.n_mels * (size_t)t_padded, 0.0f);
    for (int m = 0; m < cfg.n_mels; m++) {
        const float * src = mel.data() + (size_t)m * (size_t)n_frames;
        float * dst = mel_out.data() + (size_t)m * (size_t)t_padded;
        std::memcpy(dst, src, sizeof(float) * (size_t)t_valid);
        // remaining frames already zero-initialized.
    }

    if (out_t_valid) *out_t_valid = (size_t)t_valid;
    return (size_t)t_padded;
}
