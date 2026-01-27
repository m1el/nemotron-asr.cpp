// Audio preprocessing: PCM to mel spectrogram
// Implements NeMo's AudioToMelSpectrogramPreprocessor

#include "preprocessor.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Internal structures
// ============================================================================

struct Tensor {
    size_t n_rows = 0;
    size_t n_cols = 0;
    std::vector<float> data;

    void resize(size_t rows, size_t cols) {
        n_rows = rows;
        n_cols = cols;
        data.resize(rows * cols);
    }

    void fill(float value) {
        std::fill(data.begin(), data.end(), value);
    }

    float& operator()(size_t i, size_t j) {
        return data[i * n_cols + j];
    }

    float operator()(size_t i, size_t j) const {
        return data[i * n_cols + j];
    }
};

struct nemo_preprocessor {
    // Config (NeMo defaults for nemotron-speech-streaming-en-0.6b)
    int sample_rate = 16000;
    size_t n_window_size = 400;   // 25ms at 16kHz
    size_t n_window_stride = 160; // 10ms at 16kHz
    size_t n_fft = 512;
    float preemph = 0.97f;
    size_t n_mels = 128;
    size_t lowfreq = 0;
    size_t highfreq = 8000;  // sample_rate / 2
    float log_zero_guard = 5.960464477539063e-8f; // 2^-24
    float mag_power = 2.0f;

    // Precomputed values
    std::vector<float> window;
    Tensor filterbank;
    std::vector<float> sin_vals;
    std::vector<float> cos_vals;

    // Work buffers
    std::vector<float> frame;
    std::vector<float> real_out;
    std::vector<float> imag_out;
    std::vector<std::vector<float>> spectrogram;
    std::vector<float> audio_buf;
};

/*
// ============================================================================
// Mel filterbank computation
// ============================================================================

static float hz_to_mel(float frequency) {
    // Slaney-style mel scale
    const float f_min = 0.0f;
    const float f_sp = 200.0f / 3.0f;
    const float min_log_hz = 1000.0f;
    const float min_log_mel = (min_log_hz - f_min) / f_sp;
    const float logstep = logf(6.4f) / 27.0f;

    float mel = (frequency - f_min) / f_sp;
    if (frequency >= min_log_hz) {
        mel = min_log_mel + logf(frequency / min_log_hz) / logstep;
    }
    return mel;
}

static float mel_to_hz(float mel) {
    const float f_min = 0.0f;
    const float f_sp = 200.0f / 3.0f;
    const float min_log_hz = 1000.0f;
    const float min_log_mel = (min_log_hz - f_min) / f_sp;
    const float logstep = logf(6.4f) / 27.0f;

    float freq = f_min + f_sp * mel;
    if (mel >= min_log_mel) {
        freq = min_log_hz * expf(logstep * (mel - min_log_mel));
    }
    return freq;
}
*/

// ============================================================================
// DFT computation
// ============================================================================

static void fill_sin_cos_table(nemo_preprocessor * pp) {
    pp->sin_vals.resize(pp->n_fft);
    pp->cos_vals.resize(pp->n_fft);
    for (size_t i = 0; i < pp->n_fft; i++) {
        float theta = (2.0f * (float)M_PI * i) / pp->n_fft;
        pp->sin_vals[i] = sinf(theta);
        pp->cos_vals[i] = cosf(theta);
    }
}

static void dft_frame(nemo_preprocessor * pp, const float * frame) {
    size_t n_fft = pp->n_fft;
    size_t n_bins = 1 + n_fft / 2;

    for (size_t k = 0; k < n_bins; k++) {
        float real_sum = 0.0f;
        float imag_sum = 0.0f;
        for (size_t n = 0; n < n_fft; n++) {
            float sample = frame[n];
            size_t idx = (k * n) % n_fft;
            real_sum += sample * pp->cos_vals[idx];
            imag_sum -= sample * pp->sin_vals[idx];
        }
        pp->real_out[k] = real_sum;
        pp->imag_out[k] = imag_sum;
    }
}

// ============================================================================
// STFT magnitude
// ============================================================================

static void stft_magnitude(nemo_preprocessor * pp, const float * audio, size_t audio_len, size_t n_frames) {
    size_t n_fft = pp->n_fft;
    size_t hop_length = pp->n_window_stride;
    size_t win_length = pp->n_window_size;
    size_t n_bins = 1 + n_fft / 2;
    size_t pad_amount = n_fft / 2;

    // Ensure spectrogram columns are sized
    for (size_t k = 0; k < n_bins; k++) {
        if (pp->spectrogram[k].size() < n_frames) {
            pp->spectrogram[k].resize(n_frames);
        }
    }

    for (size_t t = 0; t < n_frames; t++) {
        int64_t start = (int64_t)(t * hop_length) - (int64_t)pad_amount;

        // Extract and window the frame
        for (size_t i = 0; i < n_fft; i++) {
            int64_t idx = start + (int64_t)i;
            float sample = 0.0f;
            if (idx >= 0 && idx < (int64_t)audio_len) {
                sample = audio[idx];
            }
            // Apply window
            size_t win_offset = (n_fft - win_length) / 2;
            if (i >= win_offset && i < win_offset + win_length) {
                sample *= pp->window[i - win_offset];
            } else {
                sample = 0.0f;
            }
            pp->frame[i] = sample;
        }

        // Compute DFT
        dft_frame(pp, pp->frame.data());

        // Compute magnitude
        for (size_t k = 0; k < n_bins; k++) {
            pp->spectrogram[k][t] = sqrtf(pp->real_out[k] * pp->real_out[k] +
                                          pp->imag_out[k] * pp->imag_out[k]);
        }
    }
}

// ============================================================================
// API implementation
// ============================================================================

// Helper to initialize work buffers (shared by both init functions)
static void init_work_buffers(nemo_preprocessor * pp) {
    size_t n_bins = 1 + pp->n_fft / 2;

    // Initialize sin/cos tables
    fill_sin_cos_table(pp);

    // Allocate work buffers
    pp->frame.resize(pp->n_fft);
    pp->real_out.resize(n_bins);
    pp->imag_out.resize(n_bins);
    pp->spectrogram.resize(n_bins);
}

struct nemo_preprocessor * nemo_preprocessor_init(
    const char * filterbank_path,
    const char * window_path
) {
    nemo_preprocessor * pp = new nemo_preprocessor();

    // Load window
    FILE * fwin = fopen(window_path, "rb");
    if (!fwin) {
        fprintf(stderr, "Failed to open window file: %s\n", window_path);
        delete pp;
        return nullptr;
    }
    pp->window.resize(pp->n_window_size);
    size_t read = fread(pp->window.data(), sizeof(float), pp->window.size(), fwin);
    fclose(fwin);
    if (read != pp->window.size()) {
        fprintf(stderr, "Failed to read window data\n");
        delete pp;
        return nullptr;
    }

    // Load filterbank
    size_t n_bins = 1 + pp->n_fft / 2;
    FILE * ffb = fopen(filterbank_path, "rb");
    if (!ffb) {
        fprintf(stderr, "Failed to open filterbank file: %s\n", filterbank_path);
        delete pp;
        return nullptr;
    }
    pp->filterbank.resize(pp->n_mels, n_bins);
    read = fread(pp->filterbank.data.data(), sizeof(float), pp->filterbank.data.size(), ffb);
    fclose(ffb);
    if (read != pp->filterbank.data.size()) {
        fprintf(stderr, "Failed to read filterbank data\n");
        delete pp;
        return nullptr;
    }

    init_work_buffers(pp);
    return pp;
}

struct nemo_preprocessor * nemo_preprocessor_init_from_data(
    const float * filterbank_data,
    size_t filterbank_size,
    const float * window_data,
    size_t window_size
) {
    nemo_preprocessor * pp = new nemo_preprocessor();

    size_t n_bins = 1 + pp->n_fft / 2;  // 257
    size_t expected_fb_size = pp->n_mels * n_bins;  // 128 * 257 = 32896

    // Validate sizes
    if (window_size != pp->n_window_size) {
        fprintf(stderr, "Window size mismatch: got %zu, expected %zu\n",
                window_size, pp->n_window_size);
        delete pp;
        return nullptr;
    }

    if (filterbank_size != expected_fb_size) {
        fprintf(stderr, "Filterbank size mismatch: got %zu, expected %zu\n",
                filterbank_size, expected_fb_size);
        delete pp;
        return nullptr;
    }

    // Copy window data
    pp->window.resize(window_size);
    memcpy(pp->window.data(), window_data, window_size * sizeof(float));

    // Copy filterbank data
    pp->filterbank.resize(pp->n_mels, n_bins);
    memcpy(pp->filterbank.data.data(), filterbank_data, filterbank_size * sizeof(float));

    init_work_buffers(pp);
    return pp;
}

void nemo_preprocessor_free(struct nemo_preprocessor * pp) {
    delete pp;
}

size_t nemo_preprocessor_get_n_frames(struct nemo_preprocessor * pp, size_t n_samples) {
    if (n_samples == 0) return 0;
    size_t pad_amount = pp->n_fft / 2;
    size_t padded_length = n_samples + 2 * pad_amount;
    return 1 + (padded_length - pp->n_fft) / pp->n_window_stride;
}

size_t nemo_preprocessor_process(
    struct nemo_preprocessor * pp,
    const int16_t * audio,
    size_t n_samples,
    std::vector<float> & mel_out
) {
    if (n_samples == 0) {
        mel_out.clear();
        return 0;
    }

    size_t n_bins = 1 + pp->n_fft / 2;
    size_t n_frames = nemo_preprocessor_get_n_frames(pp, n_samples);

    // Convert i16 to float and apply pre-emphasis
    if (pp->audio_buf.size() < n_samples) {
        pp->audio_buf.resize(n_samples);
    }

    // Scale i16 to [-1, 1] and apply pre-emphasis
    const float scale = 1.0f / 32768.0f;
    pp->audio_buf[0] = audio[0] * scale;
    for (size_t i = 1; i < n_samples; i++) {
        float curr = audio[i] * scale;
        float prev = audio[i - 1] * scale;
        pp->audio_buf[i] = curr - pp->preemph * prev;
    }

    // Compute STFT magnitude
    stft_magnitude(pp, pp->audio_buf.data(), n_samples, n_frames);

    // Power spectrum (magnitude^2)
    for (size_t k = 0; k < n_bins; k++) {
        for (size_t t = 0; t < n_frames; t++) {
            float mag = pp->spectrogram[k][t];
            pp->spectrogram[k][t] = mag * mag;
        }
    }

    // Allocate output: [n_frames, n_mels] row-major
    mel_out.resize(n_frames * pp->n_mels);

    // Apply mel filterbank and log
    for (size_t t = 0; t < n_frames; t++) {
        for (size_t m = 0; m < pp->n_mels; m++) {
            float sum = 0.0f;
            for (size_t k = 0; k < n_bins; k++) {
                sum += pp->filterbank(m, k) * pp->spectrogram[k][t];
            }
            // Log with zero guard
            mel_out[t * pp->n_mels + m] = logf(sum + pp->log_zero_guard);
        }
    }

    return n_frames;
}
