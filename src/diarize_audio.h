// Stateless 80-mel log-spectrogram preprocessor for diarization.
//
// Matches NeMo's AudioToMelSpectrogramPreprocessor:
//   - preemph 0.97 (default)
//   - hann window of `win_size` samples (zero-padded to n_fft inside STFT)
//   - center=True with reflect padding of n_fft/2 samples on both sides
//   - power spectrogram, mel filterbank from GGUF, log(x + 2^-24)
//   - optional per-feature normalization (mean/std along time axis)
//   - zero-pad output time dim up to multiple of 16 (NeMo's `pad_to`)
//
// The C++ side is non-streaming: each call processes one whole audio buffer
// and emits all (T_padded × n_mels) mel frames. Streaming VAD will call this
// per 0.63 s window from a higher level wrapper.

#ifndef NEMOTRON_DIARIZE_AUDIO_H
#define NEMOTRON_DIARIZE_AUDIO_H

#include <cstddef>
#include <vector>

struct diarize_audio_cfg {
    int   n_mels        = 80;
    int   n_fft         = 512;
    int   win_size      = 400;   // 25 ms at 16 kHz
    int   hop_size      = 160;   // 10 ms at 16 kHz
    float preemph       = 0.97f;
    float log_zero_guard = 5.960464477539063e-8f;  // 2^-24
    int   pad_to        = 16;
    bool  per_feature_normalize = false;
};

// Compute log-mel spectrogram of the given audio.
//
// Inputs:
//   audio_f32:  pointer to float32 audio in [-1, 1] at the model sample rate.
//   n_samples:  number of audio samples.
//   cfg:        preprocessor config.
//   fb:         mel filterbank, shape (n_mels, n_fft/2+1) row-major,
//               from GGUF tensor `<prefix>.preprocessor.featurizer.fb`.
//   window:     hann window, length cfg.win_size,
//               from GGUF tensor `<prefix>.preprocessor.featurizer.window`.
//
// Outputs:
//   mel_out:    resized to (T_padded × n_mels) row-major, where
//               T_padded = ceil(T_valid / pad_to) * pad_to.
//   *out_t_valid: number of valid (non-zero-padded) frames.
//
// Returns T_padded.
size_t diarize_compute_logmel(
    const float * audio_f32,
    size_t n_samples,
    const diarize_audio_cfg & cfg,
    const float * fb,
    const float * window,
    std::vector<float> & mel_out,
    size_t * out_t_valid);

#endif // NEMOTRON_DIARIZE_AUDIO_H
