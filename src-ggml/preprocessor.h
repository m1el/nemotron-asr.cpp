// Audio preprocessing: PCM to mel spectrogram
// Implements NeMo's AudioToMelSpectrogramPreprocessor

#ifndef NEMO_PREPROCESSOR_H
#define NEMO_PREPROCESSOR_H

#include <cstdint>
#include <cstddef>
#include <vector>

struct nemo_preprocessor;

// Create preprocessor with NeMo default config
// filterbank_path: path to filterbank weights (featurizer.fb.bin)
// window_path: path to window function (featurizer.window.bin)
struct nemo_preprocessor * nemo_preprocessor_init(
    const char * filterbank_path,
    const char * window_path
);

// Create preprocessor from pre-loaded weights
// filterbank_data: [n_mels, n_fft/2+1] = [128, 257] float array
// window_data: [n_window_size] = [400] float array
struct nemo_preprocessor * nemo_preprocessor_init_from_data(
    const float * filterbank_data,
    size_t filterbank_size,
    const float * window_data,
    size_t window_size
);

void nemo_preprocessor_free(struct nemo_preprocessor * pp);

// Process raw PCM audio to mel spectrogram
// audio: int16_t samples at 16kHz, mono
// n_samples: number of audio samples
// mel_out: output mel spectrogram [n_frames, 128] row-major
// Returns: number of valid frames
size_t nemo_preprocessor_process(
    struct nemo_preprocessor * pp,
    const int16_t * audio,
    size_t n_samples,
    std::vector<float> & mel_out
);

// Get number of mel frames for given audio length
size_t nemo_preprocessor_get_n_frames(
    struct nemo_preprocessor * pp,
    size_t n_samples
);

#endif // NEMO_PREPROCESSOR_H
