// MarbleNet VAD inference graph (Jasper encoder + per-frame classifier).
//
// Tensor convention inside the encoder graph: (channels, time) with
// channels innermost (ne[0]=C, ne[1]=T).
//
// At resolve time we fold each BN into a pair of (scale, bias) per-channel
// tensors so the inference graph contains no BN op. The fold is:
//   scale[c] = gamma[c] / sqrt(var[c] + eps)
//   bias[c]  = beta[c]  - mean[c] * scale[c]
// and BN(y) = scale * y + bias  (broadcast over time).

#ifndef NEMOTRON_DIARIZE_VAD_H
#define NEMOTRON_DIARIZE_VAD_H

#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

#include "diarize.h"

struct vad_subconv {
    bool separable = true;
    ggml_tensor * dw_w = nullptr;     // (C_in, k);  null when !separable
    ggml_tensor * pw_w = nullptr;     // (C_in, C_out)
    ggml_tensor * bn_scale = nullptr; // (C_out,) folded BN scale
    ggml_tensor * bn_bias  = nullptr; // (C_out,) folded BN bias
    int kernel = 1;
    int dilation = 1;
};

struct vad_block {
    std::vector<vad_subconv> subs;
    bool residual = false;
    vad_subconv res;
};

struct vad_weights {
    std::vector<vad_block> blocks;
    ggml_tensor * dec_w = nullptr;
    ggml_tensor * dec_b = nullptr;
    int n_classes = 2;
    int enc_out_channels = 128;

    // Folded BN tensors live here, so we can free them with the rest.
    ggml_context * fold_ctx = nullptr;
    ggml_backend_buffer_t fold_buf = nullptr;
};

// Resolve the vad.* tensors and precompute folded BN scale/bias on the
// given backend. After this call, `m`'s tensors are still owned by `m`,
// but `w`'s fold tensors are owned by `w`.
bool vad_weights_resolve(const diarize_model & m, ggml_backend_t backend,
                         vad_weights & w);

void vad_weights_free(vad_weights & w);

struct vad_graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph  * graph = nullptr;
    ggml_tensor  * mel_input = nullptr;
    ggml_tensor  * mask_input = nullptr;           // ne=(1, T): 1.0 for t<lens else 0.0
    std::vector<ggml_tensor *> block_out;
    ggml_tensor  * encoder_out = nullptr;
    int T = 0;
    int n_mels = 80;
};

vad_graph vad_graph_build(const vad_weights & w, int T, int n_mels = 80);

// Compute the encoder. `lens` is the number of valid mel frames; positions
// [lens, T) are zeroed before each MaskedConv1d, matching NeMo's protocol.
// Pass lens=T to disable masking (steady-state streaming).
bool vad_graph_compute(
    vad_graph & g,
    ggml_backend_t backend,
    ggml_gallocr_t alloc,
    const float * mel_data,  // (n_mels, T) channels-innermost layout
    int lens);

void vad_graph_free(vad_graph & g);

// ---- High-level streaming-friendly API -------------------------------------

constexpr int kVadSampleRate    = 16000;
constexpr int kVadWindowSamples = 10080;  // 0.63 s
constexpr int kVadShiftSamples  = 160;    // 0.01 s
constexpr int kVadMelValid      = 63;
constexpr int kVadMelPadded     = 64;     // pad_to=16 → ceil(63/16)*16
constexpr int kVadNMels         = 80;

struct vad_session;

// Initialize a VAD session backed by `m` (loaded diarize.gguf) and `w`
// (resolved vad weights). Both must outlive the session.
vad_session * vad_session_init(diarize_model & m, const vad_weights & w);
void vad_session_free(vad_session * s);

// Run VAD on a batch of audio (16 kHz mono float in [-1, 1]). Appends one
// speech probability to `out_speech_probs` per 10 ms shift, for shifts
// where a full 0.63 s window fits inside `audio`. Returns the number of
// new probabilities appended.
//
// At end-of-stream, callers wanting to flush the trailing audio (where
// the last window would extend past the available samples) can pad
// `audio` with zeros and pass an explicit shorter `lens` via
// vad_session_run_chunk.
size_t vad_session_run_batch(
    vad_session * s,
    const float * audio, size_t n_samples,
    std::vector<float> & out_speech_probs);

// Run a single 0.63 s chunk through preprocessor + encoder + decoder.
// `audio` must point to >= kVadWindowSamples samples; `lens_samples` is
// the number of "real" samples (the rest is treated as zero-fill so the
// MaskedConv1d zeroing applies). For a fully-real chunk pass
// kVadWindowSamples.
float vad_session_run_chunk(
    vad_session * s,
    const float * audio,
    int lens_samples = kVadWindowSamples);

// ---- VAD post-processing: probabilities -> speech segments ----------------

struct vad_segment {
    float start_sec;
    float end_sec;
};

struct vad_post_cfg {
    float onset            = 0.5f;
    float offset           = 0.5f;
    float pad_onset        = 0.0f;
    float pad_offset       = 0.0f;
    float min_duration_on  = 0.0f;
    float min_duration_off = 0.0f;
    float frame_period_sec = 0.01f;
};

// Threshold-based segment extraction (matching the spirit of NeMo's
// generate_vad_segment_table_per_tensor). Returns segments sorted by start.
std::vector<vad_segment> vad_extract_segments(
    const std::vector<float> & speech_probs,
    const vad_post_cfg & cfg = {});

#endif // NEMOTRON_DIARIZE_VAD_H
