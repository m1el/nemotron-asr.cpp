// Diarization model: streaming MarbleNet VAD + TitaNet-L speaker embedding.
//
// Loads a single diarize.gguf produced by scripts/convert_diarize_to_gguf.py
// containing two subnets, namespaced by tensor-name prefix:
//   vad.*   MarbleNet (vad_multilingual_marblenet)
//   spk.*   TitaNet-L (titanet_large)
//
// This module is independent from the existing ASR (nemo-ggml.{cpp,h}); the two
// share only the ggml runtime.

#ifndef NEMOTRON_DIARIZE_H
#define NEMOTRON_DIARIZE_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

// ----- Hparams (parsed from GGUF KV) ----------------------------------------

struct diarize_audio_hparams {
    int  sample_rate;
    int  n_mels;
    int  n_fft;
    float window_size;     // seconds
    float window_stride;   // seconds
    float dither;
    std::string normalize; // "None" or "per_feature"
    std::string window;    // "hann"
};

struct diarize_vad_hparams {
    diarize_audio_hparams audio;
    int n_classes;          // 2 (background, speech)
    // VAD inference window: 0.63 s = 63 frames at 10 ms hop; classic NeMo default.
    // Stored as constants here, not from GGUF.
    static constexpr float window_length_in_sec = 0.63f;
    static constexpr float shift_length_in_sec  = 0.01f;
};

struct diarize_spk_hparams {
    diarize_audio_hparams audio;
    int emb_dim;        // 192
    int attn_channels;  // 128
};

struct diarize_hparams {
    diarize_vad_hparams vad;
    diarize_spk_hparams spk;
    std::string arch;   // "nemo-diarize"
    std::string name;
};

// ----- Model handle ---------------------------------------------------------

struct diarize_model {
    diarize_hparams hparams;

    // Backend, contexts, buffer.
    ggml_backend_t backend       = nullptr;
    ggml_context * ctx_w         = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    // All tensors by name (full names with vad./spk. prefix).
    std::map<std::string, ggml_tensor *> tensors;
};

// Load a diarize.gguf into the model. Returns true on success.
// Uses CPU backend by default (the diarize subnets are tiny; CUDA gives no win).
bool diarize_model_load(const std::string & path, diarize_model & model);

void diarize_model_free(diarize_model & model);

// Helpers.
const ggml_tensor * diarize_model_get_tensor(const diarize_model & m, const std::string & name);
void diarize_model_print_tensors(const diarize_model & m);

#endif // NEMOTRON_DIARIZE_H
