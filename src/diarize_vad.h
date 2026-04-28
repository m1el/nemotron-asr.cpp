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

#endif // NEMOTRON_DIARIZE_VAD_H
