// TitaNet-L speaker embedding inference.
//
// Encoder: 5 Jasper blocks with separable depthwise+pointwise convs, BN, SE.
// Decoder: attentive pooling (mean+std stat features → tdnn attention) →
//          BN(6144) → Conv1d(6144→192) → 192-d embedding.

#ifndef NEMOTRON_DIARIZE_SPK_H
#define NEMOTRON_DIARIZE_SPK_H

#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

#include "diarize.h"

struct spk_subconv {
    bool separable = true;
    ggml_tensor * dw_w = nullptr;     // (C_in, k)
    ggml_tensor * pw_w = nullptr;     // (C_in, C_out)
    ggml_tensor * bn_scale = nullptr; // (C_out,)
    ggml_tensor * bn_bias  = nullptr; // (C_out,)
    int kernel = 1;
    int dilation = 1;
};

struct spk_se {
    ggml_tensor * fc1_w = nullptr;    // (C, C/r)
    ggml_tensor * fc2_w = nullptr;    // (C/r, C)
};

struct spk_block {
    std::vector<spk_subconv> subs;
    bool residual = false;
    spk_subconv res;
    bool has_se = false;
    spk_se se;
};

struct spk_decoder_weights {
    // Attentive pooling.
    // Input is (B, 3*C, T) where C=enc_out=3072 (concat of x, mean, std).
    // attn_conv1: (B, 3*C, T) -> (B, attn_C=128, T)  via Conv1d(9216→128, k=1)
    //   then ReLU + BN(128).
    //   In our graph this is matmul + BN-fold + ReLU.
    ggml_tensor * attn_conv1_w  = nullptr; // (3*C, attn_C)
    ggml_tensor * attn_conv1_b  = nullptr; // (attn_C,)
    ggml_tensor * attn_bn_scale = nullptr; // (attn_C,) folded
    ggml_tensor * attn_bn_bias  = nullptr; // (attn_C,) folded

    // Then Tanh, then Conv1d(attn_C → C). Has bias, no BN.
    ggml_tensor * attn_conv2_w  = nullptr; // (attn_C, C)
    ggml_tensor * attn_conv2_b  = nullptr; // (C,)

    // emb_layer: BN(2C=6144) → Conv1d(2C → emb_dim=192, k=1).
    ggml_tensor * emb_bn_scale  = nullptr; // (2C,) folded
    ggml_tensor * emb_bn_bias   = nullptr; // (2C,) folded
    ggml_tensor * emb_conv_w    = nullptr; // (2C, emb_dim)
    ggml_tensor * emb_conv_b    = nullptr; // (emb_dim,)
};

struct spk_weights {
    std::vector<spk_block> blocks;     // 5 blocks
    spk_decoder_weights dec;
    int emb_dim = 192;
    int enc_out_channels = 3072;
    int attn_channels = 128;

    ggml_context * fold_ctx = nullptr;
    ggml_backend_buffer_t fold_buf = nullptr;
};

bool spk_weights_resolve(const diarize_model & m, ggml_backend_t backend,
                         spk_weights & w);
void spk_weights_free(spk_weights & w);

struct spk_graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph  * graph = nullptr;
    ggml_tensor  * mel_input    = nullptr; // (n_mels, T)
    ggml_tensor  * mask_input   = nullptr; // (1, T) — 1.0 valid, 0.0 pad
    ggml_tensor  * inv_lens     = nullptr; // (1, 1) — 1/lens scalar
    ggml_tensor  * neg_mask_inf = nullptr; // (1, T) — 0/-1e9 for softmax mask
    std::vector<ggml_tensor *> block_out;
    ggml_tensor  * encoder_out  = nullptr;
    ggml_tensor  * pool_out     = nullptr; // (2C, 1)
    ggml_tensor  * embedding    = nullptr; // (emb_dim,)
    int T = 0;
    int n_mels = 80;
};

spk_graph spk_graph_build(const spk_weights & w, int T, int n_mels = 80);

bool spk_graph_compute(spk_graph & g, ggml_backend_t backend, ggml_gallocr_t alloc,
                       const float * mel_data, int lens);

void spk_graph_free(spk_graph & g);

// ---- High-level API --------------------------------------------------------

constexpr int kSpkSampleRate    = 16000;
constexpr int kSpkSubsegSamples = 24000;  // 1.5 s
constexpr int kSpkMelValid      = 150;
constexpr int kSpkMelPadded     = 160;    // pad_to=16
constexpr int kSpkNMels         = 80;
constexpr int kSpkEmbDim        = 192;

struct spk_session;

spk_session * spk_session_init(diarize_model & m, const spk_weights & w);
void spk_session_free(spk_session * s);

// Compute the speaker embedding of one 1.5 s sub-segment. `audio` must point
// to >= kSpkSubsegSamples samples; lens_samples is the number of "real"
// samples (the rest is treated as zero-fill so masking applies). Output is
// written to `out_emb` (size kSpkEmbDim).
bool spk_session_run_chunk(spk_session * s, const float * audio,
                           int lens_samples, float * out_emb);

#endif // NEMOTRON_DIARIZE_SPK_H
