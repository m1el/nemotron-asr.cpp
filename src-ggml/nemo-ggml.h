#ifndef NEMO_GGML_H
#define NEMO_GGML_H

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <string>
#include <vector>
#include <map>

// Model hyperparameters
struct nemo_hparams {
    int32_t n_mels      = 128;    // mel spectrogram features
    int32_t d_model     = 1024;   // model dimension
    int32_t n_heads     = 8;      // attention heads
    int32_t d_head      = 128;    // head dimension (d_model / n_heads)
    int32_t d_ff        = 4096;   // feedforward dimension
    int32_t n_layers    = 24;     // number of conformer layers
    int32_t kernel_size = 31;     // conv kernel size
    int32_t vocab_size  = 1025;   // vocabulary size (1024 tokens + blank)
    int32_t decoder_dim = 320;    // decoder LSTM hidden size
    int32_t joint_dim   = 640;    // joint network hidden size
    float   eps         = 1e-5f;  // layer norm epsilon
};

// ConvSubsampling weights
struct nemo_conv_subsampling {
    struct ggml_tensor * conv1_w;   // [256, 1, 3, 3]
    struct ggml_tensor * conv1_b;   // [256]
    struct ggml_tensor * conv2_w;   // [256, 256, 3, 3]
    struct ggml_tensor * conv2_b;   // [256]
    struct ggml_tensor * conv3_w;   // [256, 256, 3, 3]
    struct ggml_tensor * conv3_b;   // [256]
    struct ggml_tensor * out_w;     // [1024, 2048]
};

// Conformer layer weights
struct nemo_conformer_layer {
    // FFN1
    struct ggml_tensor * norm_ff1_w;
    struct ggml_tensor * norm_ff1_b;
    struct ggml_tensor * ffn1_linear1_w;  // [4096, 1024]
    struct ggml_tensor * ffn1_linear2_w;  // [1024, 4096]

    // Self-attention
    struct ggml_tensor * norm_attn_w;
    struct ggml_tensor * norm_attn_b;
    struct ggml_tensor * attn_q_w;        // [1024, 1024]
    struct ggml_tensor * attn_k_w;        // [1024, 1024]
    struct ggml_tensor * attn_v_w;        // [1024, 1024]
    struct ggml_tensor * attn_pos_w;      // [1024, 1024]
    struct ggml_tensor * attn_out_w;      // [1024, 1024]
    struct ggml_tensor * pos_bias_u;      // [8, 128]
    struct ggml_tensor * pos_bias_v;      // [8, 128]

    // Conv module
    struct ggml_tensor * norm_conv_w;
    struct ggml_tensor * norm_conv_b;
    struct ggml_tensor * conv_pw1_w;      // [2048, 1024, 1]
    struct ggml_tensor * conv_dw_w;       // [1024, 1, 31]
    struct ggml_tensor * conv_ln_w;       // [1024]
    struct ggml_tensor * conv_ln_b;       // [1024]
    struct ggml_tensor * conv_pw2_w;      // [1024, 1024, 1]

    // FFN2
    struct ggml_tensor * norm_ff2_w;
    struct ggml_tensor * norm_ff2_b;
    struct ggml_tensor * ffn2_linear1_w;  // [4096, 1024]
    struct ggml_tensor * ffn2_linear2_w;  // [1024, 4096]

    // Final norm
    struct ggml_tensor * norm_final_w;
    struct ggml_tensor * norm_final_b;
};

// Encoder weights
struct nemo_encoder {
    nemo_conv_subsampling subsampling;
    std::vector<nemo_conformer_layer> layers;
    struct ggml_tensor * final_norm_w;
    struct ggml_tensor * final_norm_b;
    struct ggml_tensor * fc_w;            // [640, 1024] encoder output projection
};

// Decoder weights
struct nemo_decoder {
    struct ggml_tensor * embedding;       // [1024, 320]
    struct ggml_tensor * lstm_w_ih;       // [1280, 320]
    struct ggml_tensor * lstm_w_hh;       // [1280, 320]
    struct ggml_tensor * lstm_b_ih;       // [1280]
    struct ggml_tensor * lstm_b_hh;       // [1280]
    struct ggml_tensor * fc_w;            // [640, 320]
};

// Joint network weights
struct nemo_joint {
    struct ggml_tensor * enc_w;           // [640, 640]
    struct ggml_tensor * dec_w;           // [640, 640]
    struct ggml_tensor * out_w;           // [1025, 640]
    struct ggml_tensor * out_b;           // [1025]
};

// Full model
struct nemo_model {
    nemo_hparams hparams;

    nemo_encoder encoder;
    nemo_decoder decoder;
    nemo_joint joint;

    // Precomputed positional embeddings
    struct ggml_tensor * pos_emb;         // [max_len*2-1, 1024]

    // ggml contexts and buffers
    struct ggml_context * ctx_w;          // weights context
    ggml_backend_t backend;
    ggml_backend_buffer_t buffer_w;

    // Tensor name mapping for loading
    std::map<std::string, struct ggml_tensor *> tensors;
};

// Runtime state for inference
struct nemo_state {
    // Decoder LSTM state
    std::vector<float> h;                 // [320]
    std::vector<float> c;                 // [320]
    int prev_token;

    // Allocator for compute graphs
    ggml_gallocr_t allocr;

    nemo_state() : h(320, 0.0f), c(320, 0.0f), prev_token(1024) {}

    void reset() {
        std::fill(h.begin(), h.end(), 0.0f);
        std::fill(c.begin(), c.end(), 0.0f);
        prev_token = 1024;  // blank token
    }
};

// Context combining model and state
struct nemo_context {
    nemo_model model;
    nemo_state state;

    // Number of threads for computation
    int n_threads = 4;
};

// API functions
struct nemo_context * nemo_init(const char * model_path);
void nemo_free(struct nemo_context * ctx);

// Load model weights from file
bool nemo_model_load(const std::string & path, nemo_model & model);

// Build computation graphs
struct ggml_cgraph * nemo_build_encoder_graph(
    struct nemo_context * ctx,
    struct ggml_context * ctx0,
    struct ggml_tensor * mel);

struct ggml_tensor * nemo_build_decoder_step(
    struct nemo_context * ctx,
    struct ggml_context * ctx0,
    int token,
    struct ggml_tensor * h_in,
    struct ggml_tensor * c_in,
    struct ggml_tensor ** h_out,
    struct ggml_tensor ** c_out);

struct ggml_tensor * nemo_build_joint(
    struct nemo_context * ctx,
    struct ggml_context * ctx0,
    struct ggml_tensor * encoder_out,
    struct ggml_tensor * decoder_out);

// Run inference
std::vector<int> nemo_encode(
    struct nemo_context * ctx,
    const float * mel_data,
    int n_mel_frames);

std::string nemo_transcribe(
    struct nemo_context * ctx,
    const float * mel_data,
    int n_mel_frames);

#endif // NEMO_GGML_H
