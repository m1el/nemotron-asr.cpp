#ifndef NEMO_GGML_H
#define NEMO_GGML_H

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cstdint>
#include <string>
#include <vector>
#include <map>

// Backend type for inference
enum nemo_backend_type {
    NEMO_BACKEND_CPU = 0,
    NEMO_BACKEND_CUDA = 1,
    NEMO_BACKEND_AUTO = 2,  // Auto-detect: prefer CUDA if available
};

// Forward declaration
struct nemo_preprocessor;

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

// ConvSubsampling weights (depthwise-separable architecture)
struct nemo_conv_subsampling {
    // Conv0: CausalConv2D(1, 256, k=3, s=2) + ReLU
    struct ggml_tensor * conv0_w;   // [3, 3, 1, 256]
    struct ggml_tensor * conv0_b;   // [256]

    // Conv2: Depthwise CausalConv2D(256, k=3, s=2, groups=256)
    struct ggml_tensor * conv2_w;   // [3, 3, 1, 256]
    struct ggml_tensor * conv2_b;   // [256]

    // Conv3: Pointwise Conv2D(256, 256, k=1, s=1) + ReLU
    struct ggml_tensor * conv3_w;   // [1, 1, 256, 256]
    struct ggml_tensor * conv3_b;   // [256]

    // Conv5: Depthwise CausalConv2D(256, k=3, s=2, groups=256)
    struct ggml_tensor * conv5_w;   // [3, 3, 1, 256]
    struct ggml_tensor * conv5_b;   // [256]

    // Conv6: Pointwise Conv2D(256, 256, k=1, s=1) + ReLU
    struct ggml_tensor * conv6_w;   // [1, 1, 256, 256]
    struct ggml_tensor * conv6_b;   // [256]

    // Output projection
    struct ggml_tensor * out_w;     // [1024, flat_dim]
    struct ggml_tensor * out_b;     // [1024]
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

// Decoder weights (2-layer LSTM)
struct nemo_decoder {
    static constexpr int NUM_LAYERS = 2;
    static constexpr int HIDDEN_SIZE = 640;
    static constexpr int EMBED_DIM = 640;

    struct ggml_tensor * embedding;             // [640, 1025] (embed_dim, vocab_size)
    // LSTM layer 0
    struct ggml_tensor * lstm_w_ih_l0;          // [2560, 640] = [4*hidden, input]
    struct ggml_tensor * lstm_w_hh_l0;          // [2560, 640] = [4*hidden, hidden]
    struct ggml_tensor * lstm_b_ih_l0;          // [2560]
    struct ggml_tensor * lstm_b_hh_l0;          // [2560]
    // LSTM layer 1
    struct ggml_tensor * lstm_w_ih_l1;          // [2560, 640]
    struct ggml_tensor * lstm_w_hh_l1;          // [2560, 640]
    struct ggml_tensor * lstm_b_ih_l1;          // [2560]
    struct ggml_tensor * lstm_b_hh_l1;          // [2560]
};

// Joint network weights
struct nemo_joint {
    struct ggml_tensor * enc_w;           // [640, 1024] encoder projection
    struct ggml_tensor * enc_b;           // [640]
    struct ggml_tensor * dec_w;           // [640, 640] decoder projection
    struct ggml_tensor * dec_b;           // [640]
    struct ggml_tensor * out_w;           // [1025, 640] vocab projection
    struct ggml_tensor * out_b;           // [1025]
};

struct char8 {
    // null-terminated string, at most 7 chars
    char data[8];
};

// Preprocessor weights (mel filterbank and window)
struct nemo_preprocessor_weights {
    struct ggml_tensor * filterbank;      // [n_mels, n_fft/2+1] = [128, 257]
    struct ggml_tensor * window;          // [n_window_size] = [400]
};

// Full model
struct nemo_model {
    nemo_hparams hparams;
    std::vector<char8> vocab;
    nemo_encoder encoder;
    nemo_decoder decoder;
    nemo_joint joint;
    nemo_preprocessor_weights preprocessor_weights;

    // Precomputed positional embeddings
    struct ggml_tensor * pos_emb;         // [max_len*2-1, 1024]

    // ggml contexts and buffers
    struct ggml_context * ctx_w;          // weights context
    ggml_backend_t backend;
    ggml_backend_buffer_t buffer_w;
    nemo_backend_type backend_type;       // which backend is in use

    // Tensor name mapping for loading
    std::map<std::string, struct ggml_tensor *> tensors;
};

// Runtime state for inference
struct nemo_state {
    // Decoder LSTM state for 2 layers
    static constexpr int HIDDEN_SIZE = 640;
    static constexpr int NUM_LAYERS = 2;

    std::vector<float> h;                 // [NUM_LAYERS * HIDDEN_SIZE]
    std::vector<float> c;                 // [NUM_LAYERS * HIDDEN_SIZE]
    int prev_token;

    // Allocator for compute graphs
    ggml_gallocr_t allocr;

    nemo_state() : h(NUM_LAYERS * HIDDEN_SIZE, 0.0f), c(NUM_LAYERS * HIDDEN_SIZE, 0.0f), prev_token(1024) {}

    void reset() {
        std::fill(h.begin(), h.end(), 0.0f);
        std::fill(c.begin(), c.end(), 0.0f);
        prev_token = 1024;  // blank token
    }

    // Get h state for layer l
    float * h_layer(int l) { return h.data() + l * HIDDEN_SIZE; }
    float * c_layer(int l) { return c.data() + l * HIDDEN_SIZE; }
};

// Context combining model and state
struct nemo_context {
    nemo_model model;
    nemo_state state;

    // Audio preprocessor (optional, for audio input)
    struct nemo_preprocessor * preprocessor = nullptr;

    // Number of threads for computation (CPU backend only)
    int n_threads = 4;
};

// API functions
// Initialize with automatic backend selection (prefers CUDA if available)
struct nemo_context * nemo_init(const char * model_path);

// Initialize with specific backend
struct nemo_context * nemo_init_with_backend(const char * model_path, nemo_backend_type backend);

void nemo_free(struct nemo_context * ctx);

// Get current backend name
const char * nemo_get_backend_name(struct nemo_context * ctx);

// Load model weights from file (with backend selection)
bool nemo_model_load(const std::string & path, nemo_model & model, nemo_backend_type backend = NEMO_BACKEND_AUTO);

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

// Build full conformer layer graph
struct ggml_tensor * build_conformer_layer(
    struct ggml_context * ctx,
    struct ggml_tensor * input,     // [d_model, time, batch]
    struct ggml_tensor * pos_emb,   // [d_model, pos_len] precomputed
    nemo_conformer_layer * layer,   // layer weights
    int n_heads,
    int d_head,
    int kernel_size);

// Build ConvSubsampling: mel -> subsampled features
// mel: [n_mels, time, batch]
// Returns: [d_model, time/8, batch]
struct ggml_tensor * build_conv_subsampling(
    struct ggml_context * ctx,
    struct ggml_tensor * mel,           // [n_mels, time, batch]
    nemo_conv_subsampling * subsampling // weights
);

// Build decoder step: embedding + 2-layer LSTM
// token_emb: [hidden_size] token embedding
// h_in, c_in: [2 * hidden_size] concatenated LSTM states
// Returns: decoder output [hidden_size], and updated h_out, c_out
struct ggml_tensor * build_decoder_step(
    struct ggml_context * ctx,
    struct ggml_tensor * token_emb,     // [hidden_size] token embedding
    struct ggml_tensor * h_in,          // [2 * hidden_size]
    struct ggml_tensor * c_in,          // [2 * hidden_size]
    nemo_decoder * decoder,
    struct ggml_tensor ** h_out,        // output: [2 * hidden_size]
    struct ggml_tensor ** c_out         // output: [2 * hidden_size]
);

// Build joint network: encoder_out + decoder_out -> logits
// encoder_out: [d_model] or [d_model, 1]
// decoder_out: [hidden_size]
// Returns: [vocab_size] logits
struct ggml_tensor * build_joint(
    struct ggml_context * ctx,
    struct ggml_tensor * encoder_out,   // [d_model] or [d_model, 1]
    struct ggml_tensor * decoder_out,   // [hidden_size]
    nemo_joint * joint
);

// Build full encoder: ConvSubsampling + 24 Conformer layers
// mel: [n_mels, time, batch]
// Returns: [d_model, time/8, batch]
struct ggml_tensor * build_encoder(
    struct ggml_context * ctx,
    struct ggml_tensor * mel,       // [n_mels, time, batch]
    nemo_model * model              // model with all weights
);

// Run inference from mel spectrogram
std::vector<int> nemo_encode(
    struct nemo_context * ctx,
    const float * mel_data,
    int n_mel_frames);

std::string nemo_transcribe(
    struct nemo_context * ctx,
    const float * mel_data,
    int n_mel_frames);

// Run inference from raw PCM audio (16-bit signed, 16kHz, mono)
std::vector<int> nemo_encode_audio(
    struct nemo_context * ctx,
    const int16_t * audio_data,
    int n_samples);

std::string nemo_transcribe_audio(
    struct nemo_context * ctx,
    const int16_t * audio_data,
    int n_samples);

#endif // NEMO_GGML_H
