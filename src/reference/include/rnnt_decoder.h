#pragma once

#include "ggml_weights.h"
#include "ops.h"

namespace nemo {

// RNNT Prediction Network (Decoder)
// Embedding -> 2-layer LSTM
class RNNTDecoder {
public:
    static constexpr size_t VOCAB_SIZE = 1025;  // 1024 tokens + 1 blank
    static constexpr size_t EMBED_DIM = 640;
    static constexpr size_t HIDDEN_SIZE = 640;
    static constexpr size_t NUM_LAYERS = 2;
    static constexpr int BLANK_TOKEN = 1024;  // Blank token ID

    // Embedding weights
    const float* embed_weight = nullptr;  // [1025, 640]

    // LSTM weights per layer
    const float* lstm_weight_ih[NUM_LAYERS] = {nullptr, nullptr};  // [2560, 640]
    const float* lstm_weight_hh[NUM_LAYERS] = {nullptr, nullptr};  // [2560, 640]
    const float* lstm_bias_ih[NUM_LAYERS] = {nullptr, nullptr};    // [2560]
    const float* lstm_bias_hh[NUM_LAYERS] = {nullptr, nullptr};    // [2560]

    // LSTM state
    LSTMState state;

    void load_weights(const ModelWeights& weights);

    // Initialize LSTM state for new sequence
    void init_state(size_t batch_size = 1);

    // Forward pass for single token
    // token: single token ID
    // output: [batch, 640] decoder output
    void forward_step(int token, TensorF& output);

    // Forward pass for multiple tokens (for training/testing)
    // tokens: [batch] token IDs
    // output: [batch, 640]
    void forward(const int* tokens, size_t batch_size, TensorF& output);

private:
    TensorF embed_out_;
    TensorF lstm_out_;
};

}  // namespace nemo
