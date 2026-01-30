#pragma once

#include "ggml_weights.h"
#include "ops.h"

namespace nemo {

// ConformerFeedForward: Linear -> Swish -> Linear
// Used twice per layer (FFN1 and FFN2)
class ConformerFeedForward {
public:
    static constexpr size_t D_MODEL = 1024;
    static constexpr size_t D_FF = 4096;

    const float* linear1_weight = nullptr;  // [4096, 1024]
    const float* linear2_weight = nullptr;  // [1024, 4096]

    void load_weights(const ModelWeights& weights, const std::string& prefix);

    // input: [batch, time, 1024]
    // output: [batch, time, 1024]
    void forward(const TensorF& input, TensorF& output);

private:
    TensorF buf_;
};

// ConformerConvolution: Pointwise -> GLU -> Depthwise -> LN -> Swish -> Pointwise
class ConformerConvolution {
public:
    static constexpr size_t D_MODEL = 1024;
    static constexpr size_t KERNEL_SIZE = 9;

    const float* pointwise_conv1_weight = nullptr;  // [2048, 1024, 1]
    const float* depthwise_conv_weight = nullptr;   // [1024, 1, 9]
    const float* batch_norm_weight = nullptr;       // [1024]
    const float* batch_norm_bias = nullptr;         // [1024]
    const float* pointwise_conv2_weight = nullptr;  // [1024, 1024, 1]

    void load_weights(const ModelWeights& weights, const std::string& prefix);

    // input: [batch, time, 1024]
    // output: [batch, time, 1024]
    void forward(const TensorF& input, TensorF& output);

private:
    TensorF buf1_, buf2_, buf3_;
};

// RelPositionalEncoding: Generates relative position embeddings
class RelPositionalEncoding {
public:
    static constexpr size_t D_MODEL = 1024;
    static constexpr size_t MAX_LEN = 5000;

    // No learnable weights - computed on the fly
    // pos_emb: [2*max_len - 1, d_model] precomputed

    void init();

    // Get position embeddings for given sequence length
    // Returns: [2*seq_len - 1, d_model]
    void get_pos_emb(size_t seq_len, TensorF& pos_emb);

private:
    TensorF pos_emb_cache_;
    bool initialized_ = false;
};

// RelPositionMultiHeadAttention: Multi-head attention with relative position bias
class RelPositionMultiHeadAttention {
public:
    static constexpr size_t D_MODEL = 1024;
    static constexpr size_t N_HEADS = 8;
    static constexpr size_t D_HEAD = D_MODEL / N_HEADS;  // 128

    // Learnable biases for relative position
    const float* pos_bias_u = nullptr;  // [8, 128]
    const float* pos_bias_v = nullptr;  // [8, 128]

    // Projection weights (no bias)
    const float* linear_q_weight = nullptr;  // [1024, 1024]
    const float* linear_k_weight = nullptr;  // [1024, 1024]
    const float* linear_v_weight = nullptr;  // [1024, 1024]
    const float* linear_out_weight = nullptr;  // [1024, 1024]
    const float* linear_pos_weight = nullptr;  // [1024, 1024]

    void load_weights(const ModelWeights& weights, const std::string& prefix);

    // input: [batch, time, 1024]
    // pos_emb: [2*time-1, 1024] from RelPositionalEncoding
    // output: [batch, time, 1024]
    void forward(const TensorF& input, const TensorF& pos_emb, TensorF& output);

private:
    // Relative shift operation for position-aware attention
    void rel_shift(const TensorF& x, TensorF& out);

    TensorF q_, k_, v_, pos_;
    TensorF attn_scores_, attn_weights_;
    TensorF context_;
};

}  // namespace nemo
