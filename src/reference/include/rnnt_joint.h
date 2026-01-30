#pragma once

#include "ggml_weights.h"
#include "ops.h"

namespace nemo {

// RNNT Joint Network
// Combines encoder and decoder outputs to produce vocabulary logits
class RNNTJoint {
public:
    static constexpr size_t ENCODER_DIM = 1024;
    static constexpr size_t DECODER_DIM = 640;
    static constexpr size_t JOINT_DIM = 640;
    static constexpr size_t VOCAB_SIZE = 1025;

    // Encoder projection: [1024] -> [640]
    const float* enc_weight = nullptr;  // [640, 1024]
    const float* enc_bias = nullptr;    // [640]

    // Decoder (prediction) projection: [640] -> [640]
    const float* pred_weight = nullptr;  // [640, 640]
    const float* pred_bias = nullptr;    // [640]

    // Output projection: [640] -> [1025]
    const float* out_weight = nullptr;   // [1025, 640]
    const float* out_bias = nullptr;     // [1025]

    void load_weights(const ModelWeights& weights);

    // Forward pass for single encoder frame and decoder output
    // enc_out: [batch, enc_dim] or [batch, 1, enc_dim] single encoder frame
    // dec_out: [batch, dec_dim] decoder output
    // logits: [batch, vocab_size] output logits
    void forward(const TensorF& enc_out, const TensorF& dec_out, TensorF& logits);

    // Forward pass for full encoder sequence (for beam search)
    // enc_out: [batch, time, enc_dim] full encoder output
    // dec_out: [batch, dec_dim] single decoder output
    // logits: [batch, time, vocab_size] output logits for all frames
    void forward_full(const TensorF& enc_out, const TensorF& dec_out, TensorF& logits);

private:
    TensorF enc_proj_, dec_proj_, joint_;
};

}  // namespace nemo
