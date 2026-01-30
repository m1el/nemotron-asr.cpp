#pragma once

#include "conformer_modules.h"
#include "conv_subsampling.h"
#include "ggml_weights.h"
#include "ops.h"

namespace nemo {

// Single Conformer layer
// Structure: x -> LN -> FFN1 -> +x*0.5 -> LN -> Attn -> +x -> LN -> Conv -> +x -> LN -> FFN2 -> +x*0.5 -> LN
class ConformerLayer {
public:
    static constexpr size_t D_MODEL = 1024;

    // Layer normalization weights
    const float* norm_ff1_weight = nullptr;
    const float* norm_ff1_bias = nullptr;
    const float* norm_attn_weight = nullptr;
    const float* norm_attn_bias = nullptr;
    const float* norm_conv_weight = nullptr;
    const float* norm_conv_bias = nullptr;
    const float* norm_ff2_weight = nullptr;
    const float* norm_ff2_bias = nullptr;
    const float* norm_out_weight = nullptr;
    const float* norm_out_bias = nullptr;

    // Sub-modules
    ConformerFeedForward ffn1;
    ConformerFeedForward ffn2;
    RelPositionMultiHeadAttention self_attn;
    ConformerConvolution conv;

    void load_weights(const ModelWeights& weights, const std::string& prefix);

    // input: [batch, time, 1024]
    // pos_emb: [2*time-1, 1024]
    // output: [batch, time, 1024]
    void forward(const TensorF& input, const TensorF& pos_emb, TensorF& output);

private:
    TensorF buf1_, buf2_, buf3_;
};

// Full Conformer Encoder
// ConvSubsampling -> 24 Conformer Layers
class ConformerEncoder {
public:
    static constexpr size_t NUM_LAYERS = 24;
    static constexpr size_t D_MODEL = 1024;

    ConvSubsampling subsampling;
    std::vector<ConformerLayer> layers;
    RelPositionalEncoding pos_enc;

    ConformerEncoder() : layers(NUM_LAYERS) {}

    void load_weights(const ModelWeights& weights);

    // input: [batch, time, 128] mel features
    // output: [batch, time/8, 1024] encoder output
    void forward(const TensorF& input, TensorF& output);

private:
    TensorF subsample_out_;
    TensorF pos_emb_;
    TensorF layer_in_, layer_out_;
};

}  // namespace nemo
