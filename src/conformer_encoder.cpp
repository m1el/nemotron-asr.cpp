#include "conformer_encoder.h"

namespace nemo {

// ============================================================================
// ConformerLayer
// ============================================================================

void ConformerLayer::load_weights(const ModelWeights& weights, const std::string& prefix) {
    // Load layer norm weights
    norm_ff1_weight = weights.require(prefix + ".norm_feed_forward1.weight").data.data();
    norm_ff1_bias = weights.require(prefix + ".norm_feed_forward1.bias").data.data();
    norm_attn_weight = weights.require(prefix + ".norm_self_att.weight").data.data();
    norm_attn_bias = weights.require(prefix + ".norm_self_att.bias").data.data();
    norm_conv_weight = weights.require(prefix + ".norm_conv.weight").data.data();
    norm_conv_bias = weights.require(prefix + ".norm_conv.bias").data.data();
    norm_ff2_weight = weights.require(prefix + ".norm_feed_forward2.weight").data.data();
    norm_ff2_bias = weights.require(prefix + ".norm_feed_forward2.bias").data.data();
    norm_out_weight = weights.require(prefix + ".norm_out.weight").data.data();
    norm_out_bias = weights.require(prefix + ".norm_out.bias").data.data();

    // Load sub-module weights
    ffn1.load_weights(weights, prefix + ".feed_forward1");
    ffn2.load_weights(weights, prefix + ".feed_forward2");
    self_attn.load_weights(weights, prefix + ".self_attn");
    conv.load_weights(weights, prefix + ".conv");
}

void ConformerLayer::forward(const TensorF& input, const TensorF& pos_emb, TensorF& output) {
    size_t batch = input.shape[0];
    size_t time = input.shape[1];

    // Start with input
    buf1_.resize({batch, time, D_MODEL});
    for (size_t i = 0; i < input.numel(); i++) {
        buf1_.data[i] = input.data[i];
    }

    // 1. FFN1 path: LN -> FFN1 -> residual * 0.5
    layer_norm(buf1_, norm_ff1_weight, norm_ff1_bias, D_MODEL, 1e-5f, buf2_);
    ffn1.forward(buf2_, buf3_);
    for (size_t i = 0; i < buf1_.numel(); i++) {
        buf1_.data[i] += 0.5f * buf3_.data[i];
    }

    // 2. Self-attention path: LN -> Attn -> residual
    layer_norm(buf1_, norm_attn_weight, norm_attn_bias, D_MODEL, 1e-5f, buf2_);
    self_attn.forward(buf2_, pos_emb, buf3_);
    for (size_t i = 0; i < buf1_.numel(); i++) {
        buf1_.data[i] += buf3_.data[i];
    }

    // 3. Conv path: LN -> Conv -> residual
    layer_norm(buf1_, norm_conv_weight, norm_conv_bias, D_MODEL, 1e-5f, buf2_);
    conv.forward(buf2_, buf3_);
    for (size_t i = 0; i < buf1_.numel(); i++) {
        buf1_.data[i] += buf3_.data[i];
    }

    // 4. FFN2 path: LN -> FFN2 -> residual * 0.5
    layer_norm(buf1_, norm_ff2_weight, norm_ff2_bias, D_MODEL, 1e-5f, buf2_);
    ffn2.forward(buf2_, buf3_);
    for (size_t i = 0; i < buf1_.numel(); i++) {
        buf1_.data[i] += 0.5f * buf3_.data[i];
    }

    // 5. Final layer norm
    layer_norm(buf1_, norm_out_weight, norm_out_bias, D_MODEL, 1e-5f, output);
}

// ============================================================================
// ConformerEncoder
// ============================================================================

void ConformerEncoder::load_weights(const ModelWeights& weights) {
    // Load subsampling weights
    subsampling.load_weights(weights);

    // Load each conformer layer
    for (size_t i = 0; i < NUM_LAYERS; i++) {
        std::string prefix = "encoder.layers." + std::to_string(i);
        layers[i].load_weights(weights, prefix);
    }

    // Initialize positional encoding
    pos_enc.init();
}

void ConformerEncoder::forward(const TensorF& input, TensorF& output) {
    // 1. Subsampling: [batch, time, 128] -> [batch, time/8, 1024]
    subsampling.forward(input, subsample_out_);

    // size_t batch = subsample_out_.shape[0];
    size_t time = subsample_out_.shape[1];

    // 2. Get positional embeddings for this sequence length
    pos_enc.get_pos_emb(time, pos_emb_);

    // 3. Process through all conformer layers
    layer_in_ = subsample_out_;

    for (size_t i = 0; i < NUM_LAYERS; i++) {
        layers[i].forward(layer_in_, pos_emb_, layer_out_);
        std::swap(layer_in_, layer_out_);
    }

    // layer_in_ now contains the final output
    output = layer_in_;
}

}  // namespace nemo
