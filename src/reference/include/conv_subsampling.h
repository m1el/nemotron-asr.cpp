#pragma once

#include "ggml_weights.h"
#include "ops.h"

namespace nemo {

// ConvSubsampling: 8x downsampling using causal convolutions
//
// Architecture:
//   Input: [batch, time, 128] (mel features)
//   -> CausalConv2D(1->256, k=3, s=2) + ReLU
//   -> DepthwiseConv2D(256, k=3, s=2) + PointwiseConv + ReLU
//   -> DepthwiseConv2D(256, k=3, s=2) + PointwiseConv + ReLU
//   -> Reshape to [batch, time/8, 256*17]
//   -> Linear(4352, 1024)
//   Output: [batch, time/8, 1024]
//
class ConvSubsampling {
public:
    // Model dimensions
    static constexpr size_t IN_FEATURES = 128;
    static constexpr size_t CONV_CHANNELS = 256;
    static constexpr size_t OUT_FEATURES = 1024;
    static constexpr size_t KERNEL_SIZE = 3;
    static constexpr size_t STRIDE = 2;

    // Weight pointers (set by load_weights)
    // Conv layer 0: CausalConv2D(1, 256, k=3, s=2)
    const float* conv0_weight = nullptr;  // [256, 1, 3, 3]
    const float* conv0_bias = nullptr;    // [256]

    // Conv layer 2: Depthwise CausalConv2D(256, 256, k=3, s=2, groups=256)
    const float* conv2_weight = nullptr;  // [256, 1, 3, 3]
    const float* conv2_bias = nullptr;    // [256]

    // Conv layer 3: Pointwise Conv2d(256, 256, k=1, s=1)
    const float* conv3_weight = nullptr;  // [256, 256, 1, 1]
    const float* conv3_bias = nullptr;    // [256]

    // Conv layer 5: Depthwise CausalConv2D(256, 256, k=3, s=2, groups=256)
    const float* conv5_weight = nullptr;  // [256, 1, 3, 3]
    const float* conv5_bias = nullptr;    // [256]

    // Conv layer 6: Pointwise Conv2d(256, 256, k=1, s=1)
    const float* conv6_weight = nullptr;  // [256, 256, 1, 1]
    const float* conv6_bias = nullptr;    // [256]

    // Output linear layer
    const float* out_weight = nullptr;    // [1024, 4352]
    const float* out_bias = nullptr;      // [1024]

    // Load weights from model
    void load_weights(const ModelWeights& weights);

    // Forward pass
    // input: [batch, time, 128] - mel spectrogram features
    // output: [batch, time/8, 1024] - subsampled encoder input
    void forward(const TensorF& input, TensorF& output);

    // Get output time dimension given input time dimension
    static size_t get_output_length(size_t input_length) {
        // NeMo CausalConv2D padding: left = k-1 = 2, right = stride-1 = 1
        // Total padding = 3, output = (input + 3 - 3) / 2 + 1 = input/2 + 1
        size_t len = input_length;
        for (int i = 0; i < 3; i++) {
            len = len / 2 + 1;
        }
        return len;
    }

private:
    // Intermediate buffers
    TensorF buf1_, buf2_;
};

}  // namespace nemo
