#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

namespace nemo {

// Simple tensor class for intermediate computations
// Data layout: row-major (C-order)
struct TensorF {
    std::vector<float> data;
    std::vector<size_t> shape;

    TensorF() = default;

    explicit TensorF(const std::vector<size_t>& shape_) : shape(shape_) {
        data.resize(numel());
    }

    TensorF(const std::vector<size_t>& shape_, float fill_value) : shape(shape_) {
        data.resize(numel(), fill_value);
    }

    size_t numel() const {
        size_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }

    size_t ndim() const { return shape.size(); }

    void resize(const std::vector<size_t>& new_shape) {
        shape = new_shape;
        data.resize(numel());
    }

    void fill(float value) {
        std::fill(data.begin(), data.end(), value);
    }

    void zero() { fill(0.0f); }

    // 1D access
    float& operator()(size_t i) { return data[i]; }
    const float& operator()(size_t i) const { return data[i]; }

    // 2D access [i, j]
    float& operator()(size_t i, size_t j) {
        return data[i * shape[1] + j];
    }
    const float& operator()(size_t i, size_t j) const {
        return data[i * shape[1] + j];
    }

    // 3D access [i, j, k]
    float& operator()(size_t i, size_t j, size_t k) {
        return data[(i * shape[1] + j) * shape[2] + k];
    }
    const float& operator()(size_t i, size_t j, size_t k) const {
        return data[(i * shape[1] + j) * shape[2] + k];
    }

    // 4D access [i, j, k, l]
    float& operator()(size_t i, size_t j, size_t k, size_t l) {
        return data[((i * shape[1] + j) * shape[2] + k) * shape[3] + l];
    }
    const float& operator()(size_t i, size_t j, size_t k, size_t l) const {
        return data[((i * shape[1] + j) * shape[2] + k) * shape[3] + l];
    }

    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }
};

// ============================================================================
// Activation functions
// ============================================================================

// Sigmoid: 1 / (1 + exp(-x))
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// Swish: x * sigmoid(x)
inline float swish(float x) {
    return x * sigmoid(x);
}

// Apply swish in-place to tensor
void swish_inplace(TensorF& x);

// Apply ReLU in-place
void relu_inplace(TensorF& x);

// ============================================================================
// Linear layers
// ============================================================================

// Linear layer: out = x @ W^T + bias
// x: [*, in_features]
// weight: [out_features, in_features]
// bias: [out_features] or nullptr
// out: [*, out_features]
void linear(
    const TensorF& x,
    const float* weight, size_t out_features, size_t in_features,
    const float* bias,
    TensorF& out
);

// Linear without bias
void linear_no_bias(
    const TensorF& x,
    const float* weight, size_t out_features, size_t in_features,
    TensorF& out
);

// ============================================================================
// Normalization
// ============================================================================

// Layer normalization
// x: [*, normalized_shape]
// weight, bias: [normalized_shape]
// eps: small value for numerical stability
void layer_norm(
    const TensorF& x,
    const float* weight, const float* bias,
    size_t normalized_shape,
    float eps,
    TensorF& out
);

// Layer norm in-place
void layer_norm_inplace(
    TensorF& x,
    const float* weight, const float* bias,
    size_t normalized_shape,
    float eps
);

// ============================================================================
// Convolution operations
// ============================================================================

// Conv1D: standard 1D convolution
// input: [batch, in_channels, length]
// weight: [out_channels, in_channels/groups, kernel_size]
// bias: [out_channels] or nullptr
// output: [batch, out_channels, out_length]
void conv1d(
    const TensorF& input,
    const float* weight, size_t out_channels, size_t in_channels,
    size_t kernel_size, size_t stride, size_t padding, size_t groups,
    const float* bias,
    TensorF& output
);

// Causal Conv1D: left-padded for streaming
// Automatically adds (kernel_size - 1) padding on the left
void causal_conv1d(
    const TensorF& input,
    const float* weight, size_t out_channels, size_t in_channels,
    size_t kernel_size, size_t stride, size_t groups,
    const float* bias,
    TensorF& output
);

// Conv2D: standard 2D convolution
// input: [batch, in_channels, height, width]
// weight: [out_channels, in_channels/groups, kH, kW]
// bias: [out_channels] or nullptr
void conv2d(
    const TensorF& input,
    const float* weight, size_t out_channels, size_t in_channels,
    size_t kH, size_t kW,
    size_t stride_h, size_t stride_w,
    size_t pad_h, size_t pad_w,
    size_t groups,
    const float* bias,
    TensorF& output
);

// Causal Conv2D: left-padded on time dimension
// Pads (kH-1, kW-1) on left/top, 0 on right/bottom
void causal_conv2d(
    const TensorF& input,
    const float* weight, size_t out_channels, size_t in_channels,
    size_t kH, size_t kW,
    size_t stride_h, size_t stride_w,
    size_t groups,
    const float* bias,
    TensorF& output
);

// ============================================================================
// Gated Linear Unit
// ============================================================================

// GLU: splits input along last dim, returns first_half * sigmoid(second_half)
// input: [*, 2*features]
// output: [*, features]
void glu(const TensorF& input, TensorF& output);

// ============================================================================
// LSTM operations
// ============================================================================

// LSTM hidden and cell state
struct LSTMState {
    TensorF h;  // [num_layers, batch, hidden_size]
    TensorF c;  // [num_layers, batch, hidden_size]
};

// Single LSTM cell forward pass
// input: [batch, input_size]
// h_prev, c_prev: [batch, hidden_size]
// weight_ih: [4*hidden_size, input_size]
// weight_hh: [4*hidden_size, hidden_size]
// bias_ih, bias_hh: [4*hidden_size]
// h_out, c_out: [batch, hidden_size]
void lstm_cell(
    const TensorF& input,
    const TensorF& h_prev, const TensorF& c_prev,
    const float* weight_ih, const float* weight_hh,
    const float* bias_ih, const float* bias_hh,
    size_t input_size, size_t hidden_size,
    TensorF& h_out, TensorF& c_out
);

// Multi-layer LSTM forward (single timestep)
// input: [batch, input_size]
// state: h and c for all layers
// Returns updated state
void lstm_forward(
    const TensorF& input,
    LSTMState& state,
    const float* const* weight_ih,  // array of weight pointers per layer
    const float* const* weight_hh,
    const float* const* bias_ih,
    const float* const* bias_hh,
    size_t input_size, size_t hidden_size,
    size_t num_layers,
    TensorF& output
);

// ============================================================================
// Embedding
// ============================================================================

// Embedding lookup
// indices: [batch] of token IDs
// weight: [vocab_size, embedding_dim]
// output: [batch, embedding_dim]
void embedding(
    const int* indices, size_t batch_size,
    const float* weight, size_t vocab_size, size_t embedding_dim,
    TensorF& output
);

// ============================================================================
// Utility operations
// ============================================================================

// Element-wise addition: out = a + b
void add(const TensorF& a, const TensorF& b, TensorF& out);

// Add in-place: a += b
void add_inplace(TensorF& a, const TensorF& b);

// Scale in-place: x *= scale
void scale_inplace(TensorF& x, float scale);

// Softmax along last dimension
void softmax(const TensorF& input, TensorF& output);

// Argmax along last dimension
// Returns indices of maximum values
void argmax(const TensorF& input, std::vector<int>& indices);

}  // namespace nemo
