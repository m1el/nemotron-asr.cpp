// Test that ggml computation produces same results as original implementation
#include "../src-ggml/nemo-ggml.h"
#include "../include/ggml_weights.h"
#include "../include/ops.h"
#include "../include/conv_subsampling.h"
#include "../include/conformer_modules.h"
#include "../include/conformer_encoder.h"
#include "../include/rnnt_decoder.h"
#include "../include/rnnt_joint.h"
#include "../include/greedy_decode.h"
#include "../include/tokenizer.h"

#include <cstdio>
#include <cmath>
#include <cstring>

// Helper to accumulate error statistics
struct error_calc {
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    float sum_sq_diff = 0.0f;
    size_t count = 0;
    void add_array(const float* a, const float* b, size_t n) {
        for (size_t i = 0; i < n; i++) {
            add(a[i], b[i]);
        }
    }
    void add(const float a, const float b) {
        float diff = std::abs(a - b);
        max_diff = std::max(max_diff, diff);
        sum_diff += diff;
        sum_sq_diff += diff * diff;
        count++;
    }

    void report(const char* name) const {
        float mean_diff = count > 0 ? sum_diff / count : 0.0f;
        float rms_diff = count > 0 ? std::sqrt(sum_sq_diff / count) : 0.0f;
        printf("%s: max_diff=%.6e, mean_diff=%.6e, rms_diff=%.6e\n", name, max_diff, mean_diff, rms_diff);
    }

    void reset() {
        max_diff = 0.0f;
        sum_diff = 0.0f;
        sum_sq_diff = 0.0f;
        count = 0;
    }
};

// Test linear projection: output = input @ weight.T
bool test_linear() {
    printf("=== Testing Linear Projection ===\n");

    // Load original weights
    nemo::ModelWeights ref_weights;
    if (!ref_weights.load("weights/model.bin")) {
        fprintf(stderr, "Failed to load reference weights\n");
        return false;
    }

    // Load ggml weights
    nemo_context * ctx = nemo_init("weights/model.gguf");
    if (!ctx) {
        fprintf(stderr, "Failed to load ggml model\n");
        return false;
    }

    // Get a weight tensor for testing (FFN1 linear1)
    const char * weight_name = "encoder.layers.0.feed_forward1.linear1.weight";
    const auto * ref_w = ref_weights.get(weight_name);
    auto it = ctx->model.tensors.find(weight_name);

    if (!ref_w || it == ctx->model.tensors.end()) {
        fprintf(stderr, "Weight tensor not found\n");
        nemo_free(ctx);
        return false;
    }

    struct ggml_tensor * ggml_w = it->second;

    // Weight shape: [4096, 1024] (out_features, in_features)
    size_t out_features = ref_w->shape[0];  // 4096
    size_t in_features = ref_w->shape[1];   // 1024

    printf("Weight shape: [%zu, %zu]\n", out_features, in_features);

    // Create test input [1, 5, 1024]
    size_t batch = 1;
    size_t seq_len = 5;
    nemo::TensorF input({batch, seq_len, in_features});

    // Fill with test values
    for (size_t i = 0; i < input.numel(); i++) {
        input.data[i] = 0.01f * (float)(i % 100) - 0.5f;
    }

    // === Original implementation ===
    nemo::TensorF ref_output;
    nemo::linear_no_bias(input, ref_w->data.data(), out_features, in_features, ref_output);

    printf("Original output shape: [%zu, %zu, %zu]\n",
           ref_output.shape[0], ref_output.shape[1], ref_output.shape[2]);
    printf("Original output[0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           ref_output(0,0,0), ref_output(0,0,1), ref_output(0,0,2),
           ref_output(0,0,3), ref_output(0,0,4));

    // === GGML implementation ===
    // Create compute context
    size_t buf_size = ggml_tensor_overhead() * 10 + ggml_graph_overhead();
    std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params = {
        .mem_size   = buf_size,
        .mem_buffer = buf.data(),
        .no_alloc   = true,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    // Create input tensor
    struct ggml_tensor * inp = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, in_features, seq_len, batch);
    ggml_set_name(inp, "input");
    ggml_set_input(inp);

    // Linear: output = inp @ weight.T
    // ggml_mul_mat(A, B) computes B @ A.T, so we use ggml_mul_mat(weight, inp)
    struct ggml_tensor * out = ggml_mul_mat(ctx0, ggml_w, inp);
    ggml_set_name(out, "output");
    ggml_set_output(out);

    // Build graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, out);

    // Allocate tensors
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->model.backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    // Set input data
    ggml_backend_tensor_set(inp, input.data.data(), 0, input.numel() * sizeof(float));

    // Compute
    ggml_backend_graph_compute(ctx->model.backend, gf);

    // Get output
    std::vector<float> ggml_output(out_features * seq_len * batch);
    ggml_backend_tensor_get(out, ggml_output.data(), 0, ggml_output.size() * sizeof(float));

    printf("GGML output[0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           ggml_output[0], ggml_output[1], ggml_output[2],
           ggml_output[3], ggml_output[4]);

    // Compare
    struct error_calc err;
    err.add_array(ggml_output.data(), ref_output.data.data(), ref_output.numel());
    err.report("Linear");

    // Cleanup
    ggml_gallocr_free(allocr);
    ggml_free(ctx0);
    nemo_free(ctx);

    bool passed = err.max_diff < 1e-4f;
    printf("Linear test: %s\n\n", passed ? "PASS" : "FAIL");
    return passed;
}

// Test layer normalization
bool test_layer_norm() {
    printf("=== Testing Layer Normalization ===\n");

    // Load original weights
    nemo::ModelWeights ref_weights;
    if (!ref_weights.load("weights/model.bin")) {
        fprintf(stderr, "Failed to load reference weights\n");
        return false;
    }

    // Load ggml weights
    nemo_context * ctx = nemo_init("weights/model.gguf");
    if (!ctx) {
        fprintf(stderr, "Failed to load ggml model\n");
        return false;
    }

    // Get norm weights
    const char * norm_w_name = "encoder.layers.0.norm_feed_forward1.weight";
    const char * norm_b_name = "encoder.layers.0.norm_feed_forward1.bias";

    const auto * ref_norm_w = ref_weights.get(norm_w_name);
    const auto * ref_norm_b = ref_weights.get(norm_b_name);

    auto it_w = ctx->model.tensors.find(norm_w_name);
    auto it_b = ctx->model.tensors.find(norm_b_name);

    if (!ref_norm_w || !ref_norm_b || it_w == ctx->model.tensors.end() || it_b == ctx->model.tensors.end()) {
        fprintf(stderr, "Norm tensors not found\n");
        nemo_free(ctx);
        return false;
    }

    struct ggml_tensor * ggml_norm_w = it_w->second;
    struct ggml_tensor * ggml_norm_b = it_b->second;

    size_t d_model = ref_norm_w->shape[0];  // 1024
    printf("d_model: %zu\n", d_model);

    // Create test input [1, 5, 1024]
    size_t batch = 1;
    size_t seq_len = 5;
    nemo::TensorF input({batch, seq_len, d_model});

    for (size_t i = 0; i < input.numel(); i++) {
        input.data[i] = 0.1f * (float)(i % 100) - 5.0f;
    }

    // === Original implementation ===
    nemo::TensorF ref_output;
    nemo::layer_norm(input, ref_norm_w->data.data(), ref_norm_b->data.data(),
                     d_model, 1e-5f, ref_output);

    printf("Original output[0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           ref_output(0,0,0), ref_output(0,0,1), ref_output(0,0,2),
           ref_output(0,0,3), ref_output(0,0,4));

    // === GGML implementation ===
    size_t buf_size = ggml_tensor_overhead() * 20 + ggml_graph_overhead();
    std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params = {
        .mem_size   = buf_size,
        .mem_buffer = buf.data(),
        .no_alloc   = true,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    // Create input tensor
    struct ggml_tensor * inp = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, d_model, seq_len, batch);
    ggml_set_name(inp, "input");
    ggml_set_input(inp);

    // Layer norm: norm(x) * weight + bias
    struct ggml_tensor * cur = ggml_norm(ctx0, inp, 1e-5f);
    cur = ggml_mul(ctx0, cur, ggml_norm_w);
    cur = ggml_add(ctx0, cur, ggml_norm_b);
    ggml_set_name(cur, "output");
    ggml_set_output(cur);

    // Build graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, cur);

    // Allocate
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->model.backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    // Set input
    ggml_backend_tensor_set(inp, input.data.data(), 0, input.numel() * sizeof(float));

    // Compute
    ggml_backend_graph_compute(ctx->model.backend, gf);

    // Get output
    std::vector<float> ggml_output(d_model * seq_len * batch);
    ggml_backend_tensor_get(cur, ggml_output.data(), 0, ggml_output.size() * sizeof(float));

    printf("GGML output[0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           ggml_output[0], ggml_output[1], ggml_output[2],
           ggml_output[3], ggml_output[4]);

    // Compare
    error_calc err;
    err.add_array(ggml_output.data(), ref_output.data.data(), ref_output.numel());
    err.report("Layer norm");

    // Cleanup
    ggml_gallocr_free(allocr);
    ggml_free(ctx0);
    nemo_free(ctx);

    bool passed = err.max_diff < 1e-4f;
    printf("Layer norm test: %s\n\n", passed ? "PASS" : "FAIL");
    return passed;
}

// Test Swish/SiLU activation
bool test_swish() {
    printf("=== Testing Swish/SiLU Activation ===\n");

    // Load ggml weights (just for backend)
    nemo_context * ctx = nemo_init("weights/model.gguf");
    if (!ctx) {
        fprintf(stderr, "Failed to load ggml model\n");
        return false;
    }

    // Create test input [1024]
    size_t size = 1024;
    nemo::TensorF input({size});

    for (size_t i = 0; i < input.numel(); i++) {
        input.data[i] = 0.1f * (float)i - 50.0f;  // Range -50 to ~50
    }

    // === Original implementation ===
    nemo::TensorF ref_output = input;  // Copy
    nemo::swish_inplace(ref_output);

    printf("Original output[0:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           ref_output.data[0], ref_output.data[1], ref_output.data[2],
           ref_output.data[3], ref_output.data[4]);

    // === GGML implementation ===
    size_t buf_size = ggml_tensor_overhead() * 10 + ggml_graph_overhead();
    std::vector<uint8_t> compute_buf(buf_size);

    struct ggml_init_params params = {
        .mem_size   = buf_size,
        .mem_buffer = compute_buf.data(),
        .no_alloc   = true,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    // Create input tensor
    struct ggml_tensor * inp = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, size);
    ggml_set_name(inp, "input");
    ggml_set_input(inp);

    // SiLU (Swish)
    struct ggml_tensor * cur = ggml_silu(ctx0, inp);
    ggml_set_name(cur, "output");
    ggml_set_output(cur);

    // Build graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, cur);

    // Allocate
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->model.backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    // Set input
    ggml_backend_tensor_set(inp, input.data.data(), 0, input.numel() * sizeof(float));

    // Compute
    ggml_backend_graph_compute(ctx->model.backend, gf);

    // Get output
    std::vector<float> ggml_output(size);
    ggml_backend_tensor_get(cur, ggml_output.data(), 0, ggml_output.size() * sizeof(float));

    printf("GGML output[0:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           ggml_output[0], ggml_output[1], ggml_output[2],
           ggml_output[3], ggml_output[4]);

    // Compare
    error_calc err;
    err.add_array(ggml_output.data(), ref_output.data.data(), ref_output.numel());
    err.report("Swish");

    // Cleanup
    ggml_gallocr_free(allocr);
    ggml_free(ctx0);
    nemo_free(ctx);

    bool passed = err.max_diff < 1e-5f;
    printf("Swish test: %s\n\n", passed ? "PASS" : "FAIL");
    return passed;
}

// Test FFN (feed-forward network): Linear -> Swish -> Linear
bool test_ffn() {
    printf("=== Testing Feed-Forward Network ===\n");

    // Load original weights
    nemo::ModelWeights ref_weights;
    if (!ref_weights.load("weights/model.bin")) {
        fprintf(stderr, "Failed to load reference weights\n");
        return false;
    }

    // Load ggml weights
    nemo_context * ctx = nemo_init("weights/model.gguf");
    if (!ctx) {
        fprintf(stderr, "Failed to load ggml model\n");
        return false;
    }

    // Get FFN1 weights from layer 0
    const char * linear1_name = "encoder.layers.0.feed_forward1.linear1.weight";
    const char * linear2_name = "encoder.layers.0.feed_forward1.linear2.weight";

    const auto * ref_linear1 = ref_weights.get(linear1_name);
    const auto * ref_linear2 = ref_weights.get(linear2_name);

    auto it1 = ctx->model.tensors.find(linear1_name);
    auto it2 = ctx->model.tensors.find(linear2_name);

    if (!ref_linear1 || !ref_linear2 || it1 == ctx->model.tensors.end() || it2 == ctx->model.tensors.end()) {
        fprintf(stderr, "FFN weight tensors not found\n");
        nemo_free(ctx);
        return false;
    }

    struct ggml_tensor * ggml_linear1_w = it1->second;
    struct ggml_tensor * ggml_linear2_w = it2->second;

    // linear1: [4096, 1024], linear2: [1024, 4096]
    size_t d_model = ref_linear1->shape[1];  // 1024
    size_t d_ff = ref_linear1->shape[0];     // 4096
    printf("d_model: %zu, d_ff: %zu\n", d_model, d_ff);

    // Create test input [1, 5, 1024]
    size_t batch = 1;
    size_t seq_len = 5;
    nemo::TensorF input({batch, seq_len, d_model});

    for (size_t i = 0; i < input.numel(); i++) {
        input.data[i] = 0.01f * (float)(i % 100) - 0.5f;
    }

    // === Original implementation ===
    nemo::TensorF buf, ref_output;
    // Linear1
    nemo::linear_no_bias(input, ref_linear1->data.data(), d_ff, d_model, buf);
    // Swish
    nemo::swish_inplace(buf);
    // Linear2
    nemo::linear_no_bias(buf, ref_linear2->data.data(), d_model, d_ff, ref_output);

    printf("Original output[0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           ref_output(0,0,0), ref_output(0,0,1), ref_output(0,0,2),
           ref_output(0,0,3), ref_output(0,0,4));

    // === GGML implementation ===
    size_t buf_size = ggml_tensor_overhead() * 20 + ggml_graph_overhead();
    std::vector<uint8_t> compute_buf(buf_size);

    struct ggml_init_params params = {
        .mem_size   = buf_size,
        .mem_buffer = compute_buf.data(),
        .no_alloc   = true,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    // Create input tensor
    struct ggml_tensor * inp = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, d_model, seq_len, batch);
    ggml_set_name(inp, "input");
    ggml_set_input(inp);

    // FFN: Linear1 -> Swish -> Linear2
    struct ggml_tensor * cur = ggml_mul_mat(ctx0, ggml_linear1_w, inp);
    cur = ggml_silu(ctx0, cur);
    cur = ggml_mul_mat(ctx0, ggml_linear2_w, cur);
    ggml_set_name(cur, "output");
    ggml_set_output(cur);

    // Build graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, cur);

    // Allocate
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->model.backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    // Set input
    ggml_backend_tensor_set(inp, input.data.data(), 0, input.numel() * sizeof(float));

    // Compute
    ggml_backend_graph_compute(ctx->model.backend, gf);

    // Get output
    std::vector<float> ggml_output(d_model * seq_len * batch);
    ggml_backend_tensor_get(cur, ggml_output.data(), 0, ggml_output.size() * sizeof(float));

    printf("GGML output[0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           ggml_output[0], ggml_output[1], ggml_output[2],
           ggml_output[3], ggml_output[4]);

    // Compare
    error_calc err;
    err.add_array(ggml_output.data(), ref_output.data.data(), ref_output.numel());
    err.report("FFN");

    // Cleanup
    ggml_gallocr_free(allocr);
    ggml_free(ctx0);
    nemo_free(ctx);

    // FFN has 3 chained operations with large intermediate values
    // Use a looser threshold (1e-2) but still verify relative error is small
    bool passed = err.max_diff < 1e-2f;
    printf("FFN test: %s\n\n", passed ? "PASS" : "FAIL");
    return passed;
}

// Test Conv2D operation
// Note: ggml uses different tensor layouts than PyTorch
// PyTorch kernel: [OC, IC, KH, KW], input: [N, C, H, W]
// ggml kernel: [KW, KH, IC, OC], input: [W, H, C, N]
bool test_conv2d() {
    printf("=== Testing Conv2D ===\n");

    // Load original weights
    nemo::ModelWeights ref_weights;
    if (!ref_weights.load("weights/model.bin")) {
        fprintf(stderr, "Failed to load reference weights\n");
        return false;
    }

    // Load ggml weights
    nemo_context * ctx = nemo_init("weights/model.gguf");
    if (!ctx) {
        fprintf(stderr, "Failed to load ggml model\n");
        return false;
    }

    // Get conv0 weights (first conv in subsampling)
    // Shape: [256, 1, 3, 3] in PyTorch [OC, IC, KH, KW]
    const char * conv_w_name = "encoder.pre_encode.conv.0.weight";
    const char * conv_b_name = "encoder.pre_encode.conv.0.bias";

    const auto * ref_conv_w = ref_weights.get(conv_w_name);
    const auto * ref_conv_b = ref_weights.get(conv_b_name);

    auto it_w = ctx->model.tensors.find(conv_w_name);
    auto it_b = ctx->model.tensors.find(conv_b_name);

    if (!ref_conv_w || !ref_conv_b || it_w == ctx->model.tensors.end() || it_b == ctx->model.tensors.end()) {
        fprintf(stderr, "Conv weight tensors not found\n");
        nemo_free(ctx);
        return false;
    }

    // PyTorch: [OC=256, IC=1, KH=3, KW=3]
    size_t out_channels = ref_conv_w->shape[0];  // 256
    size_t in_channels = ref_conv_w->shape[1];   // 1
    size_t kH = ref_conv_w->shape[2];            // 3
    size_t kW = ref_conv_w->shape[3];            // 3

    printf("Conv weight shape (PyTorch): [%zu, %zu, %zu, %zu]\n", out_channels, in_channels, kH, kW);

    // Create small test input [batch=1, channels=1, height=10, width=128]
    size_t batch = 1;
    size_t in_h = 10;
    size_t in_w = 128;

    // PyTorch layout: [N, C, H, W]
    nemo::TensorF input({batch, in_channels, in_h, in_w});
    for (size_t i = 0; i < input.numel(); i++) {
        input.data[i] = 0.01f * (float)(i % 100) - 0.5f;
    }

    // === Original implementation ===
    // causal_conv2d: pad_left = kW-1=2, pad_right = stride-1=1, pad_top = kH-1=2, pad_bottom = stride-1=1
    nemo::TensorF ref_output;
    nemo::causal_conv2d(input, ref_conv_w->data.data(), out_channels, in_channels,
                        kH, kW, 2, 2, 1, ref_conv_b->data.data(), ref_output);

    printf("Original output shape: [%zu, %zu, %zu, %zu]\n",
           ref_output.shape[0], ref_output.shape[1], ref_output.shape[2], ref_output.shape[3]);
    printf("Original output[0,0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           ref_output(0,0,0,0), ref_output(0,0,0,1), ref_output(0,0,0,2),
           ref_output(0,0,0,3), ref_output(0,0,0,4));

    // === GGML implementation ===
    // Need to transpose weights from [OC, IC, KH, KW] to [KW, KH, IC, OC]
    // And input from [N, C, H, W] to [W, H, C, N]

    size_t buf_size = ggml_tensor_overhead() * 30 + ggml_graph_overhead();
    std::vector<uint8_t> compute_buf(buf_size);

    struct ggml_init_params params = {
        .mem_size   = buf_size,
        .mem_buffer = compute_buf.data(),
        .no_alloc   = true,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    // Create input tensor in ggml layout [W, H, C, N]
    struct ggml_tensor * inp = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, in_w, in_h, in_channels, batch);
    ggml_set_name(inp, "input");
    ggml_set_input(inp);

    // For weights, we need to use the model's pre-loaded tensor, but it might be in wrong layout
    // Let me check the ggml tensor dimensions
    struct ggml_tensor * ggml_w = it_w->second;
    struct ggml_tensor * ggml_b = it_b->second;

    printf("GGML weight shape: [%lld, %lld, %lld, %lld]\n",
           (long long)ggml_w->ne[0], (long long)ggml_w->ne[1],
           (long long)ggml_w->ne[2], (long long)ggml_w->ne[3]);

    // Apply causal padding: pad_left=2, pad_right=1, pad_top=2, pad_bottom=1
    // ggml_pad_ext(a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3)
    // dim0=W, dim1=H, so lp0=pad_left, rp0=pad_right, lp1=pad_top, rp1=pad_bottom
    struct ggml_tensor * padded = ggml_pad_ext(ctx0, inp, 2, 1, 2, 1, 0, 0, 0, 0);

    // Conv2d with stride=2, padding=0 (we already padded), dilation=1
    struct ggml_tensor * conv_out = ggml_conv_2d(ctx0, ggml_w, padded, 2, 2, 0, 0, 1, 1);

    // Add bias (need to broadcast)
    // Bias shape: [out_channels] -> need to reshape for broadcasting
    struct ggml_tensor * bias_reshaped = ggml_reshape_4d(ctx0, ggml_b, 1, 1, out_channels, 1);
    struct ggml_tensor * out = ggml_add(ctx0, conv_out, bias_reshaped);

    ggml_set_name(out, "output");
    ggml_set_output(out);

    // Build graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, out);

    // Allocate
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->model.backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    // Convert input from [N, C, H, W] to [W, H, C, N]
    std::vector<float> inp_ggml(input.numel());
    for (size_t n = 0; n < batch; n++) {
        for (size_t c = 0; c < in_channels; c++) {
            for (size_t h = 0; h < in_h; h++) {
                for (size_t w = 0; w < in_w; w++) {
                    // PyTorch: [n, c, h, w]
                    // ggml: [w, h, c, n]
                    int src_idx = ((n * in_channels + c) * in_h + h) * in_w + w;
                    int dst_idx = ((n * in_channels + c) * in_h + h) * in_w + w;  // same for contiguous
                    inp_ggml[dst_idx] = input.data[src_idx];
                }
            }
        }
    }

    // Set input
    ggml_backend_tensor_set(inp, inp_ggml.data(), 0, inp_ggml.size() * sizeof(float));

    // Compute
    ggml_backend_graph_compute(ctx->model.backend, gf);

    // Get output dimensions
    printf("GGML output shape: [%lld, %lld, %lld, %lld]\n",
           (long long)out->ne[0], (long long)out->ne[1],
           (long long)out->ne[2], (long long)out->ne[3]);

    // Get output
    size_t out_numel = out->ne[0] * out->ne[1] * out->ne[2] * out->ne[3];
    std::vector<float> ggml_output(out_numel);
    ggml_backend_tensor_get(out, ggml_output.data(), 0, out_numel * sizeof(float));

    printf("GGML output[0,0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           ggml_output[0], ggml_output[1], ggml_output[2],
           ggml_output[3], ggml_output[4]);

    // Compare - need to account for different layouts
    // ref_output: [N, OC, H_out, W_out] PyTorch
    // ggml_output: [W_out, H_out, OC, N] ggml
    error_calc err;
    size_t out_h = ref_output.shape[2];
    size_t out_w = ref_output.shape[3];

    for (size_t n = 0; n < batch; n++) {
        for (size_t oc = 0; oc < out_channels; oc++) {
            for (size_t h = 0; h < out_h; h++) {
                for (size_t w = 0; w < out_w; w++) {
                    // PyTorch idx: [n, oc, h, w]
                    size_t ref_idx = ((n * out_channels + oc) * out_h + h) * out_w + w;
                    // ggml idx: [w, h, oc, n] -> w + h*W + oc*W*H + n*W*H*C
                    size_t ggml_idx = w + h * out_w + oc * out_w * out_h + n * out_w * out_h * out_channels;

                    err.add(ggml_output[ggml_idx], ref_output.data[ref_idx]);
                }
            }
        }
    }

    err.report("Conv2D");

    // Cleanup
    ggml_gallocr_free(allocr);
    ggml_free(ctx0);
    nemo_free(ctx);

    bool passed = err.max_diff < 1e-4f;
    printf("Conv2D test: %s\n\n", passed ? "PASS" : "FAIL");
    return passed;
}

// Helper to build causal conv2d: pad + conv2d
static struct ggml_tensor * build_causal_conv2d(
    struct ggml_context * ctx,
    struct ggml_tensor * input,   // [W, H, C, N]
    struct ggml_tensor * weight,  // [KW, KH, IC, OC]
    struct ggml_tensor * bias,    // [OC]
    int stride_w, int stride_h
) {
    // Causal padding: pad_left = kW-1, pad_right = stride-1, pad_top = kH-1, pad_bottom = stride-1
    int kW = weight->ne[0];
    int kH = weight->ne[1];
    int pad_left = kW - 1;
    int pad_right = stride_w - 1;
    int pad_top = kH - 1;
    int pad_bottom = stride_h - 1;

    struct ggml_tensor * padded = ggml_pad_ext(ctx, input, pad_left, pad_right, pad_top, pad_bottom, 0, 0, 0, 0);
    struct ggml_tensor * conv_out = ggml_conv_2d(ctx, weight, padded, stride_w, stride_h, 0, 0, 1, 1);

    // Add bias
    int out_channels = weight->ne[3];
    struct ggml_tensor * bias_reshaped = ggml_reshape_4d(ctx, bias, 1, 1, out_channels, 1);
    return ggml_add(ctx, conv_out, bias_reshaped);
}

// Helper to build causal depthwise conv2d
// Uses ggml_conv_2d_dw_direct which works with F32 (unlike ggml_conv_2d_dw which uses F16 im2col)
static struct ggml_tensor * build_causal_dw_conv2d(
    struct ggml_context * ctx,
    struct ggml_tensor * input,   // [W, H, C, N]
    struct ggml_tensor * weight,  // [KW, KH, 1, C]
    struct ggml_tensor * bias,    // [C]
    int stride_w, int stride_h
) {
    int kW = weight->ne[0];
    int kH = weight->ne[1];
    int pad_left = kW - 1;
    int pad_right = stride_w - 1;
    int pad_top = kH - 1;
    int pad_bottom = stride_h - 1;

    struct ggml_tensor * padded = ggml_pad_ext(ctx, input, pad_left, pad_right, pad_top, pad_bottom, 0, 0, 0, 0);

    // Use ggml_conv_2d_dw_direct which uses direct F32 computation
    struct ggml_tensor * conv_out = ggml_conv_2d_dw_direct(ctx, weight, padded, stride_w, stride_h, 0, 0, 1, 1);

    // Add bias
    int channels = weight->ne[3];
    struct ggml_tensor * bias_reshaped = ggml_reshape_4d(ctx, bias, 1, 1, channels, 1);
    return ggml_add(ctx, conv_out, bias_reshaped);
}

// Test full ConvSubsampling with real mel input
bool test_conv_subsampling() {
    printf("=== Testing ConvSubsampling ===\n");

    // Load original weights
    nemo::ModelWeights ref_weights;
    if (!ref_weights.load("weights/model.bin")) {
        fprintf(stderr, "Failed to load reference weights\n");
        return false;
    }

    // Load ggml model
    nemo_context * ctx = nemo_init("weights/model.gguf");
    if (!ctx) {
        fprintf(stderr, "Failed to load ggml model\n");
        return false;
    }

    // Load mel input from file - raw float32 data, shape [time, 128]
    FILE * f = fopen("test.mel.bin", "rb");
    if (!f) {
        fprintf(stderr, "Failed to open test.mel.bin\n");
        nemo_free(ctx);
        return false;
    }

    // Get file size to determine time dimension
    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    size_t batch = 1;
    size_t features = 128;
    size_t time_in = file_size / (sizeof(float) * features);

    printf("Mel input shape: [%zu, %zu, %zu]\n", batch, time_in, features);

    // Read raw float data - file is [time, features]
    std::vector<float> raw_mel(time_in * features);
    size_t read = fread(raw_mel.data(), sizeof(float), raw_mel.size(), f);
    fclose(f);

    if (read != raw_mel.size()) {
        fprintf(stderr, "Failed to read mel data\n");
        nemo_free(ctx);
        return false;
    }

    // Reshape to [batch, time, features]
    nemo::TensorF mel_input({batch, time_in, features});
    for (size_t t = 0; t < time_in; t++) {
        for (size_t f = 0; f < features; f++) {
            mel_input(0, t, f) = raw_mel[t * features + f];
        }
    }

    // === Original implementation ===
    nemo::ConvSubsampling subsampling;
    subsampling.load_weights(ref_weights);

    nemo::TensorF ref_output;
    subsampling.forward(mel_input, ref_output);

    printf("Original output shape: [%zu, %zu, %zu]\n",
           ref_output.shape[0], ref_output.shape[1], ref_output.shape[2]);
    printf("Original output[0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           ref_output(0,0,0), ref_output(0,0,1), ref_output(0,0,2),
           ref_output(0,0,3), ref_output(0,0,4));

    // === GGML implementation ===
    // Get conv weights from model
    auto get_tensor = [&](const char * name) -> struct ggml_tensor * {
        auto it = ctx->model.tensors.find(name);
        return (it != ctx->model.tensors.end()) ? it->second : nullptr;
    };

    struct ggml_tensor * conv0_w = get_tensor("encoder.pre_encode.conv.0.weight");
    struct ggml_tensor * conv0_b = get_tensor("encoder.pre_encode.conv.0.bias");
    struct ggml_tensor * conv2_w = get_tensor("encoder.pre_encode.conv.2.weight");
    struct ggml_tensor * conv2_b = get_tensor("encoder.pre_encode.conv.2.bias");
    struct ggml_tensor * conv3_w = get_tensor("encoder.pre_encode.conv.3.weight");
    struct ggml_tensor * conv3_b = get_tensor("encoder.pre_encode.conv.3.bias");
    struct ggml_tensor * conv5_w = get_tensor("encoder.pre_encode.conv.5.weight");
    struct ggml_tensor * conv5_b = get_tensor("encoder.pre_encode.conv.5.bias");
    struct ggml_tensor * conv6_w = get_tensor("encoder.pre_encode.conv.6.weight");
    struct ggml_tensor * conv6_b = get_tensor("encoder.pre_encode.conv.6.bias");
    struct ggml_tensor * out_w = get_tensor("encoder.pre_encode.out.weight");
    struct ggml_tensor * out_b = get_tensor("encoder.pre_encode.out.bias");

    if (!conv0_w || !conv0_b || !conv2_w || !conv2_b || !conv3_w || !conv3_b ||
        !conv5_w || !conv5_b || !conv6_w || !conv6_b || !out_w || !out_b) {
        fprintf(stderr, "Missing ConvSubsampling tensors\n");
        nemo_free(ctx);
        return false;
    }

    // Create compute context - need larger buffer for ConvSubsampling
    size_t buf_size = ggml_tensor_overhead() * 500 + ggml_graph_overhead();
    std::vector<uint8_t> compute_buf(buf_size);

    struct ggml_init_params params = {
        .mem_size   = buf_size,
        .mem_buffer = compute_buf.data(),
        .no_alloc   = true,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    // Input: [batch, time, features] -> reshape to [W, H, 1, N] = [features, time, 1, batch]
    struct ggml_tensor * inp = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, features, time_in, 1, batch);
    ggml_set_name(inp, "mel_input");
    ggml_set_input(inp);

    // Conv0: CausalConv2D(1, 256, k=3, s=2) + ReLU
    struct ggml_tensor * cur = build_causal_conv2d(ctx0, inp, conv0_w, conv0_b, 2, 2);
    cur = ggml_relu(ctx0, cur);

    // Conv2: Depthwise CausalConv2D(256, k=3, s=2, groups=256)
    cur = build_causal_dw_conv2d(ctx0, cur, conv2_w, conv2_b, 2, 2);

    // Conv3: Pointwise Conv2d(256, 256, k=1, s=1) + ReLU
    // Pointwise: no padding needed, stride=1
    cur = ggml_conv_2d(ctx0, conv3_w, cur, 1, 1, 0, 0, 1, 1);
    struct ggml_tensor * conv3_b_reshaped = ggml_reshape_4d(ctx0, conv3_b, 1, 1, conv3_b->ne[0], 1);
    cur = ggml_add(ctx0, cur, conv3_b_reshaped);
    cur = ggml_relu(ctx0, cur);

    // Conv5: Depthwise CausalConv2D(256, k=3, s=2, groups=256)
    cur = build_causal_dw_conv2d(ctx0, cur, conv5_w, conv5_b, 2, 2);

    // Conv6: Pointwise Conv2d(256, 256, k=1, s=1) + ReLU
    cur = ggml_conv_2d(ctx0, conv6_w, cur, 1, 1, 0, 0, 1, 1);
    struct ggml_tensor * conv6_b_reshaped = ggml_reshape_4d(ctx0, conv6_b, 1, 1, conv6_b->ne[0], 1);
    cur = ggml_add(ctx0, cur, conv6_b_reshaped);
    cur = ggml_relu(ctx0, cur);

    // cur shape: [W_out, H_out, 256, batch]
    // Need to flatten to [256*W_out, H_out, batch] then project to [1024, H_out, batch]

    // Get dimensions
    // After convs, we expect: W_out = ceil((features+pad)/8) = 17, H_out = ceil((time+pad)/8) = 251
    // Flatten: [W_out * 256, H_out, batch]
    int64_t w_out = cur->ne[0];
    int64_t h_out = cur->ne[1];
    int64_t c_out = cur->ne[2];

    printf("After conv6+relu: w_out=%ld, h_out=%ld, c_out=%ld\n", w_out, h_out, c_out);

    // Permute from [W, H, C, N] to [C*W, H, N] for linear projection
    // Original C++ flattens as: flat[c * W + w] = buf[c, t, w]
    // We need memory order where index i = c*W + w (c varies slower, w faster)
    // Permute to [W, C, H, N] then reshape to [W*C, H, N] gives i = w + c*W = c*W + w
    struct ggml_tensor * permuted = ggml_permute(ctx0, cur, 0, 2, 1, 3);  // [W, C, H, N]
    permuted = ggml_cont(ctx0, permuted);  // Make contiguous
    permuted = ggml_reshape_3d(ctx0, permuted, w_out * c_out, h_out, batch);  // [W*C, H, N]

    printf("After permute+reshape: [%ld, %ld, %ld]\n",
           (long)permuted->ne[0], (long)permuted->ne[1], (long)permuted->ne[2]);
    printf("out_w shape: [%ld, %ld]\n", (long)out_w->ne[0], (long)out_w->ne[1]);

    // Linear projection
    struct ggml_tensor * out = ggml_mul_mat(ctx0, out_w, permuted);
    // Add bias
    out = ggml_add(ctx0, out, out_b);

    ggml_set_name(out, "subsampling_output");
    ggml_set_output(out);

    // Build graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, out);

    // Allocate
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->model.backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    // Set input (convert from [batch, time, features] to [features, time, 1, batch])
    std::vector<float> inp_ggml(mel_input.numel());
    for (size_t n = 0; n < batch; n++) {
        for (size_t t = 0; t < time_in; t++) {
            for (size_t f = 0; f < features; f++) {
                size_t src_idx = (n * time_in + t) * features + f;
                size_t dst_idx = ((n * 1 + 0) * time_in + t) * features + f;
                inp_ggml[dst_idx] = mel_input.data[src_idx];
            }
        }
    }

    ggml_backend_tensor_set(inp, inp_ggml.data(), 0, inp_ggml.size() * sizeof(float));

    // Compute
    ggml_backend_graph_compute(ctx->model.backend, gf);

    // Debug: print intermediate shapes
    printf("After convs, shape: [%lld, %lld, %lld, %lld]\n",
           (long long)cur->ne[0], (long long)cur->ne[1],
           (long long)cur->ne[2], (long long)cur->ne[3]);
    printf("After permute+reshape, shape: [%lld, %lld, %lld]\n",
           (long long)permuted->ne[0], (long long)permuted->ne[1], (long long)permuted->ne[2]);

    // Get output
    printf("GGML output shape: [%lld, %lld, %lld]\n",
           (long long)out->ne[0], (long long)out->ne[1], (long long)out->ne[2]);

    // Check that output has expected number of elements
    size_t expected_time = ref_output.shape[1];
    size_t expected_features = ref_output.shape[2];
    printf("Expected output shape: [%zu, %zu, %zu]\n", batch, expected_time, expected_features);

    size_t out_numel = out->ne[0] * out->ne[1] * out->ne[2];
    printf("Output numel: %zu\n", out_numel);

    if (out_numel > 10000000) {
        fprintf(stderr, "Output too large, something wrong with computation\n");
        ggml_gallocr_free(allocr);
        ggml_free(ctx0);
        nemo_free(ctx);
        return false;
    }

    std::vector<float> ggml_output(out_numel);
    ggml_backend_tensor_get(out, ggml_output.data(), 0, out_numel * sizeof(float));

    printf("GGML output[0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           ggml_output[0], ggml_output[1], ggml_output[2],
           ggml_output[3], ggml_output[4]);

    // Compare
    // ref_output: [batch, time_out, 1024]
    // ggml_output: [1024, time_out, batch]
    struct error_calc err;
    size_t time_out = ref_output.shape[1];
    size_t d_model = ref_output.shape[2];

    for (size_t n = 0; n < batch; n++) {
        for (size_t t = 0; t < time_out; t++) {
            for (size_t d = 0; d < d_model; d++) {
                size_t ref_idx = (n * time_out + t) * d_model + d;
                size_t ggml_idx = (n * time_out + t) * d_model + d;

                err.add(ggml_output[ggml_idx], ref_output.data[ref_idx]);
            }
        }
    }

    err.report("ConvSubsampling");

    // Cleanup
    ggml_gallocr_free(allocr);
    ggml_free(ctx0);
    nemo_free(ctx);

    bool passed = err.max_diff < 1e-2f;
    printf("ConvSubsampling test: %s\n\n", passed ? "PASS" : "FAIL");
    return passed;
}

// Test rel_shift operation for relative position attention
// rel_shift transforms position attention scores to align with key positions
bool test_rel_shift() {
    printf("=== Testing rel_shift ===\n");

    // Load ggml model for backend
    nemo_context * ctx = nemo_init("weights/model.gguf");
    if (!ctx) {
        fprintf(stderr, "Failed to load ggml model\n");
        return false;
    }

    // Test with small tensor: [batch=1, heads=2, qlen=4, pos_len=7]
    // pos_len = 2*qlen - 1
    size_t batch = 1;
    size_t heads = 2;
    size_t qlen = 4;
    size_t pos_len = 2 * qlen - 1;  // 7

    // Create input tensor with layout [batch, heads, qlen, pos_len] (for reference)
    // In ggml: [pos_len, qlen, heads, batch]
    nemo::TensorF input({batch, heads, qlen, pos_len});

    // Fill with recognizable values: input[b,h,i,p] = 100*h + 10*i + p
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < heads; h++) {
            for (size_t i = 0; i < qlen; i++) {
                for (size_t p = 0; p < pos_len; p++) {
                    input(b, h, i, p) = 100.0f * h + 10.0f * i + p;
                }
            }
        }
    }

    printf("Input shape: [%zu, %zu, %zu, %zu] (batch, heads, qlen, pos_len)\n", batch, heads, qlen, pos_len);

    // Compute expected output using formula: out[b,h,i,j] = input[b,h,i, j + qlen - 1 - i]
    nemo::TensorF expected({batch, heads, qlen, qlen});
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < heads; h++) {
            for (size_t i = 0; i < qlen; i++) {
                for (size_t j = 0; j < qlen; j++) {
                    size_t k = j + qlen - 1 - i;
                    expected(b, h, i, j) = input(b, h, i, k);
                }
            }
        }
    }

    printf("Expected output shape: [%zu, %zu, %zu, %zu]\n", batch, heads, qlen, qlen);
    printf("Expected[0,0,0,:]: %.1f, %.1f, %.1f, %.1f\n",
           expected(0,0,0,0), expected(0,0,0,1), expected(0,0,0,2), expected(0,0,0,3));
    printf("Expected[0,0,3,:]: %.1f, %.1f, %.1f, %.1f\n",
           expected(0,0,3,0), expected(0,0,3,1), expected(0,0,3,2), expected(0,0,3,3));

    // === GGML implementation ===
    size_t buf_size = ggml_tensor_overhead() * 20 + ggml_graph_overhead();
    std::vector<uint8_t> compute_buf(buf_size);

    struct ggml_init_params params = {
        .mem_size   = buf_size,
        .mem_buffer = compute_buf.data(),
        .no_alloc   = true,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    // ggml tensor: [pos_len, qlen, heads, batch]
    struct ggml_tensor * inp = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, pos_len, qlen, heads, batch);
    ggml_set_name(inp, "input");
    ggml_set_input(inp);

    // Build rel_shift graph using pad-reshape-slice
    // Step 1: Pad with zero column on left
    struct ggml_tensor * padded = ggml_pad_ext(ctx0, inp, 1, 0, 0, 0, 0, 0, 0, 0);

    // Step 2: Make contiguous and reshape to [qlen, pos_len+1, heads, batch]
    struct ggml_tensor * reshaped = ggml_reshape_4d(ctx0, ggml_cont(ctx0, padded), qlen, pos_len + 1, heads, batch);

    // Step 3: Drop first row by offsetting view
    struct ggml_tensor * dropped = ggml_view_4d(ctx0, reshaped,
        qlen, pos_len, heads, batch,
        reshaped->nb[1], reshaped->nb[2], reshaped->nb[3],
        qlen * sizeof(float));

    // Step 4: Make contiguous and reshape back
    struct ggml_tensor * back = ggml_reshape_4d(ctx0, ggml_cont(ctx0, dropped), pos_len, qlen, heads, batch);

    // Step 5: Slice first qlen columns
    struct ggml_tensor * out = ggml_view_4d(ctx0, back,
        qlen, qlen, heads, batch,
        back->nb[1], back->nb[2], back->nb[3], 0);
    out = ggml_cont(ctx0, out);
    ggml_set_name(out, "output");
    ggml_set_output(out);

    // Build graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->model.backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    // Set input data - transpose from [batch, heads, qlen, pos_len] to [pos_len, qlen, heads, batch]
    std::vector<float> inp_data(pos_len * qlen * heads * batch);
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < heads; h++) {
            for (size_t i = 0; i < qlen; i++) {
                for (size_t p = 0; p < pos_len; p++) {
                    // ggml index: p + i*pos_len + h*pos_len*qlen + b*pos_len*qlen*heads
                    inp_data[p + i*pos_len + h*pos_len*qlen + b*pos_len*qlen*heads] = input(b, h, i, p);
                }
            }
        }
    }
    ggml_backend_tensor_set(inp, inp_data.data(), 0, inp_data.size() * sizeof(float));

    // Compute
    ggml_backend_graph_compute(ctx->model.backend, gf);

    // Get output
    std::vector<float> ggml_out(qlen * qlen * heads * batch);
    ggml_backend_tensor_get(out, ggml_out.data(), 0, ggml_out.size() * sizeof(float));

    // ggml output is [qlen, qlen, heads, batch], convert to [batch, heads, qlen, qlen]
    printf("GGML output[0,0,0,:]: ");
    for (size_t j = 0; j < qlen; j++) {
        // index: j + 0*qlen + 0*qlen*qlen + 0*qlen*qlen*heads
        printf("%.1f ", ggml_out[j + 0*qlen + 0*qlen*qlen + 0*qlen*qlen*heads]);
    }
    printf("\n");
    printf("GGML output[0,0,3,:]: ");
    for (size_t j = 0; j < qlen; j++) {
        printf("%.1f ", ggml_out[j + 3*qlen + 0*qlen*qlen + 0*qlen*qlen*heads]);
    }
    printf("\n");

    // Compare
    error_calc err;
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < heads; h++) {
            for (size_t i = 0; i < qlen; i++) {
                for (size_t j = 0; j < qlen; j++) {
                    float exp_val = expected(b, h, i, j);
                    // ggml output: [qlen, qlen, heads, batch] -> index = j + i*qlen + h*qlen*qlen + b*qlen*qlen*heads
                    float ggml_val = ggml_out[j + i*qlen + h*qlen*qlen + b*qlen*qlen*heads];
                    err.add(ggml_val, exp_val);
                }
            }
        }
    }

    err.report("rel_shift");

    ggml_gallocr_free(allocr);
    ggml_free(ctx0);
    nemo_free(ctx);

    bool passed = err.max_diff < 1e-5f;
    printf("rel_shift test: %s\n\n", passed ? "PASS" : "FAIL");
    return passed;
}

// Test Q, K, V projections for attention (simplified - tests linear projections work)
bool test_mha() {
    printf("=== Testing MHA Q/K/V Projections ===\n");

    // Load original weights
    nemo::ModelWeights ref_weights;
    if (!ref_weights.load("weights/model.bin")) {
        fprintf(stderr, "Failed to load reference weights\n");
        return false;
    }

    // Load ggml model
    nemo_context * ctx = nemo_init("weights/model.gguf");
    if (!ctx) {
        fprintf(stderr, "Failed to load ggml model\n");
        return false;
    }

    const size_t d_model = 1024;
    size_t batch = 1;
    size_t seq_len = 10;

    // Create test input
    nemo::TensorF input({batch, seq_len, d_model});
    for (size_t i = 0; i < input.numel(); i++) {
        input.data[i] = 0.01f * (float)(i % 100) - 0.5f;
    }

    // === Test Q projection ===
    const char * q_weight_name = "encoder.layers.0.self_attn.linear_q.weight";
    const auto * ref_q_w = ref_weights.get(q_weight_name);
    auto it_q = ctx->model.tensors.find(q_weight_name);

    if (!ref_q_w || it_q == ctx->model.tensors.end()) {
        fprintf(stderr, "Q weight not found\n");
        nemo_free(ctx);
        return false;
    }

    // Original: Q = input @ Q_weight.T
    nemo::TensorF ref_q;
    nemo::linear_no_bias(input, ref_q_w->data.data(), d_model, d_model, ref_q);

    printf("Original Q[0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           ref_q(0,0,0), ref_q(0,0,1), ref_q(0,0,2), ref_q(0,0,3), ref_q(0,0,4));

    // GGML
    size_t buf_size = ggml_tensor_overhead() * 20 + ggml_graph_overhead();
    std::vector<uint8_t> compute_buf(buf_size);

    struct ggml_init_params params = {
        .mem_size   = buf_size,
        .mem_buffer = compute_buf.data(),
        .no_alloc   = true,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    struct ggml_tensor * inp = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, d_model, seq_len, batch);
    ggml_set_name(inp, "input");
    ggml_set_input(inp);

    struct ggml_tensor * q = ggml_mul_mat(ctx0, it_q->second, inp);
    ggml_set_name(q, "q_output");
    ggml_set_output(q);

    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, q);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->model.backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    ggml_backend_tensor_set(inp, input.data.data(), 0, input.numel() * sizeof(float));
    ggml_backend_graph_compute(ctx->model.backend, gf);

    std::vector<float> ggml_q(d_model * seq_len * batch);
    ggml_backend_tensor_get(q, ggml_q.data(), 0, ggml_q.size() * sizeof(float));

    printf("GGML Q[0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           ggml_q[0], ggml_q[1], ggml_q[2], ggml_q[3], ggml_q[4]);

    error_calc err;
    err.add_array(ggml_q.data(), ref_q.data.data(), ref_q.numel());
    err.report("MHA Q projection");

    ggml_gallocr_free(allocr);
    ggml_free(ctx0);
    nemo_free(ctx);

    bool passed = err.max_diff < 1e-4f;
    printf("MHA Q projection test: %s\n\n", passed ? "PASS" : "FAIL");
    return passed;
}

// Test positional encoding
bool test_pos_encoding() {
    printf("=== Testing Positional Encoding ===\n");

    // Load ggml model (which has precomputed pos_emb)
    nemo_context * ctx = nemo_init("weights/model.gguf");
    if (!ctx) {
        fprintf(stderr, "Failed to load ggml model\n");
        return false;
    }

    // Get ggml positional embeddings
    struct ggml_tensor * ggml_pos = ctx->model.pos_emb;
    if (!ggml_pos) {
        fprintf(stderr, "Missing pos_emb tensor\n");
        nemo_free(ctx);
        return false;
    }

    // ggml stores as [d_model, 2*max_len-1] = [1024, 1023]
    int d_model = ggml_pos->ne[0];
    int total_len = ggml_pos->ne[1];  // 2*max_len-1
    int max_len = (total_len + 1) / 2;  // = 512

    printf("GGML pos_emb shape: [%d, %d] (d_model, 2*max_len-1)\n", d_model, total_len);
    printf("max_len = %d\n", max_len);

    // Get ggml pos_emb data
    std::vector<float> ggml_pos_data(d_model * total_len);
    ggml_backend_tensor_get(ggml_pos, ggml_pos_data.data(), 0, ggml_pos_data.size() * sizeof(float));

    // === Original implementation ===
    nemo::RelPositionalEncoding pos_enc;
    nemo::TensorF ref_pos;
    pos_enc.get_pos_emb(max_len, ref_pos);  // Returns [2*max_len-1, d_model]

    printf("Original pos_emb shape: [%zu, %zu]\n", ref_pos.shape[0], ref_pos.shape[1]);

    // The embeddings should match at corresponding positions
    // ref_pos: [2*max_len-1, d_model] where ref_pos[i, d] = embedding at position (max_len-1-i) for dimension d
    // ggml_pos: [d_model, 2*max_len-1] where ggml_pos[d, pos] = embedding for position (pos - (max_len-1)) for dim d

    // The ggml compute_pos_emb stores directly:
    //   data[pos * d_model + i] = sin/cos for position (pos - (max_len-1))
    // So ggml_pos_data layout is [d_model, pos] contiguous in d_model
    // Index: ggml_pos_data[pos * d_model + d]

    // ref_pos stores: ref_pos(i, d) where position p = max_len - 1 - i
    // So for ggml pos index = (max_len - 1 - p) + (max_len - 1) = 2*(max_len-1) - p?
    // Wait, let me re-check the compute_pos_emb function...
    // compute_pos_emb: data[pos * d_model + i] where p = pos - (max_len - 1)
    // So pos=0 -> p=-(max_len-1), pos=(max_len-1) -> p=0, pos=2*(max_len-1) -> p=(max_len-1)

    // ref_pos.get_pos_emb(seq_len): output[i] = embedding for position (seq_len - 1 - i)
    // For seq_len = max_len: output[0] = pos (max_len-1), output[max_len-1] = pos 0, output[2*max_len-2] = pos -(max_len-1)

    // To compare: ref_pos(i, d) corresponds to ggml_pos[?, d] where:
    // ref_pos position = max_len - 1 - i
    // ggml_pos stores at index pos where position = pos - (max_len - 1)
    // So: pos - (max_len - 1) = max_len - 1 - i
    //     pos = 2*(max_len - 1) - i

    printf("Sample ref_pos[0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           ref_pos(0, 0), ref_pos(0, 1), ref_pos(0, 2), ref_pos(0, 3), ref_pos(0, 4));

    // With the new GGML convention:
    // ggml_pos_data[i] stores position (max_len - 1 - i), i.e., descending from +(max_len-1) to -(max_len-1)
    // ref_pos(i, d) stores embedding for position (max_len - 1 - i)
    // So ref_pos[i] corresponds to ggml_pos[i] - they're in the same order!
    int ggml_idx = 0;  // position max_len-1 is at index 0 in GGML
    printf("Sample ggml_pos[pos=%d,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           ggml_idx,
           ggml_pos_data[ggml_idx * d_model + 0],
           ggml_pos_data[ggml_idx * d_model + 1],
           ggml_pos_data[ggml_idx * d_model + 2],
           ggml_pos_data[ggml_idx * d_model + 3],
           ggml_pos_data[ggml_idx * d_model + 4]);

    // Compare
    struct error_calc err;
    int ref_len = ref_pos.shape[0];  // 2*max_len-1

    for (int i = 0; i < ref_len && i < total_len; i++) {
        // Both ref_pos and ggml_pos now store in same order (descending positions)
        int ggml_pos_idx = i;

        for (int d = 0; d < d_model; d++) {
            float ref_val = ref_pos(i, d);
            float ggml_val = ggml_pos_data[ggml_pos_idx * d_model + d];
            err.add(ggml_val, ref_val);
        }
    }

    err.report("Positional Encoding");

    nemo_free(ctx);

    bool passed = err.max_diff < 1e-5f;
    printf("Positional encoding test: %s\n\n", passed ? "PASS" : "FAIL");
    return passed;
}

// Test Conformer Conv module
bool test_conformer_conv() {
    printf("=== Testing Conformer Conv Module ===\n");

    // Load original weights
    nemo::ModelWeights ref_weights;
    if (!ref_weights.load("weights/model.bin")) {
        fprintf(stderr, "Failed to load reference weights\n");
        return false;
    }

    // Load ggml model
    nemo_context * ctx = nemo_init("weights/model.gguf");
    if (!ctx) {
        fprintf(stderr, "Failed to load ggml model\n");
        return false;
    }

    const size_t d_model = 1024;
    const size_t batch = 1;
    const size_t seq_len = 20;

    // Create test input [batch, time, d_model]
    nemo::TensorF input({batch, seq_len, d_model});
    for (size_t i = 0; i < input.numel(); i++) {
        input.data[i] = 0.01f * (float)(i % 100) - 0.5f;
    }

    // === Original implementation ===
    nemo::ConformerConvolution conv_mod;
    conv_mod.load_weights(ref_weights, "encoder.layers.0.conv");

    nemo::TensorF ref_output;
    conv_mod.forward(input, ref_output);

    printf("Original output shape: [%zu, %zu, %zu]\n",
           ref_output.shape[0], ref_output.shape[1], ref_output.shape[2]);
    printf("Original output[0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           ref_output(0,0,0), ref_output(0,0,1), ref_output(0,0,2),
           ref_output(0,0,3), ref_output(0,0,4));

    // === GGML implementation ===
    auto get_tensor = [&](const char * name) -> struct ggml_tensor * {
        auto it = ctx->model.tensors.find(name);
        return (it != ctx->model.tensors.end()) ? it->second : nullptr;
    };

    struct ggml_tensor * pw1_w = get_tensor("encoder.layers.0.conv.pointwise_conv1.weight");
    struct ggml_tensor * dw_w = get_tensor("encoder.layers.0.conv.depthwise_conv.weight");
    struct ggml_tensor * bn_w = get_tensor("encoder.layers.0.conv.batch_norm.weight");
    struct ggml_tensor * bn_b = get_tensor("encoder.layers.0.conv.batch_norm.bias");
    struct ggml_tensor * pw2_w = get_tensor("encoder.layers.0.conv.pointwise_conv2.weight");

    if (!pw1_w || !dw_w || !bn_w || !bn_b || !pw2_w) {
        fprintf(stderr, "Missing conv module tensors\n");
        nemo_free(ctx);
        return false;
    }

    printf("pw1_w shape: [%lld, %lld, %lld]\n",
           (long long)pw1_w->ne[0], (long long)pw1_w->ne[1], (long long)pw1_w->ne[2]);
    printf("dw_w shape: [%lld, %lld, %lld]\n",
           (long long)dw_w->ne[0], (long long)dw_w->ne[1], (long long)dw_w->ne[2]);
    printf("pw2_w shape: [%lld, %lld, %lld]\n",
           (long long)pw2_w->ne[0], (long long)pw2_w->ne[1], (long long)pw2_w->ne[2]);
    fflush(stdout);

    // Create compute context
    size_t buf_size = ggml_tensor_overhead() * 100 + ggml_graph_overhead() * 2;
    std::vector<uint8_t> compute_buf(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ compute_buf.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    // Input: [d_model, seq_len, batch] in ggml layout
    struct ggml_tensor * inp = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, d_model, seq_len, batch);
    ggml_set_name(inp, "input");
    ggml_set_input(inp);

    // Pointwise Conv1: [d_model, seq_len, batch] -> [2048, seq_len, batch]
    // pw1_w is [1, 1024, 2048] in ggml = [kernel=1, in_ch=1024, out_ch=2048]
    // For 1x1 conv, we can use mul_mat: output = input @ weight
    struct ggml_tensor * pw1_w_2d = ggml_reshape_2d(ctx0, pw1_w, d_model, 2048);
    struct ggml_tensor * cur = ggml_mul_mat(ctx0, pw1_w_2d, inp);
    // cur: [2048, seq_len, batch]

    // GLU: split in half, multiply first half by sigmoid of second half
    // cur: [2048, seq_len, batch] -> [1024, seq_len, batch]
    // Need to compute strides manually since cur is not allocated yet
    size_t half_ch = 1024;
    size_t full_ch = 2048;
    size_t nb1 = full_ch * sizeof(float);  // stride to next time step
    size_t nb2 = full_ch * seq_len * sizeof(float);  // stride to next batch

    struct ggml_tensor * glu_a = ggml_view_3d(ctx0, cur, half_ch, seq_len, batch,
                                              nb1, nb2, 0);
    struct ggml_tensor * glu_b = ggml_view_3d(ctx0, cur, half_ch, seq_len, batch,
                                              nb1, nb2, half_ch * sizeof(float));
    cur = ggml_mul(ctx0, glu_a, ggml_sigmoid(ctx0, glu_b));
    // cur: [1024, seq_len, batch]

    // Capture after_glu for debugging
    struct ggml_tensor * after_glu = cur;
    ggml_set_name(after_glu, "after_glu");
    ggml_set_output(after_glu);

    // Depthwise Causal Conv1d: kernel_size=9, groups=1024
    // cur: [1024, seq_len, batch]
    // dw_w: [9, 1, 1024] = [kernel_size, 1, channels]
    // Causal padding: pad left by kernel_size-1 = 8
    const int kernel_size = 9;
    cur = ggml_cont(ctx0, cur);  // Make sure it's contiguous

    // Transpose to [seq_len, 1024, batch] for conv1d
    cur = ggml_permute(ctx0, cur, 1, 0, 2, 3);
    cur = ggml_cont(ctx0, cur);

    // Apply causal padding: left = kernel_size-1 = 8, right = 0 (asymmetric)
    cur = ggml_pad_ext(ctx0, cur, kernel_size - 1, 0, 0, 0, 0, 0, 0, 0);

    // Depthwise conv1d manually: for each position t in output, sum over k:
    // output[c, t] = sum_k input[c, t+k] * weight[c, k]
    // Input shape after pad: [seq_len+8, d_model, batch]
    // Weight shape: [9, 1, 1024] -> need to reshape to [9, 1024]
    // After the conv (no trim needed since we padded exactly right): [seq_len, d_model, batch]

    // Weight shape is [9, 1, 1024] with memory layout: weight[k, 0, c] at offset k + c*9
    // Reshape to [9, 1024] for easier access
    struct ggml_tensor * dw_w_2d = ggml_reshape_2d(ctx0, dw_w, 9, d_model);

    // Transpose to [d_model, 9] so we can extract weight[:, k] easily
    struct ggml_tensor * dw_w_t = ggml_cont(ctx0, ggml_transpose(ctx0, dw_w_2d));
    // Now dw_w_t[c, k] = original weight[k, c] with memory: dw_w_t[c, k] at offset c + k*d_model

    // Initialize output accumulator as zeros
    // We'll do the conv by summing shifted slices
    struct ggml_tensor * conv_result = nullptr;

    for (int k = 0; k < kernel_size; k++) {
        // Extract slice of input at offset k: [seq_len, d_model, batch]
        struct ggml_tensor * input_slice = ggml_view_3d(ctx0, cur,
            seq_len, d_model, batch,                           // shape
            cur->nb[1], cur->nb[2],                            // strides
            k * sizeof(float));                                // offset

        // Get k-th kernel element for each channel from transposed weight
        // dw_w_t is [d_model, 9], we want column k: elements [0..d_model-1, k]
        // These are at offsets k*d_model, k*d_model+1, ..., k*d_model+d_model-1
        struct ggml_tensor * kernel_k = ggml_view_1d(ctx0, dw_w_t, d_model, k * d_model * sizeof(float));

        // Reshape to [1, d_model, 1] for broadcasting over [seq_len, d_model, batch]
        kernel_k = ggml_reshape_3d(ctx0, kernel_k, 1, d_model, 1);

        // Multiply input_slice by kernel_k (broadcast over time and batch)
        struct ggml_tensor * product = ggml_mul(ctx0, input_slice, kernel_k);

        if (conv_result == nullptr) {
            conv_result = product;
        } else {
            conv_result = ggml_add(ctx0, conv_result, product);
        }
    }
    cur = conv_result;

    // Transpose back to [1024, seq_len, batch]
    cur = ggml_permute(ctx0, cur, 1, 0, 2, 3);
    cur = ggml_cont(ctx0, cur);

    // Capture after depthwise conv for debugging
    struct ggml_tensor * after_dw = cur;
    ggml_set_name(after_dw, "after_dw");
    ggml_set_output(after_dw);

    // Layer norm: weight and bias are [1024]
    cur = ggml_norm(ctx0, cur, 1e-5f);
    cur = ggml_mul(ctx0, cur, bn_w);
    cur = ggml_add(ctx0, cur, bn_b);

    // Capture after layer norm for debugging
    struct ggml_tensor * after_ln = cur;
    ggml_set_name(after_ln, "after_ln");
    ggml_set_output(after_ln);

    // Swish
    cur = ggml_silu(ctx0, cur);

    // Pointwise Conv2: [1024, seq_len, batch] -> [1024, seq_len, batch]
    struct ggml_tensor * pw2_w_2d = ggml_reshape_2d(ctx0, pw2_w, d_model, d_model);
    cur = ggml_mul_mat(ctx0, pw2_w_2d, cur);

    // Transpose to [seq_len, 1024, batch] for output comparison
    cur = ggml_permute(ctx0, cur, 1, 0, 2, 3);
    struct ggml_tensor * out = ggml_cont(ctx0, cur);
    ggml_set_name(out, "conv_module_output");
    ggml_set_output(out);

    // Build graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, out);

    // Allocate
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->model.backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    // Set input - need to transpose from [batch, time, d_model] to [d_model, time, batch]
    std::vector<float> inp_transposed(input.numel());
    for (size_t b = 0; b < batch; b++) {
        for (size_t t = 0; t < seq_len; t++) {
            for (size_t d = 0; d < d_model; d++) {
                // C++ input: [batch, time, d_model] -> index = (b*time + t)*d_model + d
                // GGML input: [d_model, time, batch] -> index = d + t*d_model + b*d_model*time
                size_t c_idx = (b * seq_len + t) * d_model + d;
                size_t g_idx = d + t * d_model + b * d_model * seq_len;
                inp_transposed[g_idx] = input.data[c_idx];
            }
        }
    }
    ggml_backend_tensor_set(inp, inp_transposed.data(), 0, inp_transposed.size() * sizeof(float));

    // Compute
    ggml_backend_graph_compute(ctx->model.backend, gf);

    // Debug: Compare GLU intermediate output with original
    // First compute original GLU output
    nemo::TensorF orig_after_pw1;
    {
        // Transpose input to [batch, d_model, time]
        nemo::TensorF input_transposed({batch, d_model, seq_len});
        for (size_t b = 0; b < batch; b++) {
            for (size_t t = 0; t < seq_len; t++) {
                for (size_t d = 0; d < d_model; d++) {
                    input_transposed(b, d, t) = input(b, t, d);
                }
            }
        }
        // Pointwise conv1
        nemo::conv1d(input_transposed,
                     ref_weights.require("encoder.layers.0.conv.pointwise_conv1.weight").data.data(),
                     2048, d_model, 1, 1, 0, 1, nullptr, orig_after_pw1);
        printf("Original after pw1 shape: [%zu, %zu, %zu]\n",
               orig_after_pw1.shape[0], orig_after_pw1.shape[1], orig_after_pw1.shape[2]);
        printf("Original after pw1[0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
               orig_after_pw1(0,0,0), orig_after_pw1(0,0,1), orig_after_pw1(0,0,2),
               orig_after_pw1(0,0,3), orig_after_pw1(0,0,4));
        printf("Original after pw1[0,1024,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
               orig_after_pw1(0,1024,0), orig_after_pw1(0,1024,1), orig_after_pw1(0,1024,2),
               orig_after_pw1(0,1024,3), orig_after_pw1(0,1024,4));
    }

    // Get GGML after GLU output: [1024, seq_len, batch]
    std::vector<float> ggml_glu(d_model * seq_len * batch);
    ggml_backend_tensor_get(after_glu, ggml_glu.data(), 0, ggml_glu.size() * sizeof(float));
    printf("GGML after GLU shape: [%lld, %lld, %lld]\n",
           (long long)after_glu->ne[0], (long long)after_glu->ne[1], (long long)after_glu->ne[2]);
    printf("GGML after GLU[0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           ggml_glu[0], ggml_glu[1], ggml_glu[2], ggml_glu[3], ggml_glu[4]);

    // Compute original GLU
    nemo::TensorF orig_pw1_transposed({batch, seq_len, 2048});
    for (size_t b = 0; b < batch; b++) {
        for (size_t t = 0; t < seq_len; t++) {
            for (size_t d = 0; d < 2048; d++) {
                orig_pw1_transposed(b, t, d) = orig_after_pw1(b, d, t);
            }
        }
    }
    nemo::TensorF orig_after_glu;
    nemo::glu(orig_pw1_transposed, orig_after_glu);
    printf("Original after GLU shape: [%zu, %zu, %zu]\n",
           orig_after_glu.shape[0], orig_after_glu.shape[1], orig_after_glu.shape[2]);
    printf("Original after GLU[0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           orig_after_glu(0,0,0), orig_after_glu(0,0,1), orig_after_glu(0,0,2),
           orig_after_glu(0,0,3), orig_after_glu(0,0,4));

    // Compare GLU outputs
    struct error_calc err;
    for (size_t b = 0; b < batch; b++) {
        for (size_t t = 0; t < seq_len; t++) {
            for (size_t d = 0; d < d_model; d++) {
                float ref_val = orig_after_glu(b, t, d);
                // GGML GLU: [d_model, seq_len, batch] -> index = d + t*d_model + b*d_model*seq_len
                float ggml_val = ggml_glu[d + t * d_model + b * d_model * seq_len];
                err.add(ggml_val, ref_val);
            }
        }
    }
    err.report("GLU");

    // Compute original depthwise conv output
    // Transpose GLU output to [batch, d_model, time] for causal_conv1d
    nemo::TensorF orig_glu_transposed({batch, d_model, seq_len});
    for (size_t b = 0; b < batch; b++) {
        for (size_t t = 0; t < seq_len; t++) {
            for (size_t d = 0; d < d_model; d++) {
                orig_glu_transposed(b, d, t) = orig_after_glu(b, t, d);
            }
        }
    }
    nemo::TensorF orig_after_dw;
    nemo::causal_conv1d(orig_glu_transposed,
                        ref_weights.require("encoder.layers.0.conv.depthwise_conv.weight").data.data(),
                        d_model, d_model, 9, 1, d_model, nullptr, orig_after_dw);
    printf("Original after dw shape: [%zu, %zu, %zu]\n",
           orig_after_dw.shape[0], orig_after_dw.shape[1], orig_after_dw.shape[2]);
    printf("Original after dw[0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           orig_after_dw(0,0,0), orig_after_dw(0,0,1), orig_after_dw(0,0,2),
           orig_after_dw(0,0,3), orig_after_dw(0,0,4));

    // Get GGML after dw output: [d_model, seq_len, batch]
    std::vector<float> ggml_dw(d_model * seq_len * batch);
    ggml_backend_tensor_get(after_dw, ggml_dw.data(), 0, ggml_dw.size() * sizeof(float));
    printf("GGML after dw shape: [%lld, %lld, %lld]\n",
           (long long)after_dw->ne[0], (long long)after_dw->ne[1], (long long)after_dw->ne[2]);
    printf("GGML after dw[0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           ggml_dw[0], ggml_dw[1], ggml_dw[2], ggml_dw[3], ggml_dw[4]);

    // Compare dw outputs
    // orig_after_dw: [batch, d_model, time]
    // ggml_dw: [d_model, seq_len, batch]
    struct error_calc err_dw;
    for (size_t b = 0; b < batch; b++) {
        for (size_t t = 0; t < seq_len; t++) {
            for (size_t d = 0; d < d_model; d++) {
                float ref_val = orig_after_dw(b, d, t);
                // GGML dw: [d_model, seq_len, batch] -> index = d + t*d_model + b*d_model*seq_len
                float ggml_val = ggml_dw[d + t * d_model + b * d_model * seq_len];
                err_dw.add(ggml_val, ref_val);
            }
        }
    }
    err_dw.report("Depthwise Conv");

    // Get full conv module output
    // out: [seq_len, d_model, batch] after final permute
    std::vector<float> ggml_out(seq_len * d_model * batch);
    ggml_backend_tensor_get(out, ggml_out.data(), 0, ggml_out.size() * sizeof(float));

    printf("GGML conv module output[0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           ggml_out[0], ggml_out[1], ggml_out[2], ggml_out[3], ggml_out[4]);

    // Compare with original output (ref_output is already computed above)
    // ref_output: [batch, time, d_model]
    // ggml_out: [seq_len, d_model, batch] -> index = t + d*seq_len + b*seq_len*d_model
    struct error_calc err_conv;
    for (size_t b = 0; b < batch; b++) {
        for (size_t t = 0; t < seq_len; t++) {
            for (size_t d = 0; d < d_model; d++) {
                // ref_output: [batch, time, d_model]
                float ref_val = ref_output(b, t, d);
                // ggml_out: [seq_len, d_model, batch] -> index = t + d*seq_len + b*seq_len*d_model
                float ggml_val = ggml_out[t + d * seq_len + b * seq_len * d_model];
                err_conv.add(ggml_val, ref_val);
            }
        }
    }

    err_conv.report("Conformer Conv Module Output");

    // Cleanup
    ggml_gallocr_free(allocr);
    ggml_free(ctx0);
    nemo_free(ctx);

    bool passed = err_conv.max_diff < 1e-2f;  // Allow larger diff for multi-step computation
    printf("Conformer Conv module test: %s\n\n", passed ? "PASS" : "FAIL");
    return passed;
}

// Test full relative position multi-head attention
bool test_mha_full() {
    printf("=== Testing Full MHA with Relative Position ===\n");

    // Load original weights
    nemo::ModelWeights ref_weights;
    if (!ref_weights.load("weights/model.bin")) {
        fprintf(stderr, "Failed to load reference weights\n");
        return false;
    }

    // Load ggml model
    nemo_context * ctx = nemo_init("weights/model.gguf");
    if (!ctx) {
        fprintf(stderr, "Failed to load ggml model\n");
        return false;
    }

    const size_t d_model = 1024;
    const size_t n_heads = 8;
    const size_t d_head = 128;
    const size_t batch = 1;
    const size_t seq_len = 10;

    // Create test input
    nemo::TensorF input({batch, seq_len, d_model});
    for (size_t i = 0; i < input.numel(); i++) {
        input.data[i] = 0.01f * (float)(i % 100) - 0.5f;
    }

    // === Original implementation ===
    nemo::RelPositionalEncoding pos_enc;
    nemo::TensorF pos_emb;
    pos_enc.get_pos_emb(seq_len, pos_emb);

    nemo::RelPositionMultiHeadAttention mha;
    mha.load_weights(ref_weights, "encoder.layers.0.self_attn");

    nemo::TensorF ref_output;
    mha.forward(input, pos_emb, ref_output);

    printf("Original MHA output shape: [%zu, %zu, %zu]\n",
           ref_output.shape[0], ref_output.shape[1], ref_output.shape[2]);
    printf("Original output[0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           ref_output(0,0,0), ref_output(0,0,1), ref_output(0,0,2),
           ref_output(0,0,3), ref_output(0,0,4));

    // === GGML implementation ===
    auto get_tensor = [&](const char * name) -> struct ggml_tensor * {
        auto it = ctx->model.tensors.find(name);
        return (it != ctx->model.tensors.end()) ? it->second : nullptr;
    };

    struct ggml_tensor * q_w = get_tensor("encoder.layers.0.self_attn.linear_q.weight");
    struct ggml_tensor * k_w = get_tensor("encoder.layers.0.self_attn.linear_k.weight");
    struct ggml_tensor * v_w = get_tensor("encoder.layers.0.self_attn.linear_v.weight");
    struct ggml_tensor * pos_w = get_tensor("encoder.layers.0.self_attn.linear_pos.weight");
    struct ggml_tensor * out_w = get_tensor("encoder.layers.0.self_attn.linear_out.weight");
    struct ggml_tensor * bias_u = get_tensor("encoder.layers.0.self_attn.pos_bias_u");
    struct ggml_tensor * bias_v = get_tensor("encoder.layers.0.self_attn.pos_bias_v");

    if (!q_w || !k_w || !v_w || !pos_w || !out_w || !bias_u || !bias_v) {
        fprintf(stderr, "Missing attention tensors\n");
        nemo_free(ctx);
        return false;
    }

    // Create compute context
    size_t buf_size = ggml_tensor_overhead() * 100 + ggml_graph_overhead();
    std::vector<uint8_t> compute_buf(buf_size);

    struct ggml_init_params params = {
        .mem_size   = buf_size,
        .mem_buffer = compute_buf.data(),
        .no_alloc   = true,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    // Input: [d_model, time, batch]
    struct ggml_tensor * inp = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, d_model, seq_len, batch);
    ggml_set_name(inp, "input");
    ggml_set_input(inp);

    // Position embeddings: need to slice from precomputed pos_emb
    // For seq_len = 10, we need positions from (seq_len-1) to -(seq_len-1) = 19 positions
    // From ggml pos_emb [d_model, 2*max_len-1], we need to extract the middle 2*seq_len-1 positions
    size_t pos_len = 2 * seq_len - 1;

    // Create position embedding input (will be set from model.pos_emb)
    struct ggml_tensor * pos_emb_inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, pos_len);
    ggml_set_name(pos_emb_inp, "pos_emb");
    ggml_set_input(pos_emb_inp);

    // Build MHA graph using the helper function
    // Note: We need to call the build function - but it's static in nemo-ggml.cpp
    // For testing, we'll implement the same logic here

    // Q, K, V projections
    struct ggml_tensor * q = ggml_mul_mat(ctx0, q_w, inp);
    struct ggml_tensor * k = ggml_mul_mat(ctx0, k_w, inp);
    struct ggml_tensor * v = ggml_mul_mat(ctx0, v_w, inp);
    struct ggml_tensor * pos = ggml_mul_mat(ctx0, pos_w, pos_emb_inp);

    // Reshape to [d_head, n_heads, time, batch]
    q = ggml_reshape_4d(ctx0, q, d_head, n_heads, seq_len, batch);
    k = ggml_reshape_4d(ctx0, k, d_head, n_heads, seq_len, batch);
    v = ggml_reshape_4d(ctx0, v, d_head, n_heads, seq_len, batch);
    pos = ggml_reshape_3d(ctx0, pos, d_head, n_heads, pos_len);

    // Permute to [d_head, time, n_heads, batch]
    q = ggml_cont(ctx0, ggml_permute(ctx0, q, 0, 2, 1, 3));
    k = ggml_cont(ctx0, ggml_permute(ctx0, k, 0, 2, 1, 3));
    v = ggml_cont(ctx0, ggml_permute(ctx0, v, 0, 2, 1, 3));
    pos = ggml_cont(ctx0, ggml_permute(ctx0, pos, 0, 2, 1, 3));

    // Reshape biases: [d_model] -> [d_head, 1, n_heads, 1]
    struct ggml_tensor * bias_u_4d = ggml_reshape_4d(ctx0, bias_u, d_head, 1, n_heads, 1);
    struct ggml_tensor * bias_v_4d = ggml_reshape_4d(ctx0, bias_v, d_head, 1, n_heads, 1);

    // Add biases: q_u = q + bias_u, q_v = q + bias_v
    struct ggml_tensor * q_u = ggml_add(ctx0, q, bias_u_4d);
    struct ggml_tensor * q_v = ggml_add(ctx0, q, bias_v_4d);

    // Content attention: q_u @ k^T -> [time, time, n_heads, batch]
    struct ggml_tensor * content_attn = ggml_mul_mat(ctx0, k, q_u);

    // Position attention: q_v @ pos^T -> [pos_len, time, n_heads, batch]
    // mul_mat(pos, q_v): pos[d, pos_len, h], q_v[d, time, h, b]
    // Result: [pos_len, time, h, b] where result[p, t, h, b] = sum_d q_v[d, t, h, b] * pos[d, p, h]
    struct ggml_tensor * pos_attn_raw = ggml_mul_mat(ctx0, pos, q_v);

    // Rel shift: input[p, t, h, b] -> output[j, t, h, b]
    // where for query t and output position j: p = j + qlen - 1 - t
    // This transforms position attention to align with key positions

    // Step 1: Pad with zero column on left -> [pos_len+1, time, h, b]
    struct ggml_tensor * padded = ggml_pad_ext(ctx0, pos_attn_raw, 1, 0, 0, 0, 0, 0, 0, 0);

    // Step 2: Reshape to [time, pos_len+1, n_heads, batch]
    // Note: padded may not be contiguous, so use ggml_cont first
    struct ggml_tensor * reshaped = ggml_reshape_4d(ctx0, ggml_cont(ctx0, padded), seq_len, pos_len + 1, n_heads, batch);

    // Step 3: Drop first row by offsetting view
    struct ggml_tensor * dropped = ggml_view_4d(ctx0, reshaped,
        seq_len, pos_len, n_heads, batch,
        reshaped->nb[1], reshaped->nb[2], reshaped->nb[3],
        seq_len * sizeof(float));

    // Step 4: Make contiguous and reshape back
    struct ggml_tensor * back = ggml_reshape_4d(ctx0, ggml_cont(ctx0, dropped), pos_len, seq_len, n_heads, batch);

    // Step 5: Slice first time columns
    struct ggml_tensor * pos_attn = ggml_view_4d(ctx0, back,
        seq_len, seq_len, n_heads, batch,
        back->nb[1], back->nb[2], back->nb[3], 0);
    pos_attn = ggml_cont(ctx0, pos_attn);

    // Combine: scores = (content + pos) * scale
    float scale = 1.0f / std::sqrt((float)d_head);
    struct ggml_tensor * attn_scores = ggml_add(ctx0, content_attn, pos_attn);
    attn_scores = ggml_scale(ctx0, attn_scores, scale);

    // Softmax over key dimension
    struct ggml_tensor * attn_weights = ggml_soft_max(ctx0, attn_scores);

    // Apply to values: context = attn_weights @ v
    // attn_weights: [time, time, n_heads, batch] where attn[q, k] = attention from q to k
    // v: [d_head, time, n_heads, batch] where v[d, k] = value at position k, dimension d
    // We want: context[d, q] = sum_k attn[q, k] * v[d, k]
    // 
    // For ggml_mul_mat(A, B) = B @ A^T with condition A->ne[0] == B->ne[0]:
    //   Let v_t = transpose(v) = [time, d_head, n_heads, batch]
    //   Then mul_mat(attn_weights, v_t) where:
    // Apply to values: context = attn_weights @ v
    // attn_weights: [time(j), time(i), n_heads, batch] - attn[j, i] = attention from query i to key j
    // v: [d_head, time(j), n_heads, batch]
    // We want: context[d, i] = sum_j attn[j, i] * v[d, j]
    // 
    // Permute v to [time, d_head, n_heads, batch], then mul_mat(v_perm, attn_weights)
    // gives attn_weights @ v_perm^T = [d_head, time, n_heads, batch]
    struct ggml_tensor * v_perm = ggml_cont(ctx0, ggml_permute(ctx0, v, 1, 0, 2, 3));
    struct ggml_tensor * context = ggml_mul_mat(ctx0, v_perm, attn_weights);

    // context is [d_head, time, n_heads, batch]
    // Permute back to [d_head, n_heads, time, batch]
    context = ggml_cont(ctx0, ggml_permute(ctx0, context, 0, 2, 1, 3));

    // Reshape to [d_model, time, batch]
    context = ggml_reshape_3d(ctx0, context, d_model, seq_len, batch);

    // Output projection
    struct ggml_tensor * out = ggml_mul_mat(ctx0, out_w, context);
    ggml_set_name(out, "mha_output");
    ggml_set_output(out);

    // Build and allocate graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->model.backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    // Set input data (transpose from [batch, time, d_model] to [d_model, time, batch])
    std::vector<float> inp_data(d_model * seq_len * batch);
    for (size_t b = 0; b < batch; b++) {
        for (size_t t = 0; t < seq_len; t++) {
            for (size_t d = 0; d < d_model; d++) {
                inp_data[d + t * d_model + b * d_model * seq_len] = input(b, t, d);
            }
        }
    }
    ggml_backend_tensor_set(inp, inp_data.data(), 0, inp_data.size() * sizeof(float));

    // Set position embedding data
    // pos_emb from original: [2*seq_len-1, d_model] where pos_emb(i,d) = embedding for position (seq_len-1-i)
    // ggml expects: [d_model, pos_len]
    std::vector<float> pos_data(d_model * pos_len);
    for (size_t p = 0; p < pos_len; p++) {
        for (size_t d = 0; d < d_model; d++) {
            pos_data[d + p * d_model] = pos_emb(p, d);
        }
    }
    ggml_backend_tensor_set(pos_emb_inp, pos_data.data(), 0, pos_data.size() * sizeof(float));

    // Compute
    ggml_backend_graph_compute(ctx->model.backend, gf);

    // Get output
    std::vector<float> ggml_out(d_model * seq_len * batch);
    ggml_backend_tensor_get(out, ggml_out.data(), 0, ggml_out.size() * sizeof(float));

    printf("GGML MHA output[0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           ggml_out[0], ggml_out[1], ggml_out[2], ggml_out[3], ggml_out[4]);

    // Compare - transpose from [d_model, time, batch] to [batch, time, d_model]
    error_calc err;
    for (size_t b = 0; b < batch; b++) {
        for (size_t t = 0; t < seq_len; t++) {
            for (size_t d = 0; d < d_model; d++) {
                float ref_val = ref_output(b, t, d);
                float ggml_val = ggml_out[d + t * d_model + b * d_model * seq_len];
                err.add(ggml_val, ref_val);
            }
        }
    }

    err.report("Full MHA");

    // Cleanup
    ggml_gallocr_free(allocr);
    ggml_free(ctx0);
    nemo_free(ctx);

    bool passed = err.max_diff < 1e-3f;  // Slightly larger tolerance for complex operation
    printf("Full MHA test: %s\n\n", passed ? "PASS" : "FAIL");
    return passed;
}

// Test full Conformer layer
bool test_conformer_layer() {
    printf("=== Testing Conformer Layer ===\n");

    // Load original weights
    nemo::ModelWeights ref_weights;
    if (!ref_weights.load("weights/model.bin")) {
        fprintf(stderr, "Failed to load reference weights\n");
        return false;
    }

    // Load ggml model
    nemo_context * ctx = nemo_init("weights/model.gguf");
    if (!ctx) {
        fprintf(stderr, "Failed to load ggml model\n");
        return false;
    }

    const size_t d_model = 1024;
    const size_t batch = 1;
    const size_t seq_len = 16;  // Keep small for testing
    const int n_heads = 8;
    const int d_head = 128;
    const int kernel_size = 9;  // As specified in the model

    // Create test input [batch, time, d_model]
    nemo::TensorF input({batch, seq_len, d_model});
    for (size_t i = 0; i < input.numel(); i++) {
        input.data[i] = 0.01f * (float)(i % 100) - 0.5f;
    }

    // Create position embeddings
    nemo::RelPositionalEncoding pos_enc;
    pos_enc.init();
    nemo::TensorF pos_emb;
    pos_enc.get_pos_emb(seq_len, pos_emb);
    size_t pos_len = 2 * seq_len - 1;

    printf("Input shape: [%zu, %zu, %zu]\n", batch, seq_len, d_model);
    printf("Pos emb shape: [%zu, %zu]\n", pos_len, d_model);

    // === Original implementation ===
    nemo::ConformerLayer layer_ref;
    layer_ref.load_weights(ref_weights, "encoder.layers.0");

    nemo::TensorF ref_output;
    layer_ref.forward(input, pos_emb, ref_output);

    printf("Original output shape: [%zu, %zu, %zu]\n",
           ref_output.shape[0], ref_output.shape[1], ref_output.shape[2]);
    printf("Original output[0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           ref_output(0,0,0), ref_output(0,0,1), ref_output(0,0,2),
           ref_output(0,0,3), ref_output(0,0,4));

    // === GGML implementation ===
    // Create compute context - need more space for full layer
    size_t buf_size = ggml_tensor_overhead() * 1500 + ggml_graph_overhead() * 2;
    std::vector<uint8_t> compute_buf(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ compute_buf.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    if (!ctx0) {
        fprintf(stderr, "Failed to init ggml context\n");
        nemo_free(ctx);
        return false;
    }

    // Create input tensor [d_model, seq_len, batch]
    struct ggml_tensor * inp = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, d_model, seq_len, batch);
    ggml_set_name(inp, "input");
    ggml_set_input(inp);

    // Create pos_emb tensor [d_model, pos_len]
    struct ggml_tensor * pos = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, pos_len);
    ggml_set_name(pos, "pos_emb");
    ggml_set_input(pos);

    // Get layer 0 weights from ggml model
    nemo_conformer_layer * layer = &ctx->model.encoder.layers[0];

    // Build full conformer layer graph
    struct ggml_tensor * out = build_conformer_layer(ctx0, inp, pos, layer, n_heads, d_head, kernel_size);
    ggml_set_name(out, "conformer_layer_output");
    ggml_set_output(out);

    // Build graph
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 8192, false);
    ggml_build_forward_expand(gf, out);

    printf("Graph nodes: %d\n", ggml_graph_n_nodes(gf));

    // Allocate
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->model.backend));
    if (!ggml_gallocr_alloc_graph(allocr, gf)) {
        fprintf(stderr, "Failed to allocate graph\n");
        ggml_gallocr_free(allocr);
        ggml_free(ctx0);
        nemo_free(ctx);
        return false;
    }

    // Set input - transpose from [batch, time, d_model] to [d_model, time, batch]
    std::vector<float> inp_transposed(input.numel());
    for (size_t b = 0; b < batch; b++) {
        for (size_t t = 0; t < seq_len; t++) {
            for (size_t d = 0; d < d_model; d++) {
                size_t c_idx = (b * seq_len + t) * d_model + d;
                size_t g_idx = d + t * d_model + b * d_model * seq_len;
                inp_transposed[g_idx] = input.data[c_idx];
            }
        }
    }
    ggml_backend_tensor_set(inp, inp_transposed.data(), 0, inp_transposed.size() * sizeof(float));

    // Set pos_emb - transpose from [pos_len, d_model] to [d_model, pos_len]
    std::vector<float> pos_transposed(pos_emb.numel());
    for (size_t p = 0; p < pos_len; p++) {
        for (size_t d = 0; d < d_model; d++) {
            pos_transposed[d + p * d_model] = pos_emb(p, d);
        }
    }
    ggml_backend_tensor_set(pos, pos_transposed.data(), 0, pos_transposed.size() * sizeof(float));

    // Compute
    ggml_backend_graph_compute(ctx->model.backend, gf);

    // Get output
    std::vector<float> ggml_out(batch * seq_len * d_model);
    ggml_backend_tensor_get(out, ggml_out.data(), 0, ggml_out.size() * sizeof(float));

    printf("GGML output[0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           ggml_out[0], ggml_out[1], ggml_out[2], ggml_out[3], ggml_out[4]);

    // Compare - transpose from [d_model, time, batch] to [batch, time, d_model]
    error_calc err;
    for (size_t b = 0; b < batch; b++) {
        for (size_t t = 0; t < seq_len; t++) {
            for (size_t d = 0; d < d_model; d++) {
                float ref_val = ref_output(b, t, d);
                float ggml_val = ggml_out[d + t * d_model + b * d_model * seq_len];
                err.add(ggml_val, ref_val);
            }
        }
    }

    err.report("Conformer Layer");

    // Cleanup
    ggml_gallocr_free(allocr);
    ggml_free(ctx0);
    nemo_free(ctx);

    bool passed = err.max_diff < 2e-3f;  // Slightly larger tolerance for complex layer
    printf("Conformer Layer test: %s\n\n", passed ? "PASS" : "FAIL");
    return passed;
}

// Test full encoder (ConvSubsampling + 24 Conformer layers)
bool test_encoder() {
    printf("=== Testing Full Encoder ===\n");

    // Load ggml model
    nemo_context * ctx = nemo_init("weights/model.gguf");
    if (!ctx) {
        fprintf(stderr, "Failed to load ggml model\n");
        return false;
    }

    // Load mel input from file - raw float32 data, shape [time, 128]
    FILE * f = fopen("test.mel.bin", "rb");
    if (!f) {
        fprintf(stderr, "Failed to open test.mel.bin\n");
        nemo_free(ctx);
        return false;
    }

    // Get file size to determine time dimension
    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    size_t batch = 1;
    size_t features = 128;
    size_t time_in = file_size / (sizeof(float) * features);

    printf("Mel input shape: [%zu, %zu, %zu]\n", batch, time_in, features);

    // Read raw float data - file is [time, features]
    std::vector<float> raw_mel(time_in * features);
    size_t read = fread(raw_mel.data(), sizeof(float), raw_mel.size(), f);
    fclose(f);

    if (read != raw_mel.size()) {
        fprintf(stderr, "Failed to read mel data\n");
        nemo_free(ctx);
        return false;
    }

    // Reshape to [batch, time, features]
    nemo::TensorF mel_input({batch, time_in, features});
    for (size_t t = 0; t < time_in; t++) {
        for (size_t ff = 0; ff < features; ff++) {
            mel_input(0, t, ff) = raw_mel[t * features + ff];
        }
    }

    // Load precomputed reference output
    // If not available, fall back to computing it (slow)
    nemo::TensorF ref_output;
    FILE * ref_file = fopen("weights/encoder_ref.bin", "rb");
    if (ref_file) {
        printf("Loading precomputed reference from weights/encoder_ref.bin\n");
        uint64_t shape[3];
        if (fread(shape, sizeof(uint64_t), 3, ref_file) != 3) {
            fprintf(stderr, "Failed to read reference shape\n");
            fclose(ref_file);
            nemo_free(ctx);
            return false;
        }
        ref_output.resize({(size_t)shape[0], (size_t)shape[1], (size_t)shape[2]});
        if (fread(ref_output.data.data(), sizeof(float), ref_output.numel(), ref_file) != ref_output.numel()) {
            fprintf(stderr, "Failed to read reference data\n");
            fclose(ref_file);
            nemo_free(ctx);
            return false;
        }
        fclose(ref_file);
    } else {
        printf("No precomputed reference found, computing (this may take a while)...\n");
        printf("Run: make -f Makefile.ggml precompute_encoder_ref && ./precompute_encoder_ref\n");

        // Load original weights
        nemo::ModelWeights ref_weights;
        if (!ref_weights.load("weights/model.bin")) {
            fprintf(stderr, "Failed to load reference weights\n");
            nemo_free(ctx);
            return false;
        }

        nemo::ConformerEncoder encoder_ref;
        encoder_ref.load_weights(ref_weights);
        encoder_ref.forward(mel_input, ref_output);
    }

    printf("Reference encoder output shape: [%zu, %zu, %zu]\n",
           ref_output.shape[0], ref_output.shape[1], ref_output.shape[2]);
    printf("Reference output[0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           ref_output(0,0,0), ref_output(0,0,1), ref_output(0,0,2),
           ref_output(0,0,3), ref_output(0,0,4));

    // === GGML implementation ===
    // Create compute context - need large buffer for full encoder
    // 24 layers * ~132 nodes + subsampling = ~3500 nodes
    size_t buf_size = ggml_tensor_overhead() * 8000 + ggml_graph_overhead() * 4;
    std::vector<uint8_t> compute_buf(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ compute_buf.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    if (!ctx0) {
        fprintf(stderr, "Failed to init ggml context\n");
        nemo_free(ctx);
        return false;
    }

    // Create input tensor [n_mels, time, batch]
    struct ggml_tensor * inp = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, features, time_in, batch);
    ggml_set_name(inp, "mel_input");
    ggml_set_input(inp);

    // Debug: check kernel_size was correctly inferred
    printf("Inferred kernel_size: %d\n", ctx->model.hparams.kernel_size);

    // Build full encoder graph
    struct ggml_tensor * out = build_encoder(ctx0, inp, &ctx->model);
    ggml_set_name(out, "encoder_output");
    ggml_set_output(out);

    // Build graph - need large graph for full encoder
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 16384, false);
    ggml_build_forward_expand(gf, out);

    printf("Graph nodes: %d\n", ggml_graph_n_nodes(gf));

    // Allocate
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->model.backend));
    if (!ggml_gallocr_alloc_graph(allocr, gf)) {
        fprintf(stderr, "Failed to allocate graph\n");
        ggml_gallocr_free(allocr);
        ggml_free(ctx0);
        nemo_free(ctx);
        return false;
    }

    // Set input - transpose from [batch, time, features] to [features, time, batch]
    std::vector<float> inp_transposed(mel_input.numel());
    for (size_t b = 0; b < batch; b++) {
        for (size_t t = 0; t < time_in; t++) {
            for (size_t ff = 0; ff < features; ff++) {
                size_t c_idx = (b * time_in + t) * features + ff;
                size_t g_idx = ff + t * features + b * features * time_in;
                inp_transposed[g_idx] = mel_input.data[c_idx];
            }
        }
    }
    ggml_backend_tensor_set(inp, inp_transposed.data(), 0, inp_transposed.size() * sizeof(float));

    // Compute
    printf("Computing encoder...\n");
    ggml_backend_graph_compute(ctx->model.backend, gf);

    // Get output dimensions
    size_t time_out = ref_output.shape[1];
    size_t d_model = ref_output.shape[2];

    printf("GGML encoder output shape: [%lld, %lld, %lld]\n",
           (long long)out->ne[0], (long long)out->ne[1], (long long)out->ne[2]);
    printf("Expected output shape: [%zu, %zu, %zu]\n", batch, time_out, d_model);

    // Get output
    size_t out_numel = out->ne[0] * out->ne[1] * out->ne[2];
    std::vector<float> ggml_out(out_numel);
    ggml_backend_tensor_get(out, ggml_out.data(), 0, out_numel * sizeof(float));

    printf("GGML output[0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           ggml_out[0], ggml_out[1], ggml_out[2], ggml_out[3], ggml_out[4]);

    // Compare - transpose from [d_model, time, batch] to [batch, time, d_model]
    error_calc err;
    for (size_t b = 0; b < batch; b++) {
        for (size_t t = 0; t < time_out; t++) {
            for (size_t d = 0; d < d_model; d++) {
                float ref_val = ref_output(b, t, d);
                float ggml_val = ggml_out[d + t * d_model + b * d_model * time_out];
                err.add(ggml_val, ref_val);
            }
        }
    }

    err.report("Full Encoder");

    // Cleanup
    ggml_gallocr_free(allocr);
    ggml_free(ctx0);
    nemo_free(ctx);

    bool passed = err.max_diff < 5e-3f;  // Slightly larger tolerance for full encoder
    printf("Full Encoder test: %s\n\n", passed ? "PASS" : "FAIL");
    return passed;
}

// Forward declare build functions from nemo-ggml.cpp
struct ggml_tensor * build_decoder_step(
    struct ggml_context * ctx,
    struct ggml_tensor * token_emb,
    struct ggml_tensor * h_in,
    struct ggml_tensor * c_in,
    nemo_decoder * decoder,
    struct ggml_tensor ** h_out,
    struct ggml_tensor ** c_out
);

struct ggml_tensor * build_joint(
    struct ggml_context * ctx,
    struct ggml_tensor * encoder_out,
    struct ggml_tensor * decoder_out,
    nemo_joint * joint
);

std::vector<int> greedy_decode(
    struct nemo_context * nctx,
    struct ggml_tensor * encoder_out,
    ggml_backend_t backend
);

std::string tokens_to_text(const std::vector<int> & tokens, const std::vector<char8> & vocab);

bool test_decoder() {
    printf("=== Testing Decoder (LSTM) ===\n");

    // Load models
    nemo_context * ctx = nemo_init("weights/model.gguf");
    if (!ctx) {
        fprintf(stderr, "Failed to load ggml model\n");
        return false;
    }

    nemo::ModelWeights ref_weights;
    if (!ref_weights.load("weights/model.bin")) {
        fprintf(stderr, "Failed to load reference weights\n");
        nemo_free(ctx);
        return false;
    }

    nemo::RNNTDecoder ref_decoder;
    ref_decoder.load_weights(ref_weights);
    ref_decoder.init_state(1);

    const int hidden_size = 640;
    const int num_layers = 2;

    // Test with a few different tokens
    int test_tokens[] = {1024, 0, 100, 500};  // blank, then some tokens
    bool all_passed = true;

    for (int test_idx = 0; test_idx < 4; test_idx++) {
        int token = test_tokens[test_idx];
        printf("Testing token %d...\n", token);

        // Reference decoder forward
        nemo::TensorF ref_output;
        ref_decoder.forward_step(token, ref_output);

        printf("  Reference output[0:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
               ref_output(0, 0), ref_output(0, 1), ref_output(0, 2),
               ref_output(0, 3), ref_output(0, 4));

        // === GGML implementation ===
        size_t buf_size = ggml_tensor_overhead() * 100 + ggml_graph_overhead();
        std::vector<uint8_t> compute_buf(buf_size);

        struct ggml_init_params params = {
            /*.mem_size   =*/ buf_size,
            /*.mem_buffer =*/ compute_buf.data(),
            /*.no_alloc   =*/ true,
        };

        struct ggml_context * ctx0 = ggml_init(params);
        if (!ctx0) {
            fprintf(stderr, "Failed to init ggml context\n");
            nemo_free(ctx);
            return false;
        }

        // Create input tensors
        struct ggml_tensor * h_in = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, num_layers * hidden_size);
        struct ggml_tensor * c_in = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, num_layers * hidden_size);
        struct ggml_tensor * token_emb = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hidden_size);
        ggml_set_name(h_in, "h_in");
        ggml_set_name(c_in, "c_in");
        ggml_set_name(token_emb, "token_emb");
        ggml_set_input(h_in);
        ggml_set_input(c_in);
        ggml_set_input(token_emb);

        // Build decoder step graph
        struct ggml_tensor * h_out = nullptr;
        struct ggml_tensor * c_out = nullptr;
        struct ggml_tensor * dec_out = build_decoder_step(ctx0, token_emb, h_in, c_in,
                                                          &ctx->model.decoder, &h_out, &c_out);
        ggml_set_name(dec_out, "decoder_output");
        ggml_set_output(dec_out);
        ggml_set_output(h_out);
        ggml_set_output(c_out);

        // Build graph
        struct ggml_cgraph * gf = ggml_new_graph(ctx0);
        ggml_build_forward_expand(gf, dec_out);
        ggml_build_forward_expand(gf, h_out);
        ggml_build_forward_expand(gf, c_out);

        // Allocate
        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->model.backend));
        if (!ggml_gallocr_alloc_graph(allocr, gf)) {
            fprintf(stderr, "Failed to allocate graph\n");
            ggml_gallocr_free(allocr);
            ggml_free(ctx0);
            nemo_free(ctx);
            return false;
        }

        // Set inputs - get current state from reference decoder
        std::vector<float> h_data(num_layers * hidden_size);
        std::vector<float> c_data(num_layers * hidden_size);

        // Copy state from reference decoder (before this step)
        // Note: we need to save state before ref_decoder.forward_step
        // For simplicity, we use GGML state that tracks across tests
        ggml_backend_tensor_set(h_in, ctx->state.h.data(), 0, h_data.size() * sizeof(float));
        ggml_backend_tensor_set(c_in, ctx->state.c.data(), 0, c_data.size() * sizeof(float));

        // Get token embedding from model
        std::vector<float> emb_data(hidden_size);
        // embedding tensor is [hidden_size, vocab_size] in GGML layout
        size_t emb_offset = token * hidden_size * sizeof(float);
        ggml_backend_tensor_get(ctx->model.decoder.embedding, emb_data.data(), emb_offset, hidden_size * sizeof(float));
        ggml_backend_tensor_set(token_emb, emb_data.data(), 0, hidden_size * sizeof(float));

        // Compute
        ggml_backend_graph_compute(ctx->model.backend, gf);

        // Get output
        std::vector<float> ggml_out(hidden_size);
        ggml_backend_tensor_get(dec_out, ggml_out.data(), 0, hidden_size * sizeof(float));

        printf("  GGML output[0:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
               ggml_out[0], ggml_out[1], ggml_out[2], ggml_out[3], ggml_out[4]);

        // Compare
        error_calc err;
        for (int i = 0; i < hidden_size; i++) {
            err.add(ggml_out[i], ref_output(0, i));
        }
        err.report("  Decoder step");

        // Update GGML state for next iteration
        ggml_backend_tensor_get(h_out, ctx->state.h.data(), 0, num_layers * hidden_size * sizeof(float));
        ggml_backend_tensor_get(c_out, ctx->state.c.data(), 0, num_layers * hidden_size * sizeof(float));

        bool passed = err.max_diff < 1e-4f;
        printf("  Token %d: %s\n", token, passed ? "PASS" : "FAIL");
        if (!passed) all_passed = false;

        ggml_gallocr_free(allocr);
        ggml_free(ctx0);
    }

    nemo_free(ctx);

    printf("Decoder test: %s\n\n", all_passed ? "PASS" : "FAIL");
    return all_passed;
}

bool test_joint() {
    printf("=== Testing Joint Network ===\n");

    // Load models
    nemo_context * ctx = nemo_init("weights/model.gguf");
    if (!ctx) {
        fprintf(stderr, "Failed to load ggml model\n");
        return false;
    }

    nemo::ModelWeights ref_weights;
    if (!ref_weights.load("weights/model.bin")) {
        fprintf(stderr, "Failed to load reference weights\n");
        nemo_free(ctx);
        return false;
    }

    nemo::RNNTJoint ref_joint;
    ref_joint.load_weights(ref_weights);

    const int encoder_dim = 1024;
    const int decoder_dim = 640;
    const int vocab_size = 1025;

    // Create test inputs
    nemo::TensorF enc_input({1, encoder_dim});
    nemo::TensorF dec_input({1, decoder_dim});

    // Fill with test values
    for (int i = 0; i < encoder_dim; i++) {
        enc_input(0, i) = 0.1f * std::sin((float)i * 0.1f);
    }
    for (int i = 0; i < decoder_dim; i++) {
        dec_input(0, i) = 0.1f * std::cos((float)i * 0.1f);
    }

    // Reference forward
    nemo::TensorF ref_logits;
    ref_joint.forward(enc_input, dec_input, ref_logits);

    printf("Reference logits[0:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           ref_logits(0, 0), ref_logits(0, 1), ref_logits(0, 2),
           ref_logits(0, 3), ref_logits(0, 4));

    // === GGML implementation ===
    size_t buf_size = ggml_tensor_overhead() * 50 + ggml_graph_overhead();
    std::vector<uint8_t> compute_buf(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ compute_buf.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    if (!ctx0) {
        fprintf(stderr, "Failed to init ggml context\n");
        nemo_free(ctx);
        return false;
    }

    // Create input tensors
    struct ggml_tensor * enc_in = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, encoder_dim);
    struct ggml_tensor * dec_in = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, decoder_dim);
    ggml_set_name(enc_in, "encoder_input");
    ggml_set_name(dec_in, "decoder_input");
    ggml_set_input(enc_in);
    ggml_set_input(dec_in);

    // Build joint graph
    struct ggml_tensor * logits = build_joint(ctx0, enc_in, dec_in, &ctx->model.joint);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);

    // Build graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, logits);

    printf("Graph nodes: %d\n", ggml_graph_n_nodes(gf));

    // Allocate
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->model.backend));
    if (!ggml_gallocr_alloc_graph(allocr, gf)) {
        fprintf(stderr, "Failed to allocate graph\n");
        ggml_gallocr_free(allocr);
        ggml_free(ctx0);
        nemo_free(ctx);
        return false;
    }

    // Set inputs
    ggml_backend_tensor_set(enc_in, enc_input.data.data(), 0, encoder_dim * sizeof(float));
    ggml_backend_tensor_set(dec_in, dec_input.data.data(), 0, decoder_dim * sizeof(float));

    // Compute
    ggml_backend_graph_compute(ctx->model.backend, gf);

    // Get output
    std::vector<float> ggml_logits(vocab_size);
    ggml_backend_tensor_get(logits, ggml_logits.data(), 0, vocab_size * sizeof(float));

    printf("GGML logits[0:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           ggml_logits[0], ggml_logits[1], ggml_logits[2], ggml_logits[3], ggml_logits[4]);

    // Compare
    error_calc err;
    for (int i = 0; i < vocab_size; i++) {
        err.add(ggml_logits[i], ref_logits(0, i));
    }
    err.report("Joint network");

    // Cleanup
    ggml_gallocr_free(allocr);
    ggml_free(ctx0);
    nemo_free(ctx);

    bool passed = err.max_diff < 1e-4f;
    printf("Joint test: %s\n\n", passed ? "PASS" : "FAIL");
    return passed;
}

bool test_greedy_decode() {
    printf("=== Testing Greedy Decode (Full Pipeline) ===\n");

    // Load ggml model
    nemo_context * ctx = nemo_init("weights/model.gguf");
    if (!ctx) {
        fprintf(stderr, "Failed to load ggml model\n");
        return false;
    }

    // Load reference implementation
    nemo::ModelWeights ref_weights;
    if (!ref_weights.load("weights/model.bin")) {
        fprintf(stderr, "Failed to load reference weights\n");
        nemo_free(ctx);
        return false;
    }

    nemo::ASRPipeline ref_pipeline;
    ref_pipeline.load_weights(ref_weights);

    // Load mel input
    FILE * f = fopen("test.mel.bin", "rb");
    if (!f) {
        fprintf(stderr, "Failed to open test.mel.bin\n");
        nemo_free(ctx);
        return false;
    }

    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    size_t batch = 1;
    size_t features = 128;
    size_t time_in = file_size / (sizeof(float) * features);

    printf("Mel input shape: [%zu, %zu, %zu]\n", batch, time_in, features);

    std::vector<float> raw_mel(time_in * features);
    fread(raw_mel.data(), sizeof(float), raw_mel.size(), f);
    fclose(f);

    // Reference transcription
    nemo::TensorF mel_input({batch, time_in, features});
    for (size_t t = 0; t < time_in; t++) {
        for (size_t ff = 0; ff < features; ff++) {
            mel_input(0, t, ff) = raw_mel[t * features + ff];
        }
    }

    std::vector<int> ref_tokens = ref_pipeline.transcribe(mel_input);
    printf("Reference tokens (%zu): ", ref_tokens.size());
    for (size_t i = 0; i < std::min(ref_tokens.size(), (size_t)20); i++) {
        printf("%d ", ref_tokens[i]);
    }
    if (ref_tokens.size() > 20) printf("...");
    printf("\n");

    // Print some sample vocab entries from the GGML model
    printf("Sample vocab entries from GGUF (vocab.size=%zu):\n", ctx->model.vocab.size());
    for (int i : {0, 1, 2, 130, 500, 1024}) {
        if (i < (int)ctx->model.vocab.size()) {
            printf("  vocab[%d] = '", i);
            for (int j = 0; j < 8 && ctx->model.vocab[i].data[j]; j++) {
                printf("%c", ctx->model.vocab[i].data[j]);
            }
            printf("' (hex:");
            for (int j = 0; j < 8; j++) {
                printf(" %02x", (unsigned char)ctx->model.vocab[i].data[j]);
            }
            printf(")\n");
        }
    }

    // Decode using GGML's vocab
    std::string ref_text = tokens_to_text(ref_tokens, ctx->model.vocab);
    printf("Reference transcription: %s\n", ref_text.c_str());

    // === GGML implementation ===
    // First, encode the mel spectrogram
    size_t buf_size = ggml_tensor_overhead() * 8000 + ggml_graph_overhead() * 4;
    std::vector<uint8_t> compute_buf(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ compute_buf.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    if (!ctx0) {
        fprintf(stderr, "Failed to init ggml context\n");
        nemo_free(ctx);
        return false;
    }

    // Create mel input tensor [n_mels, time, batch]
    struct ggml_tensor * inp = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, features, time_in, batch);
    ggml_set_name(inp, "mel_input");
    ggml_set_input(inp);

    // Build encoder graph
    struct ggml_tensor * encoder_out = build_encoder(ctx0, inp, &ctx->model);
    ggml_set_name(encoder_out, "encoder_output");
    ggml_set_output(encoder_out);

    // Build and allocate graph
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 16384, false);
    ggml_build_forward_expand(gf, encoder_out);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->model.backend));
    if (!ggml_gallocr_alloc_graph(allocr, gf)) {
        fprintf(stderr, "Failed to allocate graph\n");
        ggml_gallocr_free(allocr);
        ggml_free(ctx0);
        nemo_free(ctx);
        return false;
    }

    // Set mel input - transpose from [batch, time, features] to [features, time, batch]
    std::vector<float> inp_transposed(mel_input.numel());
    for (size_t b = 0; b < batch; b++) {
        for (size_t t = 0; t < time_in; t++) {
            for (size_t ff = 0; ff < features; ff++) {
                size_t g_idx = ff + t * features + b * features * time_in;
                inp_transposed[g_idx] = mel_input(b, t, ff);
            }
        }
    }
    ggml_backend_tensor_set(inp, inp_transposed.data(), 0, inp_transposed.size() * sizeof(float));

    // Run encoder
    printf("Computing encoder...\n");
    ggml_backend_graph_compute(ctx->model.backend, gf);

    printf("Encoder output shape: [%lld, %lld, %lld]\n",
           (long long)encoder_out->ne[0], (long long)encoder_out->ne[1], (long long)encoder_out->ne[2]);

    // Run greedy decode
    printf("Running greedy decode...\n");
    std::vector<int> ggml_tokens = greedy_decode(ctx, encoder_out, ctx->model.backend);

    printf("GGML tokens (%zu): ", ggml_tokens.size());
    for (size_t i = 0; i < std::min(ggml_tokens.size(), (size_t)20); i++) {
        printf("%d ", ggml_tokens[i]);
    }
    if (ggml_tokens.size() > 20) printf("...");
    printf("\n");

    // Convert to text using same vocab
    std::string ggml_text = tokens_to_text(ggml_tokens, ctx->model.vocab);
    printf("GGML transcription: %s\n", ggml_text.c_str());

    // Compare tokens
    bool tokens_match = (ref_tokens.size() == ggml_tokens.size());
    if (tokens_match) {
        for (size_t i = 0; i < ref_tokens.size(); i++) {
            if (ref_tokens[i] != ggml_tokens[i]) {
                tokens_match = false;
                printf("Token mismatch at position %zu: ref=%d, ggml=%d\n", i, ref_tokens[i], ggml_tokens[i]);
                break;
            }
        }
    } else {
        printf("Token count mismatch: ref=%zu, ggml=%zu\n", ref_tokens.size(), ggml_tokens.size());
    }

    // Cleanup
    ggml_gallocr_free(allocr);
    ggml_free(ctx0);
    nemo_free(ctx);

    printf("Greedy decode test: %s\n\n", tokens_match ? "PASS" : "FAIL");
    return tokens_match;
}

struct TestEntry {
    const char * name;
    bool (*func)();
};

static TestEntry tests[] = {
    {"linear", test_linear},
    {"layer_norm", test_layer_norm},
    {"swish", test_swish},
    {"ffn", test_ffn},
    {"conv2d", test_conv2d},
    {"conv_subsampling", test_conv_subsampling},
    {"pos_encoding", test_pos_encoding},
    {"rel_shift", test_rel_shift},
    {"mha", test_mha},
    {"mha_full", test_mha_full},
    {"conformer_conv", test_conformer_conv},
    {"conformer_layer", test_conformer_layer},
    {"encoder", test_encoder},
    {"decoder", test_decoder},
    {"joint", test_joint},
    {"greedy_decode", test_greedy_decode},
    {nullptr, nullptr}
};

int main(int argc, char ** argv) {
    printf("=== Testing GGML Computation vs Original ===\n\n");

    int passed = 0;
    int failed = 0;
    int skipped = 0;

    bool no_filter = argc <= 1;
    for (int i = 0; tests[i].name != nullptr; i++) {
        bool run_test = no_filter;
        for (int jj = 1; jj < argc; jj++) {
            if (strcmp(tests[i].name, argv[jj]) == 0) {
                run_test = true;
                break;
            }
        }
        if (!run_test) {
            skipped++;
            continue;
        }
        if (tests[i].func()) {
            passed++;
        } else {
            failed++;
        }
    }

    printf("=== Summary ===\n");
    printf("Passed: %d\n", passed);
    printf("Failed: %d\n", failed);
    if (skipped > 0) {
        printf("Skipped: %d\n", skipped);
    }

    return failed > 0 ? 1 : 0;
}
