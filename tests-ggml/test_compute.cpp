// Test that ggml computation produces same results as original implementation
#include "../src-ggml/nemo-ggml.h"
#include "../include/ggml_weights.h"
#include "../include/ops.h"

#include <cstdio>
#include <cmath>
#include <cstring>

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
    int out_features = ref_w->shape[0];  // 4096
    int in_features = ref_w->shape[1];   // 1024

    printf("Weight shape: [%d, %d]\n", out_features, in_features);

    // Create test input [1, 5, 1024]
    int batch = 1;
    int seq_len = 5;
    nemo::TensorF input({(size_t)batch, (size_t)seq_len, (size_t)in_features});

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
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true,
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
    float max_diff = 0.0f;
    for (size_t i = 0; i < ref_output.numel(); i++) {
        float diff = std::abs(ggml_output[i] - ref_output.data[i]);
        max_diff = std::max(max_diff, diff);
    }

    printf("Max diff: %.6e\n", max_diff);

    // Cleanup
    ggml_gallocr_free(allocr);
    ggml_free(ctx0);
    nemo_free(ctx);

    bool passed = max_diff < 1e-4f;
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

    int d_model = ref_norm_w->shape[0];  // 1024
    printf("d_model: %d\n", d_model);

    // Create test input [1, 5, 1024]
    int batch = 1;
    int seq_len = 5;
    nemo::TensorF input({(size_t)batch, (size_t)seq_len, (size_t)d_model});

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
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true,
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
    float max_diff = 0.0f;
    for (size_t i = 0; i < ref_output.numel(); i++) {
        float diff = std::abs(ggml_output[i] - ref_output.data[i]);
        max_diff = std::max(max_diff, diff);
    }

    printf("Max diff: %.6e\n", max_diff);

    // Cleanup
    ggml_gallocr_free(allocr);
    ggml_free(ctx0);
    nemo_free(ctx);

    bool passed = max_diff < 1e-4f;
    printf("Layer norm test: %s\n\n", passed ? "PASS" : "FAIL");
    return passed;
}

int main() {
    printf("=== Testing GGML Computation vs Original ===\n\n");

    int passed = 0;
    int failed = 0;

    if (test_linear()) passed++; else failed++;
    if (test_layer_norm()) passed++; else failed++;

    printf("=== Summary ===\n");
    printf("Passed: %d\n", passed);
    printf("Failed: %d\n", failed);

    return failed > 0 ? 1 : 0;
}
