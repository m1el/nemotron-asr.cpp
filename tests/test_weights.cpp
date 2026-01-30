// Test ggml/GGUF weight loading and compare with original implementation
#include "../src-ggml/nemo-ggml.h"
#include "../include/ggml_weights.h"

#include <cstdio>
#include <cmath>
#include <cstring>

// Compare two tensors
bool compare_tensors(struct ggml_tensor * ggml_t, const nemo::Tensor & ref, const char * name) {
    // Check element count
    size_t ggml_numel = ggml_nelements(ggml_t);
    size_t ref_numel = ref.numel();

    if (ggml_numel != ref_numel) {
        printf("FAIL %s: element count mismatch (ggml=%zu, ref=%zu)\n", name, ggml_numel, ref_numel);
        return false;
    }

    // Get ggml data
    std::vector<float> ggml_data(ggml_numel);
    ggml_backend_tensor_get(ggml_t, ggml_data.data(), 0, ggml_numel * sizeof(float));

    // Compare values
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    for (size_t i = 0; i < ggml_numel; i++) {
        float diff = std::abs(ggml_data[i] - ref.data[i]);
        max_diff = std::max(max_diff, diff);
        sum_diff += diff;
    }

    float mean_diff = sum_diff / ggml_numel;

    if (max_diff > 1e-5f) {
        printf("FAIL %s: max_diff=%.6e, mean_diff=%.6e\n", name, max_diff, mean_diff);

        // Print first few values
        printf("  ggml[0:5]: ");
        for (int i = 0; i < std::min(5, (int)ggml_numel); i++) {
            printf("%.6f ", ggml_data[i]);
        }
        printf("\n  ref[0:5]:  ");
        for (int i = 0; i < std::min(5, (int)ref_numel); i++) {
            printf("%.6f ", ref.data[i]);
        }
        printf("\n");
        return false;
    }

    printf("PASS %s: max_diff=%.6e, mean_diff=%.6e\n", name, max_diff, mean_diff);
    return true;
}

int main() {
    printf("=== Testing GGUF Weight Loading ===\n\n");

    // Load with original implementation
    printf("Loading weights with original implementation (model.bin)...\n");
    nemo::ModelWeights ref_weights;
    if (!ref_weights.load("weights/model.bin")) {
        fprintf(stderr, "Failed to load reference weights\n");
        return 1;
    }
    printf("\n");

    // Load with ggml implementation (GGUF format)
    printf("Loading weights with ggml implementation (model.gguf)...\n");
    nemo_context * ctx = nemo_init("weights/model.gguf");
    if (!ctx) {
        fprintf(stderr, "Failed to load ggml model\n");
        return 1;
    }
    printf("\n");

    // Compare key tensors
    printf("=== Comparing Tensors ===\n\n");

    int passed = 0;
    int failed = 0;

    auto test_tensor = [&](const char * name) {
        auto it = ctx->model.tensors.find(name);
        if (it == ctx->model.tensors.end()) {
            printf("SKIP %s: not in ggml model\n", name);
            return;
        }
        struct ggml_tensor * ggml_t = it->second;

        const auto * ref_t = ref_weights.get(name);
        if (!ref_t) {
            printf("SKIP %s: not in reference\n", name);
            return;
        }

        if (compare_tensors(ggml_t, *ref_t, name)) {
            passed++;
        } else {
            failed++;
        }
    };

    // Test conv subsampling
    test_tensor("encoder.pre_encode.conv.0.weight");
    test_tensor("encoder.pre_encode.conv.0.bias");
    test_tensor("encoder.pre_encode.out.weight");

    // Test layer 0 weights
    test_tensor("encoder.layers.0.norm_feed_forward1.weight");
    test_tensor("encoder.layers.0.feed_forward1.linear1.weight");
    test_tensor("encoder.layers.0.self_attn.linear_q.weight");
    test_tensor("encoder.layers.0.self_attn.pos_bias_u");
    test_tensor("encoder.layers.0.conv_module.pointwise_conv1.weight");
    test_tensor("encoder.layers.0.conv_module.depthwise_conv.weight");

    // Test encoder output (no final norm/fc in this model, uses joint.enc instead)

    // Test decoder
    test_tensor("decoder.prediction.embed.weight");
    test_tensor("decoder.prediction.dec_rnn.lstm.weight_ih_l0");
    test_tensor("decoder.prediction.fc.weight");

    // Test joint
    test_tensor("joint.enc.weight");
    test_tensor("joint.pred.weight");
    test_tensor("joint.joint_net.2.weight");
    test_tensor("joint.joint_net.2.bias");

    printf("\n=== Summary ===\n");
    printf("Passed: %d\n", passed);
    printf("Failed: %d\n", failed);

    nemo_free(ctx);

    return failed > 0 ? 1 : 0;
}
