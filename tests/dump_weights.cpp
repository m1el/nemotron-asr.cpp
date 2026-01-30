// Quick test to dump GGML attention weights for comparison
#include "../src-ggml/nemo-ggml.h"
#include <cstdio>
#include <vector>

int main() {
    printf("Loading model...\n");
    nemo_context* ctx = nemo_init("weights/model.gguf");
    if (!ctx) {
        printf("Failed to load model\n");
        return 1;
    }
    
    // Get layer 0 attention weights
    nemo_conformer_layer* layer = &ctx->model.encoder.layers[0];
    
    // Check Q weight dimensions
    printf("attn_q_w shape: [%lld, %lld]\n", 
           (long long)layer->attn_q_w->ne[0], 
           (long long)layer->attn_q_w->ne[1]);
    
    // Get first few values
    std::vector<float> q_w(1024 * 5);
    ggml_backend_tensor_get(layer->attn_q_w, q_w.data(), 0, q_w.size() * sizeof(float));
    
    printf("attn_q_w[0,:5]: ");
    for (int i = 0; i < 5; i++) {
        printf("%.6f ", q_w[i]);
    }
    printf("\n");
    
    // Compute checksum
    std::vector<float> full_q_w(1024 * 1024);
    ggml_backend_tensor_get(layer->attn_q_w, full_q_w.data(), 0, full_q_w.size() * sizeof(float));
    double sum = 0;
    for (auto v : full_q_w) sum += v;
    printf("attn_q_w sum: %.6f\n", sum);
    
    nemo_free(ctx);
    return 0;
}
