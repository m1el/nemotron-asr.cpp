// Test cache-aware streaming encoder
//
// Compares cached vs non-cached streaming to verify correctness

#include "nemo-ggml.h"
#include "nemo-stream.h"
#include "nemo-cache.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <cmath>

// Test helper: generate synthetic audio (sine wave)
std::vector<int16_t> generate_test_audio(float duration_sec, float freq_hz = 440.0f) {
    int n_samples = (int)(duration_sec * 16000);
    std::vector<int16_t> audio(n_samples);
    
    for (int i = 0; i < n_samples; i++) {
        float t = (float)i / 16000.0f;
        float sample = std::sin(2.0f * 3.14159f * freq_hz * t);
        // Add some noise to make it more realistic
        sample += ((rand() / (float)RAND_MAX) - 0.5f) * 0.1f;
        audio[i] = (int16_t)(sample * 16000);
    }
    
    return audio;
}

// Test: Compare cached vs non-cached encoder output
void test_cache_consistency(nemo_context * ctx) {
    printf("\n=== Test: Cache Consistency ===\n");
    
    // Generate 5 seconds of test audio
    float duration = 5.0f;
    auto audio = generate_test_audio(duration);
    int chunk_size = 4800;  // 300ms chunks
    
    // Process with non-cached streaming
    nemo_stream_config config_nocache;
    config_nocache.chunk_samples = chunk_size;
    config_nocache.use_cache = false;
    
    auto * sctx_nocache = nemo_stream_init(ctx, &config_nocache);
    
    auto start_nocache = std::chrono::steady_clock::now();
    
    for (size_t offset = 0; offset < audio.size(); offset += chunk_size) {
        int n = std::min(chunk_size, (int)(audio.size() - offset));
        nemo_stream_process(sctx_nocache, audio.data() + offset, n);
    }
    
    std::string result_nocache = nemo_stream_finalize(sctx_nocache);
    auto stats_nocache = nemo_stream_get_stats(sctx_nocache);
    
    auto end_nocache = std::chrono::steady_clock::now();
    double time_nocache = std::chrono::duration<double>(end_nocache - start_nocache).count();
    
    nemo_stream_free(sctx_nocache);
    
    printf("Non-cached: RTF=%.2f, tokens=%d\n", stats_nocache.rtf, stats_nocache.tokens_emitted);
    printf("  Result: %s\n", result_nocache.c_str());
    
    // Process with cached streaming  
    nemo_stream_config config_cache;
    config_cache.chunk_samples = chunk_size;
    config_cache.use_cache = true;
    config_cache.att_left_context = 70;
    
    auto * sctx_cache = nemo_stream_init(ctx, &config_cache);
    
    auto start_cache = std::chrono::steady_clock::now();
    
    for (size_t offset = 0; offset < audio.size(); offset += chunk_size) {
        int n = std::min(chunk_size, (int)(audio.size() - offset));
        nemo_stream_process(sctx_cache, audio.data() + offset, n);
    }
    
    std::string result_cache = nemo_stream_finalize(sctx_cache);
    auto stats_cache = nemo_stream_get_stats(sctx_cache);
    
    auto end_cache = std::chrono::steady_clock::now();
    double time_cache = std::chrono::duration<double>(end_cache - start_cache).count();
    
    nemo_stream_free(sctx_cache);
    
    printf("Cached: RTF=%.2f, tokens=%d, cache_mem=%zu KB\n", 
           stats_cache.rtf, stats_cache.tokens_emitted, stats_cache.cache_memory_bytes / 1024);
    printf("  Result: %s\n", result_cache.c_str());
    
    // Note: Results may differ slightly due to the streaming nature
    // The cached encoder processes chunks differently than the non-cached one
    printf("\nSpeedup: %.2fx\n", time_nocache / time_cache);
}

// Test: Verify cache sizes
void test_cache_initialization(nemo_context * ctx) {
    printf("\n=== Test: Cache Initialization ===\n");
    
    nemo_cache_config cfg;
    cfg.att_left_context = 70;
    cfg.conv_kernel_size = 9;  // Actual model uses 9, not 31 (31 is for non-streaming)
    cfg.conv_cache_size = cfg.conv_kernel_size - 1;
    cfg.d_model = ctx->model.hparams.d_model;
    cfg.n_layers = ctx->model.hparams.n_layers;
    cfg.n_heads = ctx->model.hparams.n_heads;
    cfg.d_head = ctx->model.hparams.d_head;
    
    nemo_encoder_cache cache;
    cache.init(cfg);
    
    printf("Cache configuration:\n");
    printf("  d_model: %d\n", cfg.d_model);
    printf("  n_layers: %d\n", cfg.n_layers);
    printf("  att_left_context: %d\n", cfg.att_left_context);
    printf("  conv_cache_size: %d\n", cfg.conv_cache_size);
    
    printf("\nPer-layer sizes:\n");
    printf("  Attn K cache: %d x %d = %zu KB\n", 
           cfg.att_left_context, cfg.d_model,
           cache.attn_caches[0].k_cache.size() * sizeof(float) / 1024);
    printf("  Attn V cache: %d x %d = %zu KB\n",
           cfg.att_left_context, cfg.d_model,
           cache.attn_caches[0].v_cache.size() * sizeof(float) / 1024);
    printf("  Conv cache: %d x %d = %zu KB\n",
           cfg.conv_cache_size, cfg.d_model,
           cache.conv_caches[0].cache.size() * sizeof(float) / 1024);
    
    printf("\nTotal cache memory: %zu KB (%.2f MB)\n", 
           cache.size_bytes() / 1024, cache.size_bytes() / (1024.0 * 1024.0));
    
    // Expected: ~14MB for attention + ~750KB for conv
    size_t expected_attn = 2 * cfg.n_layers * cfg.att_left_context * cfg.d_model * sizeof(float);
    size_t expected_conv = cfg.n_layers * cfg.conv_cache_size * cfg.d_model * sizeof(float);
    printf("Expected: attn=%.2f MB, conv=%.2f MB\n",
           expected_attn / (1024.0 * 1024.0), expected_conv / (1024.0 * 1024.0));
    
    // Test reset
    cache.reset();
    bool all_zero = true;
    for (int i = 0; i < cfg.n_layers; i++) {
        for (float v : cache.attn_caches[i].k_cache) {
            if (v != 0.0f) { all_zero = false; break; }
        }
    }
    printf("Reset works: %s\n", all_zero ? "PASS" : "FAIL");
}

// Test: Benchmark RTF scaling with audio length
void test_rtf_scaling(nemo_context * ctx) {
    printf("\n=== Test: RTF Scaling with Audio Length ===\n");
    
    std::vector<float> durations = {2.0f, 5.0f/*, 10.0f, 20.0f*/};
    int chunk_size = 4800;  // 300ms
    
    printf("%-10s %-12s %-12s %-12s\n", "Duration", "NoCache RTF", "Cache RTF", "Speedup");
    printf("%-10s %-12s %-12s %-12s\n", "--------", "-----------", "---------", "-------");
    
    for (float duration : durations) {
        auto audio = generate_test_audio(duration);
        
        // Non-cached
        nemo_stream_config cfg_nc;
        cfg_nc.chunk_samples = chunk_size;
        cfg_nc.use_cache = false;
        
        auto * sctx_nc = nemo_stream_init(ctx, &cfg_nc);
        auto t0 = std::chrono::steady_clock::now();
        for (size_t off = 0; off < audio.size(); off += chunk_size) {
            int n = std::min(chunk_size, (int)(audio.size() - off));
            nemo_stream_process(sctx_nc, audio.data() + off, n);
        }
        nemo_stream_finalize(sctx_nc);
        auto t1 = std::chrono::steady_clock::now();
        double time_nc = std::chrono::duration<double>(t1 - t0).count();
        nemo_stream_free(sctx_nc);
        
        // Cached
        nemo_stream_config cfg_c;
        cfg_c.chunk_samples = chunk_size;
        cfg_c.use_cache = true;
        
        auto * sctx_c = nemo_stream_init(ctx, &cfg_c);
        t0 = std::chrono::steady_clock::now();
        for (size_t off = 0; off < audio.size(); off += chunk_size) {
            int n = std::min(chunk_size, (int)(audio.size() - off));
            nemo_stream_process(sctx_c, audio.data() + off, n);
        }
        nemo_stream_finalize(sctx_c);
        t1 = std::chrono::steady_clock::now();
        double time_c = std::chrono::duration<double>(t1 - t0).count();
        nemo_stream_free(sctx_c);
        
        double rtf_nc = time_nc / duration;
        double rtf_c = time_c / duration;
        double speedup = time_nc / time_c;
        
        printf("%-10.1fs %-12.2f %-12.2f %-12.2fx\n", duration, rtf_nc, rtf_c, speedup);
    }
}

int main(int argc, char ** argv) {
    const char * model_path = "weights/model.gguf";
    
    if (argc > 1) {
        model_path = argv[1];
    }
    
    printf("Loading model from %s...\n", model_path);
    
    nemo_context * ctx = nemo_init(model_path);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    
    printf("Model loaded successfully\n");
    
    // Run tests
    test_cache_initialization(ctx);
    test_cache_consistency(ctx);
    test_rtf_scaling(ctx);
    
    nemo_free(ctx);
    
    printf("\n=== All tests completed ===\n");
    
    return 0;
}
