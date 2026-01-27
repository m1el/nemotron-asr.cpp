// Streaming transcription example
//
// Demonstrates cache-aware streaming ASR with real-time processing
//
// Usage: ./transcribe_stream model.gguf audio.pcm [chunk_ms]
//
// Input: Raw PCM audio: 16-bit signed, 16kHz, mono

#include "../src-ggml/nemo-ggml.h"
#include "../src-ggml/nemo-stream.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <fstream>

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s <model.gguf> <audio.pcm> [chunk_ms] [right_context] [--cpu|--cuda]\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "  model.gguf      - GGUF model file\n");
    fprintf(stderr, "  audio.pcm       - Audio file (PCM i16le 16kHz mono)\n");
    fprintf(stderr, "  chunk_ms        - Chunk size in milliseconds (default: 80)\n");
    fprintf(stderr, "  right_context   - Attention right context (0, 1, 6, or 13, default: 0)\n");
    fprintf(stderr, "  --cpu           - Force CPU backend\n");
    fprintf(stderr, "  --cuda          - Force CUDA backend\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Streaming modes:\n");
    fprintf(stderr, "  right_context=0  - Pure causal, 80ms latency\n");
    fprintf(stderr, "  right_context=1  - 160ms latency\n");
    fprintf(stderr, "  right_context=6  - 560ms latency\n");
    fprintf(stderr, "  right_context=13 - 1120ms latency (best quality)\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Example:\n");
    fprintf(stderr, "  %s weights/model.gguf audio.pcm 80 0\n", prog);
    fprintf(stderr, "  %s weights/model.gguf audio.pcm 80 0 --cuda\n", prog);
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    const char * audio_path = argv[2];
    int chunk_ms = 80;  // default 80ms chunks
    int right_context = 0;  // default: pure causal
    nemo_backend_type backend = NEMO_BACKEND_AUTO;

    // Parse arguments
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--cpu") == 0) {
            backend = NEMO_BACKEND_CPU;
        } else if (strcmp(argv[i], "--cuda") == 0) {
            backend = NEMO_BACKEND_CUDA;
        } else if (i == 3) {
            chunk_ms = atoi(argv[i]);
            if (chunk_ms < 10) {
                fprintf(stderr, "Error: chunk_ms must be >= 10 (got %d)\n", chunk_ms);
                return 1;
            }
        } else if (i == 4) {
            right_context = atoi(argv[i]);
            if (right_context != 0 && right_context != 1 && 
                right_context != 6 && right_context != 13) {
                fprintf(stderr, "Warning: non-standard right_context=%d (use 0, 1, 6, or 13)\n", 
                        right_context);
            }
        }
    }

    // Calculate chunk size in samples  
    // chunk_ms should be multiple of 80ms (min frame size after 8x subsampling)
    int chunk_samples = chunk_ms * 16;  // 16 samples per ms at 16kHz

    printf("Configuration:\n");
    printf("  Model:          %s\n", model_path);
    printf("  Audio:          %s\n", audio_path);
    printf("  Chunk size:     %d ms (%d samples)\n", chunk_ms, chunk_samples);
    printf("  Right context:  %d (latency: %d ms)\n", right_context, 80 + right_context * 80);
    printf("\n");

    printf("Loading model from %s...\n", model_path);
    struct nemo_context * ctx = nemo_init_with_backend(model_path, backend);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    printf("Model loaded successfully (backend: %s)\n", nemo_get_backend_name(ctx));

    // Load audio file
    std::ifstream audio_file(audio_path, std::ios::binary);
    if (!audio_file.is_open()) {
        fprintf(stderr, "Failed to open audio file: %s\n", audio_path);
        nemo_free(ctx);
        return 1;
    }

    audio_file.seekg(0, std::ios::end);
    size_t file_size = audio_file.tellg();
    audio_file.seekg(0, std::ios::beg);

    int total_samples = file_size / sizeof(int16_t);
    std::vector<int16_t> audio_data(total_samples);

    audio_file.read(reinterpret_cast<char*>(audio_data.data()), file_size);
    audio_file.close();

    float total_duration_sec = (float)total_samples / 16000.0f;
    printf("Audio: %d samples (%.2f seconds)\n", total_samples, total_duration_sec);
    printf("\n");

    // Initialize cache-aware streaming context
    nemo_cache_config cache_cfg = nemo_cache_config::default_config();
    cache_cfg.att_right_context = right_context;  // User-selected latency mode
    cache_cfg.chunk_samples = chunk_samples;

    struct nemo_stream_context * sctx = nemo_stream_init(ctx, &cache_cfg);
    if (!sctx) {
        fprintf(stderr, "Failed to create streaming context\n");
        nemo_free(ctx);
        return 1;
    }

    printf("Cache-aware streaming initialized:\n");
    printf("  Attention left context:  %d frames\n", cache_cfg.att_left_context);
    printf("  Attention right context: %d frames\n", cache_cfg.att_right_context);
    printf("  Conv kernel size:        %d\n", cache_cfg.conv_kernel_size);
    printf("  Number of layers:        %d\n", cache_cfg.n_layers);
    printf("  Model dimension:         %d\n", cache_cfg.d_model);
    printf("\n");

    // Process audio in chunks
    printf("=== Streaming Transcription ===\n");
    printf("(Processing audio in 80ms chunks...)\n\n");
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    int chunks_processed = 0;
    int samples_processed = 0;
    
    while (samples_processed < total_samples) {
        int remaining = total_samples - samples_processed;
        int n_samples = (remaining < chunk_samples) ? remaining : chunk_samples;
        
        const int16_t * chunk_ptr = audio_data.data() + samples_processed;
        
        // Process chunk - accumulates audio, O(1) per chunk
        nemo_stream_process(sctx, chunk_ptr, n_samples);
        
        samples_processed += n_samples;
        chunks_processed++;
    }
    
    // Show we're about to finalize
    printf("Accumulated %d chunks (%.2f sec), now transcribing...\n", 
           chunks_processed, total_duration_sec);

    // Finalize - flush any remaining buffered audio
    printf("\nFinalizing...\n");
    std::string final_text = nemo_stream_finalize(sctx);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double processing_time_sec = std::chrono::duration<double>(end_time - start_time).count();

    printf("\n=== Final Transcription ===\n");
    printf("%s\n", final_text.c_str());
    printf("============================\n");

    // Print statistics
    printf("\nStatistics:\n");
    printf("  Chunks processed:    %d\n", chunks_processed);
    printf("  Audio duration:      %.2f sec\n", total_duration_sec);
    printf("  Processing time:     %.2f sec\n", processing_time_sec);
    printf("  Real-time factor:    %.3fx\n", processing_time_sec / total_duration_sec);

    // Cleanup
    nemo_stream_free(sctx);
    nemo_free(ctx);

    return 0;
}
