// Streaming transcription example
//
// Demonstrates cache-aware streaming ASR with real-time processing
//
// Usage: ./transcribe_stream model.gguf audio.pcm [chunk_ms]
//        ./transcribe_stream model.gguf - [chunk_ms]   # read from stdin
//
// Input: Raw PCM audio: 16-bit signed, 16kHz, mono

#include "nemo-ggml.h"
#include "nemo-stream.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <fstream>
#include <fcntl.h>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s <model.gguf> <audio.pcm|-|--stdin> [chunk_ms] [right_context] [--cpu|--cuda]\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "  model.gguf      - GGUF model file\n");
    fprintf(stderr, "  audio.pcm       - Audio file (PCM i16le 16kHz mono)\n");
    fprintf(stderr, "  - or --stdin    - Read audio from stdin\n");
    fprintf(stderr, "  chunk_ms        - Chunk size in milliseconds (default: 80)\n");
    fprintf(stderr, "  right_context   - Attention right context (0, 1, 6, or 13, default: 0)\n");
    fprintf(stderr, "  --cpu           - Force CPU backend\n");
    fprintf(stderr, "  --cuda          - Force CUDA backend\n");
    fprintf(stderr, "  --metal         - Force Metal backend\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Streaming modes:\n");
    fprintf(stderr, "  right_context=0  - Pure causal, 80ms latency\n");
    fprintf(stderr, "  right_context=1  - 160ms latency\n");
    fprintf(stderr, "  right_context=6  - 560ms latency\n");
    fprintf(stderr, "  right_context=13 - 1120ms latency (best quality)\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Examples:\n");
    fprintf(stderr, "  %s weights/model.gguf audio.pcm 80 0\n", prog);
    fprintf(stderr, "  %s weights/model.gguf audio.pcm 80 0 --cuda\n", prog);
    fprintf(stderr, "  ffmpeg -i audio.mp3 -f s16le -ar 16000 -ac 1 - | %s weights/model.gguf -\n", prog);
    fprintf(stderr, "  arecord -f S16_LE -r 16000 -c 1 - | %s weights/model.gguf - 80 0 --cuda\n", prog);
    fprintf(stderr, "  %s weights/model.gguf audio.pcm 80 0 --metal\n", prog);
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
    bool read_from_stdin = (strcmp(audio_path, "-") == 0 || strcmp(audio_path, "--stdin") == 0);

    // Parse arguments
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--cpu") == 0) {
            backend = NEMO_BACKEND_CPU;
        } else if (strcmp(argv[i], "--cuda") == 0) {
            backend = NEMO_BACKEND_CUDA;
        } else if (strcmp(argv[i], "--metal") == 0) {
            backend = NEMO_BACKEND_METAL;
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

    fprintf(stderr, "Configuration:\n");
    fprintf(stderr, "  Model:          %s\n", model_path);
    fprintf(stderr, "  Audio:          %s\n", read_from_stdin ? "stdin" : audio_path);
    fprintf(stderr, "  Chunk size:     %d ms (%d samples)\n", chunk_ms, chunk_samples);
    fprintf(stderr, "  Right context:  %d (latency: %d ms)\n", right_context, 80 + right_context * 80);
    fprintf(stderr, "\n");

    fprintf(stderr, "Loading model from %s...\n", model_path);
    struct nemo_context * ctx = nemo_init_with_backend(model_path, backend);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    fprintf(stderr, "Model loaded successfully (backend: %s)\n", nemo_get_backend_name(ctx));

    // Initialize cache-aware streaming context
    nemo_cache_config cache_cfg = nemo_cache_config::default_config();
    cache_cfg.att_right_context = right_context;  // User-selected latency mode

    struct nemo_stream_context * sctx = nemo_stream_init(ctx, &cache_cfg);
    if (!sctx) {
        fprintf(stderr, "Failed to create streaming context\n");
        nemo_free(ctx);
        return 1;
    }

    int computed_chunk_samples = cache_cfg.get_chunk_samples();
    size_t total_samples_processed = 0;

    // Open input source (stdin or file)
    FILE* input = nullptr;
    if (read_from_stdin) {
#ifdef _WIN32
        _setmode(_fileno(stdin), _O_BINARY);
#endif
        input = stdin;
        fprintf(stderr, "Reading audio from stdin...\n\n");
    } else {
        input = fopen(audio_path, "rb");
        if (!input) {
            fprintf(stderr, "Failed to open audio file: %s\n", audio_path);
            nemo_stream_free(sctx);
            nemo_free(ctx);
            return 1;
        }
        fprintf(stderr, "Streaming from file...\n\n");
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<int16_t> buffer(computed_chunk_samples);

    while (true) {
        size_t bytes_to_read = computed_chunk_samples * sizeof(int16_t);
        size_t bytes_read = fread(buffer.data(), 1, bytes_to_read, input);

        if (bytes_read == 0) {
            break;
        }

        int n_samples = bytes_read / sizeof(int16_t);
        total_samples_processed += n_samples;

        std::string new_text = nemo_stream_process_incremental(sctx, buffer.data(), n_samples);

        if (!new_text.empty()) {
            printf("%s", new_text.c_str());
            fflush(stdout);
        }

        if (bytes_read < bytes_to_read) {
            break;
        }
    }

    // Finalize to flush any remaining audio
    std::string final_text = nemo_stream_finalize(sctx);
    if (!final_text.empty()) {
        printf("%s", final_text.c_str());
        fflush(stdout);
    }
    printf("\n");

    if (!read_from_stdin && input) {
        fclose(input);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double processing_time_sec = std::chrono::duration<double>(end_time - start_time).count();
    float total_duration_sec = (float)total_samples_processed / 16000.0f;

    fprintf(stderr, "\n=== Complete ===\n");
    fprintf(stderr, "\nStatistics:\n");
    fprintf(stderr, "  Chunks processed:    %d\n", sctx->total_chunks_processed);
    fprintf(stderr, "  Audio duration:      %.2f sec\n", total_duration_sec);
    fprintf(stderr, "  Processing time:     %.2f sec\n", processing_time_sec);
    if (total_duration_sec > 0) {
        fprintf(stderr, "  Real-time factor:    %.3fx\n", processing_time_sec / total_duration_sec);
    }

    // Cleanup
    nemo_stream_free(sctx);
    nemo_free(ctx);

    return 0;
}
