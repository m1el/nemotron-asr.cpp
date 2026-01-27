// Simple example: Transcribe audio using GGML-based NeMo ASR
//
// Usage: ./transcribe model.gguf input_file [--mel]
//
// Input formats:
//   - Raw PCM audio: 16-bit signed, 16kHz, mono (.pcm, .raw)
//   - Mel spectrogram: float32 [time, 128] row-major (.mel.bin) with --mel flag

#include "../src-ggml/nemo-ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s <model.gguf> <input_file> [--mel] [--cpu|--cuda]\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "  model.gguf  - GGUF model file (convert with scripts/convert_to_gguf.py)\n");
    fprintf(stderr, "  input_file  - Audio file (PCM i16le 16kHz mono) or mel spectrogram\n");
    fprintf(stderr, "  --mel       - Input is mel spectrogram [time, 128] float32 (optional)\n");
    fprintf(stderr, "  --cpu       - Force CPU backend\n");
    fprintf(stderr, "  --cuda      - Force CUDA backend\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Examples:\n");
    fprintf(stderr, "  %s weights/model.gguf audio.pcm          # PCM audio input (auto backend)\n", prog);
    fprintf(stderr, "  %s weights/model.gguf audio.pcm --cuda   # Force CUDA backend\n", prog);
    fprintf(stderr, "  %s weights/model.gguf test.mel.bin --mel # Mel spectrogram input\n", prog);
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    const char * input_path = argv[2];

    // Parse optional args
    nemo_backend_type backend = NEMO_BACKEND_AUTO;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--cpu") == 0) {
            backend = NEMO_BACKEND_CPU;
        } else if (strcmp(argv[i], "--cuda") == 0) {
            backend = NEMO_BACKEND_CUDA;
        }
    }

    // Load model
    printf("Loading model from %s...\n", model_path);
    struct nemo_context * ctx = nemo_init_with_backend(model_path, backend);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    printf("Model loaded successfully (backend: %s)\n", nemo_get_backend_name(ctx));

    // Load input file
    FILE * f = fopen(input_path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", input_path);
        nemo_free(ctx);
        return 1;
    }

    // Get file size
    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    std::string text;


    // PCM audio input (16-bit signed, 16kHz, mono)
    int n_samples = file_size / sizeof(int16_t);

    printf("Audio input: %d samples (%.2f seconds at 16kHz)\n",
            n_samples, (float)n_samples / 16000.0f);

    std::vector<int16_t> audio_data(n_samples);
    size_t read = fread(audio_data.data(), sizeof(int16_t), audio_data.size(), f);
    fclose(f);

    if (read != audio_data.size()) {
        fprintf(stderr, "Failed to read audio data\n");
        nemo_free(ctx);
        return 1;
    }

    printf("Transcribing...\n");
    text = nemo_transcribe_audio(ctx, audio_data.data(), n_samples);

    if (text.empty()) {
        fprintf(stderr, "Transcription failed or produced no output\n");
        nemo_free(ctx);
        return 1;
    }

    printf("\n=== Transcription ===\n");
    printf("%s\n", text.c_str());
    printf("=====================\n");

    // Cleanup
    nemo_free(ctx);

    return 0;
}
