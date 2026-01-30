#include "../src-ggml/nemo-ggml.h"
#include "../src-ggml/preprocessor.h"
#include <stdint.h>
#include <vector>
#include <cassert>
#include <cmath>


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

std::vector<float> read_sample_mel(const char * filename) {
    FILE * f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open mel data file\n");
        return {};
    }
    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    size_t n_samples = file_size / sizeof(float);
    std::vector<float> mel_data(n_samples);
    size_t bread = fread(mel_data.data(), sizeof(float), n_samples, f);
    assert(bread == n_samples);
    fclose(f);
    return mel_data;
}

std::vector<int16_t> read_samples_from_file(const char* filename) {
    FILE * f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open test audio file\n");
        return {};
    }
    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    size_t n_samples = file_size / sizeof(int16_t);
    std::vector<int16_t> audio_data(n_samples);
    size_t bread = fread(audio_data.data(), sizeof(int16_t), n_samples, f);
    assert(bread == n_samples);
    fclose(f);
    return audio_data;
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    struct nemo_context *ctx = nemo_init("weights/model.gguf");
    if (!ctx) {
        fprintf(stderr, "Failed to initialize nemo context\n");
        return 1;
    }
    struct nemo_preprocessor* pp = ctx->preprocessor;
    if (!pp) {
        fprintf(stderr, "Preprocessor not initialized in model\n");
        nemo_free(ctx);
        return 1;
    }
    std::vector<int16_t> audio_data = read_samples_from_file("test_audio.pcm");
    size_t target_size = audio_data.size() + 100 + 160;
    size_t stride_padding = target_size % 160;
    if (stride_padding != 0) {
        target_size += 160 - stride_padding;
    }
    audio_data.resize(target_size, 0);
    printf("Audio data size: %zu samples\n", audio_data.size());
    std::vector<float> mel_data = read_sample_mel("my_bin/nemo_mel_data.bin");
    std::vector<float> mel_out;
    for (size_t i = 0; i < audio_data.size(); i += 180) {
        std::vector<float> chunk_mel;
        size_t size = std::min((size_t)180, audio_data.size() - i);
        nemo_preprocessor_process(pp, audio_data.data() + i, size, chunk_mel);
        mel_out.insert(mel_out.end(), chunk_mel.begin(), chunk_mel.end());
    }
    // nemo_preprocessor_process(pp, audio_data.data(), audio_data.size(), mel_out);

    if (mel_out.size() != mel_data.size()) {
        fprintf(stderr, "Mel output size mismatch: got %zu, expected %zu\n",
                mel_out.size(), mel_data.size());
        nemo_free(ctx);
        return 1;
    }
    struct error_calc err_calc;
    err_calc.add_array(mel_out.data(), mel_data.data(), mel_out.size() -128*5);
    for (size_t ii = 0; ii < 10; ii++) {
        printf("mel_out[%zu] = %.6f, mel_data[%zu] = %.6f\n",
               ii, mel_out[ii], ii, mel_data[ii]);
    }
    for (size_t ii = mel_out.size() - 128*5; ii < mel_out.size(); ii++) {
        printf("mel_out[%zu] = %.6f, mel_data[%zu] = %.6f\n",
               ii, mel_out[ii], ii, mel_data[ii]);
    }
    err_calc.report("Mel spectrogram");
    nemo_free(ctx);
    return 0;
}