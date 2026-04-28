// Streaming transcription with optional speaker diarization.
//
// Usage:
//   ./nemotron-asr.cpp <model.gguf> <audio.pcm|-> [chunk_ms] [right_context]
//                      [--cpu|--cuda|--metal]
//                      [--diarize <diarize.gguf>]
//                      [--rttm <out.rttm>]
//                      [--speaker-text <out.txt>]
//                      [--json <out.jsonl>]
//                      [--num-speakers <K>]
//                      [--sub-shift <sec>]
//
// Input audio is raw PCM int16 little-endian, 16 kHz, mono.

#include "nemo-ggml.h"
#include "nemo-stream.h"
#include "diarize_pipeline.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

static void print_usage(const char * prog) {
    fprintf(stderr,
        "Usage: %s <model.gguf> <audio.pcm|-> [chunk_ms] [right_context]\n"
        "       [--cpu|--cuda|--metal]\n"
        "       [--diarize <diarize.gguf>]\n"
        "       [--rttm <path>] [--speaker-text <path>] [--json <path>]\n"
        "       [--num-speakers <K>] [--sub-shift <sec>]\n"
        "\n"
        "  model.gguf       ASR weights\n"
        "  audio.pcm        PCM int16le 16 kHz mono, or '-' for stdin\n"
        "  chunk_ms         streaming chunk size (default 80)\n"
        "  right_context    attention right context (0/1/6/13, default 0)\n"
        "  --cpu|--cuda|--metal  backend selection\n"
        "  --diarize <file> enable diarization with the given diarize.gguf\n"
        "  --rttm <file>    write RTTM output (only with --diarize)\n"
        "  --speaker-text <file>  write speaker-tagged transcript at EOF\n"
        "                         (default: stdout when --diarize is set)\n"
        "  --json <file>    write per-word JSON lines as they are emitted\n"
        "  --num-speakers K force K speakers (overrides NME estimate)\n"
        "  --sub-shift sec  sub-segment shift in seconds (default 0.75)\n"
        "\n"
        "Examples:\n"
        "  %s weights/model.gguf audio.pcm 80 0\n"
        "  %s weights/model.gguf audio.pcm 80 0 --cuda\n"
        "  %s weights/model.gguf audio.pcm 80 0 \\\n"
        "       --diarize weights/diarize.gguf --num-speakers 2\n"
        "  ffmpeg -i a.mp3 -f s16le -ar 16000 -ac 1 - | \\\n"
        "       %s weights/model.gguf - 80 0 --diarize weights/diarize.gguf\n",
        prog, prog, prog, prog, prog);
}

int main(int argc, char ** argv) {
    if (argc < 3) { print_usage(argv[0]); return 1; }

    const char * model_path = argv[1];
    const char * audio_path = argv[2];
    int chunk_ms = 80;
    int right_context = 0;
    nemo_backend_type backend = NEMO_BACKEND_AUTO;

    std::string diarize_gguf, rttm_path, speaker_text_path, json_path;
    int   num_speakers   = -1;
    float sub_shift_sec  = 0.75f;
    bool  read_from_stdin =
        (strcmp(audio_path, "-") == 0 || strcmp(audio_path, "--stdin") == 0);

    // Parse positional + flag arguments.
    int positional_idx = 0;
    for (int i = 3; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--cpu")        backend = NEMO_BACKEND_CPU;
        else if (a == "--cuda")  backend = NEMO_BACKEND_CUDA;
        else if (a == "--metal") backend = NEMO_BACKEND_METAL;
        else if (a == "--diarize"      && i + 1 < argc) diarize_gguf      = argv[++i];
        else if (a == "--rttm"         && i + 1 < argc) rttm_path         = argv[++i];
        else if (a == "--speaker-text" && i + 1 < argc) speaker_text_path = argv[++i];
        else if (a == "--json"         && i + 1 < argc) json_path         = argv[++i];
        else if (a == "--num-speakers" && i + 1 < argc) num_speakers = atoi(argv[++i]);
        else if (a == "--sub-shift"    && i + 1 < argc) sub_shift_sec = (float)atof(argv[++i]);
        else if (!a.empty() && a[0] == '-') {
            fprintf(stderr, "Unknown flag: %s\n", a.c_str());
            return 1;
        } else {
            // Positional after model+audio: chunk_ms then right_context.
            if (positional_idx == 0) chunk_ms = atoi(argv[i]);
            else if (positional_idx == 1) right_context = atoi(argv[i]);
            positional_idx++;
        }
    }

    if (chunk_ms < 10) {
        fprintf(stderr, "chunk_ms must be >= 10 (got %d)\n", chunk_ms);
        return 1;
    }

    fprintf(stderr, "Configuration:\n");
    fprintf(stderr, "  Model:          %s\n", model_path);
    fprintf(stderr, "  Audio:          %s\n", read_from_stdin ? "stdin" : audio_path);
    fprintf(stderr, "  Chunk size:     %d ms\n", chunk_ms);
    fprintf(stderr, "  Right context:  %d\n", right_context);
    if (!diarize_gguf.empty()) {
        fprintf(stderr, "  Diarization:    %s\n", diarize_gguf.c_str());
        if (num_speakers > 0)
            fprintf(stderr, "  Num speakers:   %d (forced)\n", num_speakers);
        fprintf(stderr, "  Sub-shift:      %.2f s\n", sub_shift_sec);
    }
    fprintf(stderr, "\n");

    nemo_context * ctx = nemo_init_with_backend(model_path, backend);
    if (!ctx) { fprintf(stderr, "Failed to load ASR model\n"); return 1; }

    nemo_cache_config cache_cfg = nemo_cache_config::default_config();
    cache_cfg.att_right_context = right_context;
    nemo_stream_context * sctx = nemo_stream_init(ctx, &cache_cfg);
    if (!sctx) { fprintf(stderr, "Failed to create streaming context\n"); nemo_free(ctx); return 1; }

    int computed_chunk_samples = cache_cfg.get_chunk_samples();

    // Diarization pipeline (optional).
    diarize_pipeline * dp = nullptr;
    if (!diarize_gguf.empty()) {
        diarize_pipeline_cfg dcfg = diarize_pipeline_default_cfg();
        dcfg.diarize_gguf_path = diarize_gguf;
        dcfg.sub_shift_sec     = sub_shift_sec;
        dcfg.cluster.oracle_num_speakers = num_speakers;
        dcfg.cluster.min_samples_for_nmesc = 4;
        dcfg.rttm_path         = rttm_path;
        dcfg.speaker_text_path = speaker_text_path.empty() && !diarize_gguf.empty()
            ? "-"
            : speaker_text_path;
        // Mirror ASR's backend choice: CUDA/Metal if requested or available.
        switch (backend) {
            case NEMO_BACKEND_CPU:   dcfg.backend = diarize_backend::CPU;   break;
            case NEMO_BACKEND_CUDA:  dcfg.backend = diarize_backend::CUDA;  break;
            case NEMO_BACKEND_METAL: dcfg.backend = diarize_backend::METAL; break;
            default:                 dcfg.backend = diarize_backend::AUTO;  break;
        }
        dp = diarize_pipeline_init(dcfg);
        if (!dp) {
            fprintf(stderr, "Failed to init diarization pipeline\n");
            nemo_stream_free(sctx);
            nemo_free(ctx);
            return 1;
        }
    }

    // Optional JSON sink for per-word output.
    std::ofstream json_file;
    bool json_to_stdout = false;
    if (!json_path.empty()) {
        if (json_path == "-") json_to_stdout = true;
        else                  json_file.open(json_path);
    }

    // Open audio source.
    FILE * input = nullptr;
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
            if (dp) diarize_pipeline_free(dp);
            nemo_stream_free(sctx);
            nemo_free(ctx);
            return 1;
        }
        fprintf(stderr, "Streaming from file...\n\n");
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<int16_t> i16(computed_chunk_samples);
    std::vector<float>   f32; // reused, sized as needed
    size_t total_samples = 0;

    // True iff `text` is incremental (new since last call). Finalize returns
    // the full cumulative transcript, which we still want to print but must
    // NOT push to the diarization pipeline (would duplicate every word).
    auto handle_text = [&](const std::string & text, size_t total_samples_now,
                           bool is_incremental) {
        if (text.empty()) return;
        printf("%s", text.c_str());
        fflush(stdout);

        if (dp && is_incremental) {
            const double at_sec = (double)total_samples_now / 16000.0;
            diarize_pipeline_push_text(dp, text, at_sec);
            if (json_file.is_open() || json_to_stdout) {
                std::string j = diarize_pipeline_drain_json(dp);
                if (!j.empty()) {
                    if (json_to_stdout) { fputs(j.c_str(), stdout); fflush(stdout); }
                    else                { json_file << j; json_file.flush(); }
                }
            }
        }
    };

    while (true) {
        const size_t want_bytes = (size_t)computed_chunk_samples * sizeof(int16_t);
        const size_t got_bytes  = fread(i16.data(), 1, want_bytes, input);
        if (got_bytes == 0) break;

        const int n = (int)(got_bytes / sizeof(int16_t));
        total_samples += (size_t)n;

        // ASR.
        std::string text = nemo_stream_process_incremental(sctx, i16.data(), n);
        handle_text(text, total_samples, /*is_incremental=*/true);

        // Diarization (audio).
        if (dp) {
            f32.resize((size_t)n);
            for (int k = 0; k < n; k++) f32[(size_t)k] = i16[(size_t)k] / 32768.0f;
            diarize_pipeline_push_audio(dp, f32.data(), (size_t)n);
        }

        if (got_bytes < want_bytes) break;
    }

    // Flush ASR. nemo_stream_finalize returns the cumulative transcript,
    // not the trailing increment, so we ignore it for diarization to avoid
    // duplicating words. Replace the live-streaming line with an explicit
    // newline so subsequent output starts on its own line.
    (void)nemo_stream_finalize(sctx);
    printf("\n");

    if (!read_from_stdin && input) fclose(input);

    auto end_time = std::chrono::high_resolution_clock::now();
    const double processing_sec = std::chrono::duration<double>(end_time - start_time).count();
    const float  audio_sec      = (float)total_samples / 16000.0f;
    fprintf(stderr, "\n=== Complete ===\n");
    fprintf(stderr, "  Audio duration:   %.2f s\n", audio_sec);
    fprintf(stderr, "  Processing time:  %.2f s\n", processing_sec);
    if (audio_sec > 0)
        fprintf(stderr, "  Real-time factor: %.3fx\n", processing_sec / audio_sec);

    // Diarization finalize.
    if (dp) {
        fprintf(stderr, "\nFinalizing diarization (%zu sub-segments, %zu words)...\n",
                diarize_pipeline_n_embeddings(dp),
                diarize_pipeline_n_words(dp));
        std::string spk_text = diarize_pipeline_finalize(dp);
        // Drain any remaining JSON.
        if (json_file.is_open() || json_to_stdout) {
            std::string j = diarize_pipeline_drain_json(dp);
            if (!j.empty()) {
                if (json_to_stdout) fputs(j.c_str(), stdout);
                else                json_file << j;
            }
        }
        // If the user didn't redirect to a file, print speaker-tagged transcript.
        if (speaker_text_path.empty() || speaker_text_path == "-") {
            fprintf(stderr, "\n=== Speaker-tagged transcript ===\n");
            fputs(spk_text.c_str(), stdout);
        }
        if (!rttm_path.empty()) {
            fprintf(stderr, "Wrote RTTM: %s\n", rttm_path.c_str());
        }
        diarize_pipeline_free(dp);
    }

    nemo_stream_free(sctx);
    nemo_free(ctx);
    return 0;
}
