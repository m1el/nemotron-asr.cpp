// Streaming diarization pipeline that hangs off the existing ASR binary.
//
// Architecture:
//   - Audio is pushed in chunks; the pipeline drives a VAD frame stream and
//     opens / closes speech segments as VAD probabilities cross thresholds.
//   - When a speech segment has >= 1.5 s of buffered audio, a sub-segment is
//     extracted, embedded with TitaNet, and the audio for it is dropped.
//   - Word emissions from the ASR are tagged with the current audio time
//     and accumulated.
//   - On finalize: cluster all embeddings, align speakers with words via
//     timestamp overlap, emit RTTM and/or speaker-tagged transcript.
//
// The pipeline owns its own diarize_model + vad/spk sessions; the ASR side
// is unaffected.

#ifndef NEMOTRON_DIARIZE_PIPELINE_H
#define NEMOTRON_DIARIZE_PIPELINE_H

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "diarize.h"
#include "diarize_vad.h"
#include "diarize_spk.h"
#include "diarize_cluster.h"

struct diarize_pipeline_cfg {
    std::string diarize_gguf_path;

    // Sub-segment cadence.
    float sub_window_sec = 1.5f;
    float sub_shift_sec  = 0.75f;
    float min_seg_sec    = 0.5f;   // skip speech regions shorter than this

    // VAD thresholding (matches diar_infer_meeting.yaml defaults).
    vad_post_cfg vad_post;

    // Clustering knobs.
    nmesc_cfg cluster;

    // Output paths (empty = don't emit). speaker_text_path == "-" → stdout.
    std::string rttm_path;
    std::string speaker_text_path;
    std::string json_path;     // per-word JSON during streaming (empty = off)
};

inline diarize_pipeline_cfg diarize_pipeline_default_cfg() {
    diarize_pipeline_cfg c;
    c.vad_post.onset            = 0.9f;
    c.vad_post.offset           = 0.5f;
    c.vad_post.min_duration_on  = 0.0f;
    c.vad_post.min_duration_off = 0.6f;
    c.vad_post.pad_onset        = 0.0f;
    c.vad_post.pad_offset       = 0.0f;
    c.vad_post.frame_period_sec = 0.01f;
    return c;
}

struct diarize_pipeline;

diarize_pipeline * diarize_pipeline_init(const diarize_pipeline_cfg & cfg);
void diarize_pipeline_free(diarize_pipeline * p);

// Push audio (16 kHz, float32 mono in [-1, 1]). Returns the number of new VAD
// probability frames produced.
size_t diarize_pipeline_push_audio(diarize_pipeline * p, const float * audio, size_t n);

// Push a text emission produced by the ASR side at the given audio time.
// `text` may contain multiple whitespace-separated words; each is recorded
// with the same time.
void diarize_pipeline_push_text(diarize_pipeline * p, const std::string & text, double at_sec);

// Pull JSON for any words emitted since the last call (one JSON object per
// line).  Returns the new lines as a single string (empty if none).
std::string diarize_pipeline_drain_json(diarize_pipeline * p);

// Finalize at EOF: close any open speech segment, run clustering, write
// configured outputs (RTTM, speaker-tagged transcript). Returns the
// speaker-tagged transcript (also written to `speaker_text_path` if set).
std::string diarize_pipeline_finalize(diarize_pipeline * p);

// Introspection.
size_t diarize_pipeline_n_embeddings(const diarize_pipeline * p);
size_t diarize_pipeline_n_segments(const diarize_pipeline * p);
size_t diarize_pipeline_n_words(const diarize_pipeline * p);

#endif // NEMOTRON_DIARIZE_PIPELINE_H
