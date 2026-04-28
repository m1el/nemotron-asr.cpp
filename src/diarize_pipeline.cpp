#include "diarize_pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

namespace {
constexpr int SR             = 16000;
constexpr int VAD_WIN        = 10080;   // 0.63 s
constexpr int VAD_SHIFT      = 160;     // 0.01 s
constexpr int SPK_WIN        = 24000;   // 1.5 s
}

struct diarize_pipeline {
    diarize_pipeline_cfg cfg;

    // Owned by the pipeline.
    diarize_model    model;
    vad_weights      vw;
    spk_weights      sw;
    vad_session    * vs = nullptr;
    spk_session    * ss = nullptr;

    // Audio rolling buffer. audio_buf[k] corresponds to absolute sample
    // index `audio_dropped_samples + k`.
    std::vector<float> audio_buf;
    int64_t audio_dropped_samples = 0;
    int64_t audio_total_samples   = 0;

    // VAD frame stream.
    int64_t              vad_next_frame = 0;   // index of next VAD frame to compute
    std::vector<float>   vad_probs;            // length = vad_next_frame at all times

    // Speech segment state machine.
    bool    in_speech         = false;
    int     n_off_run         = 0;        // consecutive frames below offset (when in speech)
    int     pre_pad_off_run   = 0;        // for tail trim
    int64_t open_seg_start_frame = -1;    // VAD frame index where speech opened
    int     open_seg_id          = -1;
    int     next_seg_id          = 0;
    int64_t open_seg_next_subseg_idx = 0; // sub-seg cursor inside current open segment

    // Closed speech segments (for RTTM later).
    struct seg_record {
        int     seg_id;
        float   start_sec;
        float   end_sec;
    };
    std::vector<seg_record> segments;

    // Embedding store.
    struct sub_record {
        int                seg_id;
        float              start_sec;
        float              end_sec;
        std::vector<float> emb;            // 192-d
    };
    std::vector<sub_record> subs;

    // Word emissions.
    struct word_record {
        std::string text;
        double      at_sec;
        int         speaker = -1;
    };
    std::vector<word_record> words;
    size_t json_drained_words = 0;

    // Smoothed onset/offset thresholds (NeMo-style asymmetric).
    float thr_onset = 0.9f;
    float thr_offset = 0.5f;
    int   min_off_frames = 60;   // 0.6 s of silence to close a segment
};

// ---------------------------------------------------------------------------
// Init / free
// ---------------------------------------------------------------------------

diarize_pipeline * diarize_pipeline_init(const diarize_pipeline_cfg & cfg) {
    auto * p = new diarize_pipeline;
    p->cfg = cfg;
    p->thr_onset       = cfg.vad_post.onset;
    p->thr_offset      = cfg.vad_post.offset;
    p->min_off_frames  = (int)std::ceil(cfg.vad_post.min_duration_off / cfg.vad_post.frame_period_sec);
    if (!diarize_model_load(cfg.diarize_gguf_path, p->model)) {
        fprintf(stderr, "diarize_pipeline: failed to load %s\n", cfg.diarize_gguf_path.c_str());
        delete p;
        return nullptr;
    }
    if (!vad_weights_resolve(p->model, p->model.backend, p->vw)) {
        fprintf(stderr, "diarize_pipeline: vad_weights_resolve failed\n");
        diarize_model_free(p->model);
        delete p;
        return nullptr;
    }
    if (!spk_weights_resolve(p->model, p->model.backend, p->sw)) {
        fprintf(stderr, "diarize_pipeline: spk_weights_resolve failed\n");
        vad_weights_free(p->vw);
        diarize_model_free(p->model);
        delete p;
        return nullptr;
    }
    p->vs = vad_session_init(p->model, p->vw);
    p->ss = spk_session_init(p->model, p->sw);
    return p;
}

void diarize_pipeline_free(diarize_pipeline * p) {
    if (!p) return;
    if (p->ss) spk_session_free(p->ss);
    if (p->vs) vad_session_free(p->vs);
    spk_weights_free(p->sw);
    vad_weights_free(p->vw);
    diarize_model_free(p->model);
    delete p;
}

// ---------------------------------------------------------------------------
// Audio helpers
// ---------------------------------------------------------------------------

static const float * audio_at(const diarize_pipeline & p, int64_t abs_sample) {
    int64_t k = abs_sample - p.audio_dropped_samples;
    if (k < 0 || k >= (int64_t)p.audio_buf.size()) return nullptr;
    return p.audio_buf.data() + k;
}

static bool audio_has_through(const diarize_pipeline & p, int64_t abs_sample_end) {
    return abs_sample_end <= (int64_t)p.audio_dropped_samples + (int64_t)p.audio_buf.size();
}

// Drop audio strictly before abs_sample.
static void drop_audio_before(diarize_pipeline & p, int64_t abs_sample) {
    if (abs_sample <= p.audio_dropped_samples) return;
    int64_t to_drop = std::min<int64_t>(abs_sample - p.audio_dropped_samples,
                                         (int64_t)p.audio_buf.size());
    if (to_drop <= 0) return;
    p.audio_buf.erase(p.audio_buf.begin(),
                      p.audio_buf.begin() + (size_t)to_drop);
    p.audio_dropped_samples += to_drop;
}

// ---------------------------------------------------------------------------
// State machine: try to advance VAD + sub-segment extraction as far as
// possible given currently-buffered audio.
// ---------------------------------------------------------------------------

namespace {

void close_open_segment(diarize_pipeline & p, int64_t end_frame) {
    if (!p.in_speech) return;
    diarize_pipeline::seg_record r;
    r.seg_id    = p.open_seg_id;
    r.start_sec = (float)p.open_seg_start_frame * 0.01f;
    r.end_sec   = (float)end_frame * 0.01f;
    p.segments.push_back(r);
    p.in_speech = false;
    p.open_seg_id = -1;
    p.open_seg_start_frame = -1;
    p.open_seg_next_subseg_idx = 0;
    p.n_off_run = 0;
}

// Run the speaker session on a 1.5 s window starting at abs_sample.
// `lens_samples` is the number of "real" samples (rest is zero-padded
// inside this function). Stores the embedding in p.subs.
void emit_subseg(diarize_pipeline & p, int64_t abs_sample,
                 int64_t lens_samples) {
    static thread_local std::vector<float> chunk;
    chunk.assign(SPK_WIN, 0.0f);

    const int real = (int)std::min<int64_t>(lens_samples, SPK_WIN);
    const float * src = audio_at(p, abs_sample);
    if (src) std::memcpy(chunk.data(), src, real * sizeof(float));

    diarize_pipeline::sub_record s;
    s.seg_id    = p.open_seg_id;
    s.start_sec = (float)abs_sample / SR;
    s.end_sec   = (float)(abs_sample + real) / SR;
    s.emb.resize(kSpkEmbDim);
    if (!spk_session_run_chunk(p.ss, chunk.data(), real, s.emb.data())) {
        fprintf(stderr, "diarize_pipeline: spk_session_run_chunk failed\n");
        return;
    }
    // L2 normalize the embedding for clustering downstream.
    double n = 0;
    for (float v : s.emb) n += (double)v * v;
    float inv = 1.0f / (std::sqrt((float)n) + 1e-8f);
    for (float & v : s.emb) v *= inv;
    p.subs.push_back(std::move(s));
    p.open_seg_next_subseg_idx++;
}

void try_advance(diarize_pipeline & p) {
    const int sub_shift_samp  = (int)std::lround(p.cfg.sub_shift_sec  * SR);
    const int sub_window_samp = (int)std::lround(p.cfg.sub_window_sec * SR);

    // Advance VAD frames.
    while (true) {
        const int64_t abs_start = p.vad_next_frame * VAD_SHIFT;
        const int64_t abs_end   = abs_start + VAD_WIN;
        if (!audio_has_through(p, abs_end)) break;

        const float * window = audio_at(p, abs_start);
        const float prob = vad_session_run_chunk(p.vs, window, VAD_WIN);
        p.vad_probs.push_back(prob);

        // State machine.
        if (!p.in_speech) {
            if (prob >= p.thr_onset) {
                p.in_speech = true;
                p.open_seg_id = p.next_seg_id++;
                p.open_seg_start_frame = p.vad_next_frame;
                p.open_seg_next_subseg_idx = 0;
                p.n_off_run = 0;
            }
        } else {
            if (prob < p.thr_offset) {
                p.n_off_run++;
                if (p.n_off_run >= p.min_off_frames) {
                    // Close the segment. The segment "ended" min_off_frames ago.
                    int64_t end_frame = p.vad_next_frame + 1 - p.n_off_run;
                    if (end_frame < p.open_seg_start_frame) end_frame = p.open_seg_start_frame;
                    // Maybe emit a tail subseg with the leftover audio inside the segment.
                    int64_t seg_start_sample = p.open_seg_start_frame * VAD_SHIFT;
                    int64_t seg_end_sample   = end_frame * VAD_SHIFT;
                    int64_t covered_end = seg_start_sample +
                        (p.open_seg_next_subseg_idx > 0
                         ? (p.open_seg_next_subseg_idx - 1) * sub_shift_samp + sub_window_samp
                         : 0);
                    int64_t leftover = seg_end_sample - covered_end;
                    if (leftover >= (int64_t)std::lround(p.cfg.min_seg_sec * SR)
                        && p.open_seg_next_subseg_idx > 0) {
                        // emit one trailing subseg with masked padding
                        int64_t tail_start = covered_end;
                        emit_subseg(p, tail_start, leftover);
                    } else if (p.open_seg_next_subseg_idx == 0
                               && (seg_end_sample - seg_start_sample)
                                  >= (int64_t)std::lround(p.cfg.min_seg_sec * SR)) {
                        // Segment was shorter than 1.5 s: emit one masked-pad subseg.
                        emit_subseg(p, seg_start_sample,
                                    seg_end_sample - seg_start_sample);
                    }
                    close_open_segment(p, end_frame);
                }
            } else {
                p.n_off_run = 0;
            }
        }

        p.vad_next_frame++;

        // While in speech, try to emit any sub-segment whose audio is now available.
        if (p.in_speech) {
            int64_t seg_start_sample = p.open_seg_start_frame * VAD_SHIFT;
            while (true) {
                int64_t k = p.open_seg_next_subseg_idx;
                int64_t s_start = seg_start_sample + k * sub_shift_samp;
                int64_t s_end   = s_start + sub_window_samp;
                if (!audio_has_through(p, s_end)) break;
                emit_subseg(p, s_start, sub_window_samp);
            }
        }
    }

    // Drop audio that is no longer needed.
    int64_t drop_to = p.vad_next_frame * VAD_SHIFT;
    if (p.in_speech) {
        // We still need samples for the next sub-segment of the open segment.
        int64_t seg_start_sample = p.open_seg_start_frame * VAD_SHIFT;
        int64_t next_subseg_start = seg_start_sample +
            p.open_seg_next_subseg_idx * sub_shift_samp;
        drop_to = std::min(drop_to, next_subseg_start);
    }
    drop_audio_before(p, drop_to);
}

} // namespace

// ---------------------------------------------------------------------------
// Public push / drain / finalize
// ---------------------------------------------------------------------------

size_t diarize_pipeline_push_audio(diarize_pipeline * p, const float * audio, size_t n) {
    if (!p || !audio || n == 0) return 0;
    p->audio_buf.insert(p->audio_buf.end(), audio, audio + n);
    p->audio_total_samples += (int64_t)n;
    const size_t before = p->vad_probs.size();
    try_advance(*p);
    return p->vad_probs.size() - before;
}

void diarize_pipeline_push_text(diarize_pipeline * p, const std::string & text,
                                double at_sec) {
    if (!p) return;
    // Split on whitespace; record each whitespace-separated token as a word.
    std::istringstream iss(text);
    std::string tok;
    while (iss >> tok) {
        diarize_pipeline::word_record w;
        w.text   = std::move(tok);
        w.at_sec = at_sec;
        p->words.push_back(std::move(w));
    }
}

std::string diarize_pipeline_drain_json(diarize_pipeline * p) {
    if (!p) return "";
    std::string out;
    for (size_t i = p->json_drained_words; i < p->words.size(); i++) {
        const auto & w = p->words[i];
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "{\"word\":\"%s\",\"at\":%.3f}\n", w.text.c_str(), w.at_sec);
        out += buf;
    }
    p->json_drained_words = p->words.size();
    return out;
}

// At finalize: process any remaining audio (already done during push_audio
// for whole windows; flush a trailing partial speech segment if open).
static void finalize_open_segment(diarize_pipeline & p) {
    const int sub_shift_samp  = (int)std::lround(p.cfg.sub_shift_sec  * SR);
    const int sub_window_samp = (int)std::lround(p.cfg.sub_window_sec * SR);

    if (p.in_speech) {
        // Treat end of input as end of speech.
        int64_t end_frame = p.vad_next_frame; // last frame we've seen
        int64_t seg_start_sample = p.open_seg_start_frame * VAD_SHIFT;
        int64_t seg_end_sample   = std::min<int64_t>(end_frame * VAD_SHIFT,
                                                     p.audio_total_samples);
        int64_t covered_end = seg_start_sample +
            (p.open_seg_next_subseg_idx > 0
             ? (p.open_seg_next_subseg_idx - 1) * sub_shift_samp + sub_window_samp
             : 0);
        int64_t leftover = seg_end_sample - covered_end;
        if (leftover >= (int64_t)std::lround(p.cfg.min_seg_sec * SR)) {
            emit_subseg(p, covered_end, leftover);
        } else if (p.open_seg_next_subseg_idx == 0
                   && (seg_end_sample - seg_start_sample)
                      >= (int64_t)std::lround(p.cfg.min_seg_sec * SR)) {
            emit_subseg(p, seg_start_sample, seg_end_sample - seg_start_sample);
        }
        close_open_segment(p, end_frame);
    }
}

// Build a per-frame speaker timeline from labelled sub-segments.
//  For each frame (10ms), the speaker is taken from the most recent
//  enclosing sub-segment, with overlapping sub-segments resolved by
//  picking the one whose center is closest to that frame.
struct speaker_span { float start_sec; float end_sec; int speaker; };

static std::vector<speaker_span> build_speaker_timeline(
    const std::vector<diarize_pipeline::sub_record> & subs,
    const std::vector<int> & labels)
{
    std::vector<speaker_span> spans;
    if (subs.empty()) return spans;
    // For each sub-segment, emit a span. Then merge contiguous identical-
    // speaker spans. Where sub-segments overlap, the later one's start wins
    // for the assignment of the overlap region — simple but works for our
    // 0.75 s-shift / 1.5 s-window setup.
    std::vector<speaker_span> raw;
    raw.reserve(subs.size());
    for (size_t i = 0; i < subs.size(); i++) {
        speaker_span s;
        s.start_sec = subs[i].start_sec;
        s.end_sec   = subs[i].end_sec;
        s.speaker   = labels[i];
        raw.push_back(s);
    }
    // Sort by start.
    std::sort(raw.begin(), raw.end(),
              [](const speaker_span & a, const speaker_span & b) {
                  return a.start_sec < b.start_sec;
              });
    // Walk and resolve overlaps: at each time step, the active speaker is the
    // last sub-segment that started before this time and hasn't ended.
    // Easier: build a sorted event list of (time, +1 for span open, -1 close, idx).
    // For our use we collapse to non-overlapping by trimming each span's start
    // to max(start, prev_end_for_other_speaker).
    std::vector<speaker_span> merged;
    for (size_t i = 0; i < raw.size(); i++) {
        speaker_span s = raw[i];
        if (!merged.empty()) {
            // If overlap with previous and same speaker: extend.
            if (merged.back().speaker == s.speaker
                && s.start_sec <= merged.back().end_sec + 1e-3f) {
                merged.back().end_sec = std::max(merged.back().end_sec, s.end_sec);
                continue;
            }
            // If overlap with previous but different speaker: split at midpoint.
            if (s.start_sec < merged.back().end_sec) {
                float mid = 0.5f * (s.start_sec + merged.back().end_sec);
                merged.back().end_sec = mid;
                s.start_sec = mid;
            }
        }
        merged.push_back(s);
    }
    return merged;
}

static int speaker_at(const std::vector<speaker_span> & timeline, double t) {
    if (timeline.empty()) return -1;
    // Binary search for the last span whose start_sec <= t.
    int lo = 0, hi = (int)timeline.size() - 1, best = -1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (timeline[mid].start_sec <= t) { best = mid; lo = mid + 1; }
        else                              { hi = mid - 1; }
    }
    if (best < 0) return -1;
    if (t > timeline[best].end_sec) return -1;
    return timeline[best].speaker;
}

std::string diarize_pipeline_finalize(diarize_pipeline * p) {
    if (!p) return "";
    finalize_open_segment(*p);

    if (p->subs.empty()) return "";

    // Pack embeddings.
    std::vector<float> all_embs;
    all_embs.reserve(p->subs.size() * (size_t)kSpkEmbDim);
    for (auto & s : p->subs) all_embs.insert(all_embs.end(), s.emb.begin(), s.emb.end());

    auto cl = nmesc_cluster(all_embs.data(),
                            p->subs.size(), (size_t)kSpkEmbDim,
                            p->cfg.cluster);

    auto timeline = build_speaker_timeline(p->subs, cl.labels);

    // Assign speakers to words.
    for (auto & w : p->words) {
        w.speaker = speaker_at(timeline, w.at_sec);
    }

    // Build speaker-tagged transcript.
    std::ostringstream sst;
    int last_speaker = -2;
    for (size_t i = 0; i < p->words.size(); i++) {
        const auto & w = p->words[i];
        if (w.speaker != last_speaker) {
            if (last_speaker != -2) sst << "\n";
            sst << "[spk_" << (w.speaker < 0 ? -1 : w.speaker) << "] ";
            last_speaker = w.speaker;
        }
        sst << w.text << " ";
    }
    if (!p->words.empty()) sst << "\n";
    std::string out = sst.str();

    if (!p->cfg.speaker_text_path.empty() && p->cfg.speaker_text_path != "-") {
        std::ofstream f(p->cfg.speaker_text_path);
        f << out;
    }

    if (!p->cfg.rttm_path.empty()) {
        std::ofstream f(p->cfg.rttm_path);
        // RTTM line format used by NeMo:
        // SPEAKER <uri> <chan> <start> <dur> <NA> <NA> <spk_label> <NA> <NA>
        for (auto & sp : timeline) {
            if (sp.speaker < 0) continue;
            f << "SPEAKER session 1 "
              << sp.start_sec << " " << (sp.end_sec - sp.start_sec)
              << " <NA> <NA> spk_" << sp.speaker
              << " <NA> <NA>\n";
        }
    }

    return out;
}

size_t diarize_pipeline_n_embeddings(const diarize_pipeline * p) { return p ? p->subs.size() : 0; }
size_t diarize_pipeline_n_segments(const diarize_pipeline * p)   { return p ? p->segments.size() : 0; }
size_t diarize_pipeline_n_words(const diarize_pipeline * p)      { return p ? p->words.size() : 0; }
