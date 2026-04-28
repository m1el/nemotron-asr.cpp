#include "diarize_vad.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-cpu.h"

// MarbleNet block topology (multilingual variant).
namespace {
struct block_topology {
    int kernel;
    int dilation;
    int repeat;
    int in_ch;
    int out_ch;
    bool residual;
    bool separable;
};

constexpr block_topology kBlocks[6] = {
    {11, 1, 1,  80, 128, false, true },
    {13, 1, 2, 128,  64, true,  true },
    {15, 1, 2,  64,  64, true,  true },
    {17, 1, 2,  64,  64, true,  true },
    {29, 2, 1,  64, 128, false, true },
    { 1, 1, 1, 128, 128, false, false},
};

// NeMo's Jasper modules build their BN as nn.BatchNorm1d(C, eps=1e-3) (NOT
// PyTorch's default 1e-5). See jasper.py:_get_conv_bn_layer.
constexpr float kBNEps = 1e-3f;

ggml_tensor * find(const diarize_model & m, const std::string & name) {
    auto it = m.tensors.find(name);
    if (it == m.tensors.end()) {
        fprintf(stderr, "vad_weights: missing tensor '%s'\n", name.c_str());
        return nullptr;
    }
    return it->second;
}

// Read a CPU-resident float tensor into a host buffer.
std::vector<float> read_tensor(const ggml_tensor * t) {
    const int64_t n = (int64_t)ggml_nelements(t);
    std::vector<float> out(n);
    ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    return out;
}

// Fold one BN into (scale, bias) per channel:
//   scale[c] = gamma[c] / sqrt(var[c] + eps)
//   bias[c]  = beta[c]  - mean[c] * scale[c]
struct bn_fold {
    std::vector<float> scale;
    std::vector<float> bias;
};

bn_fold fold_bn(const ggml_tensor * gamma, const ggml_tensor * beta,
                const ggml_tensor * mean, const ggml_tensor * var) {
    auto g = read_tensor(gamma);
    auto b = read_tensor(beta);
    auto m = read_tensor(mean);
    auto v = read_tensor(var);
    const size_t C = g.size();
    bn_fold f;
    f.scale.resize(C);
    f.bias.resize(C);
    for (size_t i = 0; i < C; i++) {
        float s = g[i] / std::sqrt(v[i] + kBNEps);
        f.scale[i] = s;
        f.bias[i]  = b[i] - m[i] * s;
    }
    return f;
}

} // namespace

// ---------------------------------------------------------------------------
// Resolve
// ---------------------------------------------------------------------------

namespace {

// Resolve one sub-conv. For separable=true we expect dw_idx, pw_idx, bn_idx
// inside `prefix.mconv.<idx>`. For separable=false (block 5) the single conv
// is at mconv.0 and BN at mconv.1.
struct fold_alloc {
    // Pending BN folds: (scale_vec, bias_vec, C, ptr-to-write-scale, ptr-to-write-bias)
    struct entry {
        std::vector<float> scale;
        std::vector<float> bias;
        ggml_tensor ** scale_dst;
        ggml_tensor ** bias_dst;
    };
    std::vector<entry> pending;
};

bool resolve_subconv(const diarize_model & m, const std::string & prefix,
                     int dw_idx, int pw_idx, int bn_idx,
                     bool separable, int kernel, int dilation,
                     vad_subconv & s, fold_alloc & fa) {
    s.separable = separable;
    s.kernel = kernel;
    s.dilation = dilation;

    if (separable) {
        s.dw_w = find(m, prefix + ".mconv." + std::to_string(dw_idx) + ".conv.weight");
        if (!s.dw_w) return false;
    }
    s.pw_w = find(m, prefix + ".mconv." + std::to_string(pw_idx) + ".conv.weight");
    if (!s.pw_w) return false;

    auto * gamma = find(m, prefix + ".mconv." + std::to_string(bn_idx) + ".weight");
    auto * beta  = find(m, prefix + ".mconv." + std::to_string(bn_idx) + ".bias");
    auto * mean  = find(m, prefix + ".mconv." + std::to_string(bn_idx) + ".running_mean");
    auto * var   = find(m, prefix + ".mconv." + std::to_string(bn_idx) + ".running_var");
    if (!gamma || !beta || !mean || !var) return false;

    auto f = fold_bn(gamma, beta, mean, var);
    fa.pending.push_back({std::move(f.scale), std::move(f.bias), &s.bn_scale, &s.bn_bias});
    return true;
}

bool resolve_residual(const diarize_model & m, const std::string & prefix,
                      vad_subconv & s, fold_alloc & fa) {
    s.separable = false;
    s.kernel = 1;
    s.dilation = 1;
    std::string rp = prefix + ".res.0";
    s.pw_w = find(m, rp + ".0.conv.weight");
    auto * gamma = find(m, rp + ".1.weight");
    auto * beta  = find(m, rp + ".1.bias");
    auto * mean  = find(m, rp + ".1.running_mean");
    auto * var   = find(m, rp + ".1.running_var");
    if (!s.pw_w || !gamma || !beta || !mean || !var) return false;
    auto f = fold_bn(gamma, beta, mean, var);
    fa.pending.push_back({std::move(f.scale), std::move(f.bias), &s.bn_scale, &s.bn_bias});
    return true;
}

} // namespace

bool vad_weights_resolve(const diarize_model & m, ggml_backend_t backend,
                         vad_weights & w) {
    fold_alloc fa;
    w.blocks.resize(6);

    for (int b = 0; b < 6; b++) {
        const block_topology & t = kBlocks[b];
        std::string prefix = "vad.encoder.encoder." + std::to_string(b);
        vad_block & blk = w.blocks[b];
        blk.subs.resize(t.repeat);
        blk.residual = t.residual;

        for (int s = 0; s < t.repeat; s++) {
            int base = 5 * s;
            int dw_idx = base, pw_idx = base + 1, bn_idx = base + 2;
            if (!t.separable) { dw_idx = -1; pw_idx = 0; bn_idx = 1; }
            if (!resolve_subconv(m, prefix, dw_idx, pw_idx, bn_idx,
                                  t.separable, t.kernel, t.dilation,
                                  blk.subs[s], fa)) return false;
        }

        if (t.residual) {
            if (!resolve_residual(m, prefix, blk.res, fa)) return false;
        }
    }

    w.dec_w = find(m, "vad.decoder.decoder_layers.0.weight");
    w.dec_b = find(m, "vad.decoder.decoder_layers.0.bias");
    if (!w.dec_w || !w.dec_b) return false;
    w.n_classes = (int)w.dec_w->ne[1];
    w.enc_out_channels = (int)w.dec_w->ne[0];

    // Allocate fold tensors in a fresh context, then upload data.
    ggml_init_params p = {
        .mem_size   = ggml_tensor_overhead() * (fa.pending.size() * 2 + 16),
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    w.fold_ctx = ggml_init(p);
    if (!w.fold_ctx) { fprintf(stderr, "vad_weights: fold_ctx init failed\n"); return false; }

    for (auto & e : fa.pending) {
        const int64_t C = (int64_t)e.scale.size();
        ggml_tensor * scale = ggml_new_tensor_1d(w.fold_ctx, GGML_TYPE_F32, C);
        ggml_tensor * bias  = ggml_new_tensor_1d(w.fold_ctx, GGML_TYPE_F32, C);
        ggml_set_name(scale, "bn_scale");
        ggml_set_name(bias,  "bn_bias");
        *e.scale_dst = scale;
        *e.bias_dst  = bias;
    }

    w.fold_buf = ggml_backend_alloc_ctx_tensors(w.fold_ctx, backend);
    if (!w.fold_buf) { fprintf(stderr, "vad_weights: fold buffer alloc failed\n"); return false; }

    for (auto & e : fa.pending) {
        ggml_tensor * scale_t = *e.scale_dst;
        ggml_tensor * bias_t  = *e.bias_dst;
        ggml_backend_tensor_set(scale_t, e.scale.data(), 0, e.scale.size() * sizeof(float));
        ggml_backend_tensor_set(bias_t,  e.bias.data(),  0, e.bias.size()  * sizeof(float));
    }

    return true;
}

void vad_weights_free(vad_weights & w) {
    if (w.fold_buf) { ggml_backend_buffer_free(w.fold_buf); w.fold_buf = nullptr; }
    if (w.fold_ctx) { ggml_free(w.fold_ctx); w.fold_ctx = nullptr; }
    w.blocks.clear();
}

// ---------------------------------------------------------------------------
// Graph build
// ---------------------------------------------------------------------------

namespace {

// Pointwise conv1d via matmul: out (C_out, T) = pw_w (C_in, C_out) @ x (C_in, T).
ggml_tensor * pointwise(ggml_context * ctx, ggml_tensor * x, ggml_tensor * pw_w) {
    return ggml_mul_mat(ctx, pw_w, x);
}

// Depthwise conv1d with same padding (and dilation).
ggml_tensor * depthwise_same(ggml_context * ctx, ggml_tensor * x,
                             ggml_tensor * dw_w, int kernel, int dilation) {
    const int64_t C = x->ne[0];
    const int64_t T = x->ne[1];
    const int pad = dilation * (kernel - 1) / 2;
    GGML_ASSERT(dw_w->ne[0] == C);
    GGML_ASSERT(dw_w->ne[1] == kernel);
    ggml_tensor * padded = ggml_pad_ext(ctx, x, 0, 0, pad, pad, 0, 0, 0, 0);
    GGML_ASSERT(padded->ne[1] == T + 2 * pad);

    ggml_tensor * acc = nullptr;
    for (int i = 0; i < kernel; i++) {
        const size_t time_off_bytes = (size_t)(i * dilation) * padded->nb[1];
        ggml_tensor * slice = ggml_view_2d(ctx, padded, C, T, padded->nb[1], time_off_bytes);
        ggml_tensor * kcol_1d = ggml_view_1d(ctx, dw_w, C, (size_t)i * dw_w->nb[1]);
        ggml_tensor * kcol = ggml_reshape_2d(ctx, kcol_1d, C, 1);
        ggml_tensor * prod = ggml_mul(ctx, slice, kcol);
        acc = (acc == nullptr) ? prod : ggml_add(ctx, acc, prod);
    }
    return acc;
}

// Folded BN: y = x * scale + bias  (broadcast scale, bias of shape (C,) over T).
ggml_tensor * apply_bn(ggml_context * ctx, ggml_tensor * x,
                       ggml_tensor * scale, ggml_tensor * bias) {
    const int64_t C = x->ne[0];
    GGML_ASSERT(scale->ne[0] == C);
    GGML_ASSERT(bias->ne[0] == C);
    ggml_tensor * scale2 = ggml_reshape_2d(ctx, scale, C, 1);
    ggml_tensor * bias2  = ggml_reshape_2d(ctx, bias,  C, 1);
    ggml_tensor * y = ggml_mul(ctx, x, scale2);
    y = ggml_add(ctx, y, bias2);
    return y;
}

} // namespace

vad_graph vad_graph_build(const vad_weights & w, int T, int n_mels) {
    vad_graph g;
    g.T = T;
    g.n_mels = n_mels;

    ggml_init_params p = { .mem_size = 2 * 1024 * 1024, .mem_buffer = nullptr, .no_alloc = true };
    g.ctx = ggml_init(p);
    g.graph = ggml_new_graph_custom(g.ctx, /*size=*/8192, /*grads=*/false);

    g.mel_input = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, n_mels, T);
    ggml_set_name(g.mel_input, "mel_input");
    ggml_set_input(g.mel_input);

    // Per-time mask broadcast over channels. Matches NeMo's MaskedConv1d:
    // before every conv, input positions [lens, T) are zeroed out.
    g.mask_input = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, 1, T);
    ggml_set_name(g.mask_input, "mask_input");
    ggml_set_input(g.mask_input);

    auto apply_mask = [&](ggml_tensor * x) -> ggml_tensor * {
        return ggml_mul(g.ctx, x, g.mask_input);
    };

    auto apply_subconv_masked = [&](ggml_tensor * x, const vad_subconv & s) -> ggml_tensor * {
        ggml_tensor * y = x;
        if (s.separable) {
            y = apply_mask(y);
            y = depthwise_same(g.ctx, y, s.dw_w, s.kernel, s.dilation);
        }
        y = apply_mask(y);
        y = pointwise(g.ctx, y, s.pw_w);
        y = apply_bn(g.ctx, y, s.bn_scale, s.bn_bias);
        return y;
    };

    ggml_tensor * cur = g.mel_input;
    g.block_out.resize(6);
    for (int b = 0; b < 6; b++) {
        const vad_block & blk = w.blocks[b];
        ggml_tensor * x_in = cur;
        for (size_t s = 0; s < blk.subs.size(); s++) {
            cur = apply_subconv_masked(cur, blk.subs[s]);
            if (s + 1 < blk.subs.size()) cur = ggml_relu(g.ctx, cur);
        }
        if (blk.residual) {
            ggml_tensor * r = apply_mask(x_in);
            r = pointwise(g.ctx, r, blk.res.pw_w);
            r = apply_bn(g.ctx, r, blk.res.bn_scale, blk.res.bn_bias);
            cur = ggml_add(g.ctx, cur, r);
        }
        cur = ggml_relu(g.ctx, cur);
        std::string nm = "block_out_" + std::to_string(b);
        ggml_set_name(cur, nm.c_str());
        ggml_set_output(cur);
        g.block_out[b] = cur;
    }
    g.encoder_out = g.block_out.back();
    for (int b = 0; b < 6; b++) ggml_build_forward_expand(g.graph, g.block_out[b]);
    return g;
}

bool vad_graph_compute(vad_graph & g, ggml_backend_t backend, ggml_gallocr_t alloc,
                       const float * mel_data, int lens) {
    if (!ggml_gallocr_alloc_graph(alloc, g.graph)) {
        fprintf(stderr, "vad_graph_compute: gallocr_alloc_graph failed\n");
        return false;
    }
    ggml_backend_tensor_set(g.mel_input, mel_data, 0, ggml_nbytes(g.mel_input));

    // Build the per-time mask: 1.0 for t<lens, 0.0 otherwise.
    std::vector<float> mask((size_t)g.T, 0.0f);
    const int lens_clamped = (lens < 0) ? 0 : (lens > g.T ? g.T : lens);
    for (int t = 0; t < lens_clamped; t++) mask[t] = 1.0f;
    ggml_backend_tensor_set(g.mask_input, mask.data(), 0, mask.size() * sizeof(float));

    ggml_status st = ggml_backend_graph_compute(backend, g.graph);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "vad_graph_compute: graph_compute returned %d\n", (int)st);
        return false;
    }
    return true;
}

void vad_graph_free(vad_graph & g) {
    if (g.ctx) { ggml_free(g.ctx); g.ctx = nullptr; }
    g.graph = nullptr;
    g.block_out.clear();
    g.encoder_out = nullptr;
    g.mel_input = nullptr;
    g.mask_input = nullptr;
}

// ---------------------------------------------------------------------------
// vad_session: high-level API
// ---------------------------------------------------------------------------

#include "diarize_audio.h"

struct vad_session {
    diarize_model     * m   = nullptr;     // borrowed
    const vad_weights * w   = nullptr;     // borrowed
    vad_graph           graph;             // owned
    ggml_gallocr_t      alloc = nullptr;   // owned

    // Cached host copies of the small decoder weights (avoids a gallocr round-trip per call).
    std::vector<float> dec_w;              // (ENC_C * N_CLASSES) row-major ne=(C, K)
    std::vector<float> dec_b;              // (N_CLASSES,)
    int n_classes  = 2;
    int enc_c      = 128;

    // Borrowed pointers to the preprocessor weights inside the gguf-backed buffer.
    const float * fb     = nullptr;
    const float * window = nullptr;

    diarize_audio_cfg pp_cfg;

    // Scratch.
    std::vector<float> mel_pp;        // (n_mels, T_padded) row-major
    std::vector<float> mel_chan;      // (T_padded, n_mels) channels-innermost
    std::vector<float> enc_buf;       // (T_padded * ENC_C) channels-innermost
};

vad_session * vad_session_init(diarize_model & m, const vad_weights & w) {
    auto * s = new vad_session;
    s->m = &m;
    s->w = &w;
    s->n_classes = w.n_classes;
    s->enc_c     = w.enc_out_channels;

    s->graph = vad_graph_build(w, kVadMelPadded, kVadNMels);
    s->alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));

    s->dec_w.resize((size_t)s->enc_c * s->n_classes);
    s->dec_b.resize(s->n_classes);
    ggml_backend_tensor_get(w.dec_w, s->dec_w.data(), 0, s->dec_w.size() * sizeof(float));
    ggml_backend_tensor_get(w.dec_b, s->dec_b.data(), 0, s->dec_b.size() * sizeof(float));

    auto * fb_t  = diarize_model_get_tensor(m, "vad.preprocessor.featurizer.fb");
    auto * win_t = diarize_model_get_tensor(m, "vad.preprocessor.featurizer.window");
    GGML_ASSERT(fb_t && win_t);
    s->fb     = static_cast<const float *>(fb_t->data);
    s->window = static_cast<const float *>(win_t->data);

    s->mel_chan.resize((size_t)kVadNMels * kVadMelPadded);
    s->enc_buf.resize((size_t)s->enc_c * kVadMelPadded);

    return s;
}

void vad_session_free(vad_session * s) {
    if (!s) return;
    if (s->alloc) ggml_gallocr_free(s->alloc);
    vad_graph_free(s->graph);
    delete s;
}

namespace {

void to_chan_first_inplace(const float * in, int C, int T, float * out) {
    for (int c = 0; c < C; c++)
        for (int t = 0; t < T; t++)
            out[(size_t)t * C + c] = in[(size_t)c * T + t];
}

} // namespace

float vad_session_run_chunk(vad_session * s, const float * audio, int lens_samples) {
    // Preprocess (always feed the full kVadWindowSamples window — caller is
    // expected to zero-pad if the audio is shorter).
    size_t t_valid = 0;
    size_t t_padded = diarize_compute_logmel(audio, kVadWindowSamples, s->pp_cfg,
                                             s->fb, s->window, s->mel_pp, &t_valid);
    GGML_ASSERT((int)t_padded == kVadMelPadded);
    GGML_ASSERT((int)t_valid  == kVadMelValid);
    (void)t_padded; (void)t_valid;

    // Determine the per-conv mask threshold: how many mel frames are "real".
    int lens_mel = lens_samples / kVadShiftSamples;
    if (lens_mel > kVadMelValid) lens_mel = kVadMelValid;
    if (lens_mel < 0) lens_mel = 0;

    to_chan_first_inplace(s->mel_pp.data(), kVadNMels, kVadMelPadded, s->mel_chan.data());
    if (!vad_graph_compute(s->graph, s->m->backend, s->alloc, s->mel_chan.data(), lens_mel)) {
        return 0.0f;
    }

    ggml_backend_tensor_get(s->graph.encoder_out, s->enc_buf.data(),
                            0, s->enc_buf.size() * sizeof(float));

    // AdaptiveAvgPool1d(1): mean over T (all kVadMelPadded frames).
    std::vector<float> mean_ch(s->enc_c, 0.0f);
    for (int t = 0; t < kVadMelPadded; t++) {
        const float * row = s->enc_buf.data() + (size_t)t * s->enc_c;
        for (int c = 0; c < s->enc_c; c++) mean_ch[c] += row[c];
    }
    const float inv_T = 1.0f / (float)kVadMelPadded;
    for (int c = 0; c < s->enc_c; c++) mean_ch[c] *= inv_T;

    // Linear: logits[k] = sum_c dec_w[c, k] * mean[c] + dec_b[k].
    // dec_w storage: ne=(C, K), C innermost → memory row dec_w[k*C + c].
    std::vector<float> logits(s->n_classes);
    for (int k = 0; k < s->n_classes; k++) {
        float v = s->dec_b[k];
        const float * row = s->dec_w.data() + (size_t)k * s->enc_c;
        for (int c = 0; c < s->enc_c; c++) v += row[c] * mean_ch[c];
        logits[k] = v;
    }

    // Softmax over the 2 classes; return P(speech).
    float mx = logits[0];
    for (int k = 1; k < s->n_classes; k++) if (logits[k] > mx) mx = logits[k];
    float Z = 0.0f;
    for (int k = 0; k < s->n_classes; k++) {
        logits[k] = std::exp(logits[k] - mx);
        Z += logits[k];
    }
    return (s->n_classes > 1) ? (logits[1] / Z) : 0.0f;
}

size_t vad_session_run_batch(vad_session * s, const float * audio, size_t n_samples,
                             std::vector<float> & out) {
    if ((int)n_samples < kVadWindowSamples) return 0;
    const int n_chunks = 1 + ((int)n_samples - kVadWindowSamples) / kVadShiftSamples;
    const size_t before = out.size();
    out.reserve(before + (size_t)n_chunks);
    for (int i = 0; i < n_chunks; i++) {
        const float * chunk = audio + (size_t)i * kVadShiftSamples;
        out.push_back(vad_session_run_chunk(s, chunk, kVadWindowSamples));
    }
    return (size_t)n_chunks;
}

// ---------------------------------------------------------------------------
// Segment extraction
// ---------------------------------------------------------------------------

std::vector<vad_segment> vad_extract_segments(
    const std::vector<float> & probs,
    const vad_post_cfg & cfg)
{
    std::vector<vad_segment> out;
    const float fp = cfg.frame_period_sec;
    const int n = (int)probs.size();
    const int min_on  = (int)std::ceil(cfg.min_duration_on  / fp);
    const int min_off = (int)std::ceil(cfg.min_duration_off / fp);

    bool in_seg = false;
    int  seg_start = -1;
    for (int t = 0; t < n; t++) {
        const float p = probs[t];
        if (!in_seg) {
            if (p >= cfg.onset) { in_seg = true; seg_start = t; }
        } else {
            if (p < cfg.offset) {
                int seg_end = t;
                if (seg_end - seg_start >= min_on) {
                    out.push_back({(seg_start * fp) - cfg.pad_onset,
                                   (seg_end   * fp) + cfg.pad_offset});
                }
                in_seg = false;
            }
        }
    }
    if (in_seg) {
        int seg_end = n;
        if (seg_end - seg_start >= min_on) {
            out.push_back({(seg_start * fp) - cfg.pad_onset,
                           (seg_end   * fp) + cfg.pad_offset});
        }
    }

    // Merge close segments (separation < min_duration_off).
    if (min_off > 0 && out.size() >= 2) {
        std::vector<vad_segment> merged;
        merged.push_back(out[0]);
        for (size_t i = 1; i < out.size(); i++) {
            const float gap_frames = (out[i].start_sec - merged.back().end_sec) / fp;
            if (gap_frames < min_off) {
                merged.back().end_sec = out[i].end_sec;
            } else {
                merged.push_back(out[i]);
            }
        }
        out = std::move(merged);
    }

    // Clamp negatives caused by pad_onset > 0.
    for (auto & seg : out) {
        if (seg.start_sec < 0.0f) seg.start_sec = 0.0f;
        if (seg.end_sec   < seg.start_sec) seg.end_sec = seg.start_sec;
    }
    return out;
}
