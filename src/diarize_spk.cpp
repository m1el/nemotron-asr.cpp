#include "diarize_spk.h"
#include "diarize_audio.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-cpu.h"

// TitaNet-L topology (titanet_large checkpoint).
namespace {
struct block_topology {
    int kernel;
    int dilation;
    int repeat;
    int in_ch;
    int out_ch;
    bool residual;
    bool separable;
    bool has_se;
    int  se_reduction; // 8 for TitaNet
};

constexpr block_topology kBlocks[5] = {
    { 3, 1, 1,   80, 1024, false, true, true, 8},  // block 0
    { 7, 1, 3, 1024, 1024, true,  true, true, 8},  // block 1
    {11, 1, 3, 1024, 1024, true,  true, true, 8},  // block 2
    {15, 1, 3, 1024, 1024, true,  true, true, 8},  // block 3
    { 1, 1, 1, 1024, 3072, false, true, true, 8},  // block 4
};

// Jasper convs build BN as nn.BatchNorm1d(C, eps=1e-3). The decoder side,
// however, uses the PyTorch default eps=1e-5: TDNNModule.__init__ does
// `nn.BatchNorm1d(out_filters)` (no eps) and SpeakerDecoder.affine_layer
// uses `nn.BatchNorm1d(inp_shape, affine=True, track_running_stats=True)`.
constexpr float kEncBNEps = 1e-3f;
constexpr float kDecBNEps = 1e-5f;
constexpr int   kAttnChannels = 128;

ggml_tensor * find(const diarize_model & m, const std::string & name) {
    auto it = m.tensors.find(name);
    if (it == m.tensors.end()) {
        fprintf(stderr, "spk_weights: missing tensor '%s'\n", name.c_str());
        return nullptr;
    }
    return it->second;
}

std::vector<float> read_tensor_host(const ggml_tensor * t) {
    std::vector<float> out(ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
    return out;
}

struct fold_pair { std::vector<float> scale, bias; };

fold_pair fold_bn(const ggml_tensor * gamma, const ggml_tensor * beta,
                  const ggml_tensor * mean, const ggml_tensor * var,
                  float eps) {
    auto g = read_tensor_host(gamma);
    auto b = read_tensor_host(beta);
    auto m = read_tensor_host(mean);
    auto v = read_tensor_host(var);
    fold_pair r;
    const size_t C = g.size();
    r.scale.resize(C); r.bias.resize(C);
    for (size_t i = 0; i < C; i++) {
        float s = g[i] / std::sqrt(v[i] + eps);
        r.scale[i] = s;
        r.bias[i]  = b[i] - m[i] * s;
    }
    return r;
}

struct fold_pending {
    std::vector<float> scale, bias;
    ggml_tensor ** scale_dst;
    ggml_tensor ** bias_dst;
};

bool resolve_subconv(const diarize_model & m, const std::string & prefix,
                     int dw_idx, int pw_idx, int bn_idx,
                     bool separable, int kernel, int dilation,
                     spk_subconv & s,
                     std::vector<fold_pending> & pending) {
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
    auto fp = fold_bn(gamma, beta, mean, var, kEncBNEps);
    pending.push_back({std::move(fp.scale), std::move(fp.bias), &s.bn_scale, &s.bn_bias});
    return true;
}

bool resolve_residual(const diarize_model & m, const std::string & prefix,
                      spk_subconv & s, std::vector<fold_pending> & pending) {
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
    auto fp = fold_bn(gamma, beta, mean, var, kEncBNEps);
    pending.push_back({std::move(fp.scale), std::move(fp.bias), &s.bn_scale, &s.bn_bias});
    return true;
}

} // namespace

bool spk_weights_resolve(const diarize_model & m, ggml_backend_t backend,
                         spk_weights & w) {
    w.attn_channels = kAttnChannels;
    w.blocks.resize(5);
    std::vector<fold_pending> pending;

    for (int b = 0; b < 5; b++) {
        const block_topology & t = kBlocks[b];
        std::string prefix = "spk.encoder.encoder." + std::to_string(b);
        spk_block & blk = w.blocks[b];
        blk.subs.resize(t.repeat);
        blk.residual = t.residual;
        blk.has_se   = t.has_se;

        for (int s = 0; s < t.repeat; s++) {
            int base = 5 * s;
            int dw_idx = base, pw_idx = base + 1, bn_idx = base + 2;
            if (!resolve_subconv(m, prefix, dw_idx, pw_idx, bn_idx,
                                  t.separable, t.kernel, t.dilation,
                                  blk.subs[s], pending)) return false;
        }

        if (t.residual) {
            if (!resolve_residual(m, prefix, blk.res, pending)) return false;
        }

        if (t.has_se) {
            // SE module is at mconv index 5*(R-1)+3.
            const int se_idx = 5 * (t.repeat - 1) + 3;
            std::string se_prefix = prefix + ".mconv." + std::to_string(se_idx);
            blk.se.fc1_w = find(m, se_prefix + ".fc.0.weight");
            blk.se.fc2_w = find(m, se_prefix + ".fc.2.weight");
            if (!blk.se.fc1_w || !blk.se.fc2_w) return false;
        }
    }

    // Decoder: attentive pool + emb conv.
    const std::string dp = "spk.decoder";
    auto & d = w.dec;
    d.attn_conv1_w = find(m, dp + "._pooling.attention_layer.0.conv_layer.weight");
    d.attn_conv1_b = find(m, dp + "._pooling.attention_layer.0.conv_layer.bias");
    auto * agm = find(m, dp + "._pooling.attention_layer.0.bn.weight");
    auto * abt = find(m, dp + "._pooling.attention_layer.0.bn.bias");
    auto * amn = find(m, dp + "._pooling.attention_layer.0.bn.running_mean");
    auto * avr = find(m, dp + "._pooling.attention_layer.0.bn.running_var");
    if (!d.attn_conv1_w || !d.attn_conv1_b || !agm || !abt || !amn || !avr) return false;
    auto fa = fold_bn(agm, abt, amn, avr, kDecBNEps);
    pending.push_back({std::move(fa.scale), std::move(fa.bias), &d.attn_bn_scale, &d.attn_bn_bias});

    d.attn_conv2_w = find(m, dp + "._pooling.attention_layer.2.weight");
    d.attn_conv2_b = find(m, dp + "._pooling.attention_layer.2.bias");
    if (!d.attn_conv2_w || !d.attn_conv2_b) return false;

    auto * egm = find(m, dp + ".emb_layers.0.0.weight");
    auto * ebt = find(m, dp + ".emb_layers.0.0.bias");
    auto * emn = find(m, dp + ".emb_layers.0.0.running_mean");
    auto * evr = find(m, dp + ".emb_layers.0.0.running_var");
    if (!egm || !ebt || !emn || !evr) return false;
    auto fe = fold_bn(egm, ebt, emn, evr, kDecBNEps);
    pending.push_back({std::move(fe.scale), std::move(fe.bias), &d.emb_bn_scale, &d.emb_bn_bias});

    d.emb_conv_w = find(m, dp + ".emb_layers.0.1.weight");
    d.emb_conv_b = find(m, dp + ".emb_layers.0.1.bias");
    if (!d.emb_conv_w || !d.emb_conv_b) return false;

    w.enc_out_channels = (int)d.attn_conv2_b->ne[0];      // 3072
    w.emb_dim          = (int)d.emb_conv_b->ne[0];        // 192

    // Allocate fold tensors.
    ggml_init_params p = {
        .mem_size   = ggml_tensor_overhead() * (pending.size() * 2 + 16),
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    w.fold_ctx = ggml_init(p);
    if (!w.fold_ctx) return false;

    for (auto & e : pending) {
        const int64_t C = (int64_t)e.scale.size();
        ggml_tensor * scale = ggml_new_tensor_1d(w.fold_ctx, GGML_TYPE_F32, C);
        ggml_tensor * bias  = ggml_new_tensor_1d(w.fold_ctx, GGML_TYPE_F32, C);
        ggml_set_name(scale, "spk_bn_scale");
        ggml_set_name(bias,  "spk_bn_bias");
        *e.scale_dst = scale;
        *e.bias_dst  = bias;
    }

    w.fold_buf = ggml_backend_alloc_ctx_tensors(w.fold_ctx, backend);
    if (!w.fold_buf) return false;

    for (auto & e : pending) {
        ggml_backend_tensor_set(*e.scale_dst, e.scale.data(), 0, e.scale.size() * sizeof(float));
        ggml_backend_tensor_set(*e.bias_dst,  e.bias.data(),  0, e.bias.size()  * sizeof(float));
    }

    return true;
}

void spk_weights_free(spk_weights & w) {
    if (w.fold_buf) { ggml_backend_buffer_free(w.fold_buf); w.fold_buf = nullptr; }
    if (w.fold_ctx) { ggml_free(w.fold_ctx); w.fold_ctx = nullptr; }
    w.blocks.clear();
}

// ---------------------------------------------------------------------------
// Graph build
// ---------------------------------------------------------------------------

namespace {

ggml_tensor * pointwise(ggml_context * ctx, ggml_tensor * x, ggml_tensor * pw_w) {
    return ggml_mul_mat(ctx, pw_w, x);
}

ggml_tensor * depthwise_same(ggml_context * ctx, ggml_tensor * x,
                             ggml_tensor * dw_w, int kernel, int dilation) {
    const int64_t C = x->ne[0];
    const int64_t T = x->ne[1];
    const int pad = dilation * (kernel - 1) / 2;
    GGML_ASSERT(dw_w->ne[0] == C);
    GGML_ASSERT(dw_w->ne[1] == kernel);
    if (kernel == 1) {
        // Per-channel scaling: x[c, t] *= dw_w[c]. dw_w stored as ne=(C, 1).
        ggml_tensor * kvec = ggml_reshape_2d(ctx, dw_w, C, 1);
        return ggml_mul(ctx, x, kvec);
    }
    ggml_tensor * padded = ggml_pad_ext(ctx, x, 0, 0, pad, pad, 0, 0, 0, 0);
    ggml_tensor * acc = nullptr;
    for (int i = 0; i < kernel; i++) {
        const size_t time_off = (size_t)(i * dilation) * padded->nb[1];
        ggml_tensor * slice = ggml_view_2d(ctx, padded, C, T, padded->nb[1], time_off);
        ggml_tensor * kcol_1d = ggml_view_1d(ctx, dw_w, C, (size_t)i * dw_w->nb[1]);
        ggml_tensor * kcol = ggml_reshape_2d(ctx, kcol_1d, C, 1);
        ggml_tensor * prod = ggml_mul(ctx, slice, kcol);
        acc = (acc == nullptr) ? prod : ggml_add(ctx, acc, prod);
    }
    return acc;
}

ggml_tensor * apply_bn_fold(ggml_context * ctx, ggml_tensor * x,
                            ggml_tensor * scale, ggml_tensor * bias) {
    const int64_t C = x->ne[0];
    GGML_ASSERT(scale->ne[0] == C);
    ggml_tensor * scale2 = ggml_reshape_2d(ctx, scale, C, 1);
    ggml_tensor * bias2  = ggml_reshape_2d(ctx, bias,  C, 1);
    ggml_tensor * y = ggml_mul(ctx, x, scale2);
    return ggml_add(ctx, y, bias2);
}

// Apply per-time mask: x (C, T) * mask (1, T) → x with masked frames zeroed.
ggml_tensor * apply_mask(ggml_context * ctx, ggml_tensor * x, ggml_tensor * mask) {
    return ggml_mul(ctx, x, mask);
}

// Mean over T for each channel (masked: input is already x_masked).
// inv_lens is a (1, 1) scalar tensor = 1/lens (so we get a true mean over
// valid frames, not over T_padded). Result has shape (C, 1).
ggml_tensor * masked_mean_over_T(ggml_context * ctx, ggml_tensor * x_masked,
                                 ggml_tensor * inv_lens) {
    // Sum over T: transpose to (T, C), sum_rows → (1, C), reshape to (C, 1).
    ggml_tensor * xT = ggml_cont(ctx, ggml_permute(ctx, x_masked, 1, 0, 2, 3)); // (T, C)
    ggml_tensor * sum = ggml_sum_rows(ctx, xT);                                  // (1, C)
    const int64_t C = x_masked->ne[0];
    sum = ggml_reshape_2d(ctx, sum, C, 1);                                       // (C, 1)
    return ggml_mul(ctx, sum, inv_lens);                                         // broadcast (1,1)
}

// SE block: y = x * sigmoid(fc2(relu(fc1(masked_mean(x))))).
// Returns the SE-scaled output; the input x is expected to already be masked.
ggml_tensor * apply_se(ggml_context * ctx, ggml_tensor * x_masked,
                       const spk_se & se, ggml_tensor * inv_lens) {
    ggml_tensor * mean = masked_mean_over_T(ctx, x_masked, inv_lens); // (C, 1)
    // fc1: (C, C/r) @ (C, 1) → (C/r, 1)
    ggml_tensor * y = ggml_mul_mat(ctx, se.fc1_w, mean);
    y = ggml_relu(ctx, y);
    // fc2: (C/r, C) @ (C/r, 1) → (C, 1)
    y = ggml_mul_mat(ctx, se.fc2_w, y);
    y = ggml_sigmoid(ctx, y);
    // Multiply across time: (C, T) * (C, 1) broadcasts.
    return ggml_mul(ctx, x_masked, y);
}

// Per-bias addition: y[c, t] = x[c, t] + b[c]. b stored as (C,).
ggml_tensor * add_bias(ggml_context * ctx, ggml_tensor * x, ggml_tensor * b) {
    const int64_t C = x->ne[0];
    GGML_ASSERT(b->ne[0] == C);
    ggml_tensor * b2 = ggml_reshape_2d(ctx, b, C, 1);
    return ggml_add(ctx, x, b2);
}

} // namespace

spk_graph spk_graph_build(const spk_weights & w, int T, int n_mels) {
    spk_graph g;
    g.T = T;
    g.n_mels = n_mels;

    ggml_init_params p = { .mem_size = 8 * 1024 * 1024, .mem_buffer = nullptr, .no_alloc = true };
    g.ctx = ggml_init(p);
    g.graph = ggml_new_graph_custom(g.ctx, /*size=*/16384, /*grads=*/false);

    g.mel_input  = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, n_mels, T);
    g.mask_input = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, 1, T);
    g.inv_lens   = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, 1, 1);
    ggml_set_name(g.mel_input,  "mel_input");
    ggml_set_name(g.mask_input, "mask_input");
    ggml_set_name(g.inv_lens,   "inv_lens");
    ggml_set_input(g.mel_input);
    ggml_set_input(g.mask_input);
    ggml_set_input(g.inv_lens);

    auto subconv_masked = [&](ggml_tensor * x, const spk_subconv & s) -> ggml_tensor * {
        ggml_tensor * y = x;
        if (s.separable) {
            y = apply_mask(g.ctx, y, g.mask_input);
            y = depthwise_same(g.ctx, y, s.dw_w, s.kernel, s.dilation);
        }
        y = apply_mask(g.ctx, y, g.mask_input);
        y = pointwise(g.ctx, y, s.pw_w);
        y = apply_bn_fold(g.ctx, y, s.bn_scale, s.bn_bias);
        return y;
    };

    // Encoder. JasperBlock.forward order:
    //   for module in mconv: out = module(out)        -- mconv ends with SE
    //   if residual: out = out + res_path(input)
    //   out = mout(out)                                -- ReLU
    ggml_tensor * cur = g.mel_input;
    g.block_out.resize(5);
    for (int b = 0; b < 5; b++) {
        const spk_block & blk = w.blocks[b];
        ggml_tensor * x_in = cur;
        for (size_t s = 0; s < blk.subs.size(); s++) {
            cur = subconv_masked(cur, blk.subs[s]);
            if (s + 1 < blk.subs.size()) cur = ggml_relu(g.ctx, cur);
        }
        // SE is the last item inside mconv — applied BEFORE the residual.
        if (blk.has_se) {
            ggml_tensor * x_masked = apply_mask(g.ctx, cur, g.mask_input);
            cur = apply_se(g.ctx, x_masked, blk.se, g.inv_lens);
        }
        if (blk.residual) {
            ggml_tensor * r = apply_mask(g.ctx, x_in, g.mask_input);
            r = pointwise(g.ctx, r, blk.res.pw_w);
            r = apply_bn_fold(g.ctx, r, blk.res.bn_scale, blk.res.bn_bias);
            cur = ggml_add(g.ctx, cur, r);
        }
        // mout = ReLU (Dropout is identity at eval).
        cur = ggml_relu(g.ctx, cur);
        std::string nm = "block_out_" + std::to_string(b);
        ggml_set_name(cur, nm.c_str());
        ggml_set_output(cur);
        g.block_out[b] = cur;
    }
    g.encoder_out = g.block_out.back(); // (3072, T)

    // -------- Decoder: attentive pooling + emb conv ----------
    // Compute mean(x) and std(x) over T (masked).
    const int64_t C  = g.encoder_out->ne[0];
    const int64_t Tg = g.encoder_out->ne[1];

    // Mask the encoder output for the stat computation and for the final
    // weighted stats. NeMo masks before computing mean/std and again before
    // softmax over the attention weights.
    ggml_tensor * x_masked = apply_mask(g.ctx, g.encoder_out, g.mask_input);

    // mean over T: (C, T) -> (C, 1).
    ggml_tensor * mean = masked_mean_over_T(g.ctx, x_masked, g.inv_lens);
    // For the std, compute (x - mean)^2 then masked-mean-then-sqrt with eps.
    ggml_tensor * mean_bcast = ggml_repeat(g.ctx, mean, x_masked); // (C, T)
    ggml_tensor * diff = ggml_sub(g.ctx, x_masked, mean_bcast);
    ggml_tensor * diff_masked = apply_mask(g.ctx, diff, g.mask_input);  // re-mask after subtract
    ggml_tensor * sq = ggml_sqr(g.ctx, diff_masked);
    ggml_tensor * var = masked_mean_over_T(g.ctx, sq, g.inv_lens);  // (C, 1)
    // std = sqrt(var + eps).
    // We need var + eps. ggml has no add-scalar; use a 1-element constant tensor.
    // For simplicity: use ggml_sqrt; eps≈1e-10 is the NeMo default in
    // get_statistics_with_mask (eps via clamp(eps)). NeMo does
    //   std = sqrt(((m * (x - mean)^2).sum(dim).clamp(eps))
    // i.e. clamp the sum (not divided) to eps. We approximate by clamping var.
    // Achievable with ggml_clamp.
    ggml_tensor * std_t = ggml_sqrt(g.ctx, ggml_clamp(g.ctx, var, 1e-10f, 1e30f));

    // Broadcast mean and std to (C, T) for concat.
    ggml_tensor * mean_bcast2 = ggml_repeat(g.ctx, mean, x_masked);  // (C, T)
    ggml_tensor * std_bcast   = ggml_repeat(g.ctx, std_t, x_masked); // (C, T)

    // Concat [x, mean_bcast, std_bcast] along channel dim → (3C, T).
    // ggml_concat axis 0 = ne[0] (channel for our convention).
    ggml_tensor * concat_xm  = ggml_concat(g.ctx, x_masked,   mean_bcast2, 0);
    ggml_tensor * concat_xms = ggml_concat(g.ctx, concat_xm,  std_bcast,    0);
    GGML_ASSERT(concat_xms->ne[0] == 3 * C);
    GGML_ASSERT(concat_xms->ne[1] == Tg);

    // attn_conv1: matmul (3C, attn_C) → (attn_C, T), then add bias, ReLU, BN_fold.
    ggml_tensor * a = pointwise(g.ctx, concat_xms, w.dec.attn_conv1_w); // (attn_C, T)
    a = add_bias(g.ctx, a, w.dec.attn_conv1_b);
    a = ggml_relu(g.ctx, a);
    a = apply_bn_fold(g.ctx, a, w.dec.attn_bn_scale, w.dec.attn_bn_bias);

    // Tanh.
    a = ggml_tanh(g.ctx, a);

    // attn_conv2: matmul (attn_C, C) → (C, T), then add bias.
    a = pointwise(g.ctx, a, w.dec.attn_conv2_w); // (C, T)
    a = add_bias(g.ctx, a, w.dec.attn_conv2_b);

    // Mask: set masked positions to -inf so they vanish under softmax.
    // ggml has no mask-fill; we add a large negative value where mask=0.
    // Build (mask - 1) * 1e9 = 0 where valid, -1e9 where pad. Add to a.
    ggml_tensor * mask_minus1 = ggml_add(g.ctx, g.mask_input,
        ggml_scale(g.ctx, ggml_dup_tensor(g.ctx, g.mask_input), 0.0f));
    (void)mask_minus1;
    // Simpler: a = a * mask  + (mask - 1) * BIG  ... but no scalar-add ops.
    // Approach: multiply attention by mask first; for the masked positions
    // softmax will contribute zero IF the masked positions have very
    // negative logits. Since mask zeros them, softmax(0) for masked positions
    // is exp(0)=1 which is wrong. We need to push masked logits to -inf.
    //
    // Workaround: use ggml_soft_max_ext which supports an additive mask.
    // ggml_soft_max_ext expects mask shape (T, T) for self-attn. For our
    // simple per-time mask, easier: subtract a big constant where mask=0.
    //
    // Build (1 - mask) by -1*mask + 1; multiply by big; subtract.
    // ggml_scale(mask, -1) gives -mask. Add 1 needs scalar add (no op).
    //
    // Cleaner: compute (mask - 1.0) as bias. But there's no scalar-add op
    // in ggml. We use ggml_add1 which adds a scalar tensor (1-element)
    // broadcast.
    ggml_tensor * neg_mask = ggml_scale(g.ctx, g.mask_input, -1.0f);
    // Add scalar 1.0 broadcast: use ggml_add1 with a (1,) scalar tensor.
    // We don't have a constant 1.0 as a graph input, so create it.
    // Simpler: precompute (1 - mask) on the host and pass as a graph input.
    // But we already pass mask as input; instead, pass a separate "neg_mask_inf"
    // tensor as a graph input that the host fills with 0 at valid positions
    // and -1e9 at masked positions. Then add to logits.
    (void)neg_mask;
    // Build neg_mask_inf as a separate graph input.
    g.neg_mask_inf = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, 1, T);
    ggml_set_name(g.neg_mask_inf, "neg_mask_inf");
    ggml_set_input(g.neg_mask_inf);
    a = ggml_add(g.ctx, a, g.neg_mask_inf);

    // Softmax over T axis. ggml_soft_max operates on ne[0]. To softmax over T
    // (ne[1]), permute first.
    ggml_tensor * a_TC = ggml_cont(g.ctx, ggml_permute(g.ctx, a, 1, 0, 2, 3)); // (T, C)
    ggml_tensor * alpha_TC = ggml_soft_max(g.ctx, a_TC);  // softmax over ne[0]=T
    ggml_tensor * alpha = ggml_cont(g.ctx, ggml_permute(g.ctx, alpha_TC, 1, 0, 2, 3)); // (C, T)

    // Weighted statistics: mu = sum_t alpha[c,t] * x[c,t], sg = sqrt(sum_t alpha[c,t] * (x[c,t] - mu)^2).
    // NOTE: The encoder output x_masked is used (already masked), and
    // alpha is per-channel-per-time so we don't need to divide by lens here
    // (alpha sums to 1 over T per channel after softmax).
    ggml_tensor * x_alpha = ggml_mul(g.ctx, x_masked, alpha);  // (C, T)
    // sum over T: transpose, sum_rows, reshape.
    ggml_tensor * mu_T = ggml_cont(g.ctx, ggml_permute(g.ctx, x_alpha, 1, 0, 2, 3));
    ggml_tensor * mu = ggml_sum_rows(g.ctx, mu_T);  // (1, C)
    mu = ggml_reshape_2d(g.ctx, mu, C, 1);          // (C, 1)

    ggml_tensor * mu_bcast = ggml_repeat(g.ctx, mu, x_masked);
    ggml_tensor * dxw = ggml_sub(g.ctx, x_masked, mu_bcast);
    ggml_tensor * dxw_sq = ggml_sqr(g.ctx, dxw);
    ggml_tensor * w_dxw = ggml_mul(g.ctx, dxw_sq, alpha);   // (C, T)
    ggml_tensor * sg_T = ggml_cont(g.ctx, ggml_permute(g.ctx, w_dxw, 1, 0, 2, 3));
    ggml_tensor * sg2 = ggml_sum_rows(g.ctx, sg_T);          // (1, C)
    sg2 = ggml_reshape_2d(g.ctx, sg2, C, 1);                  // (C, 1)
    ggml_tensor * sg = ggml_sqrt(g.ctx, ggml_clamp(g.ctx, sg2, 1e-10f, 1e30f));

    // Concat (mu, sg) along channels → (2C, 1).
    ggml_tensor * pool = ggml_concat(g.ctx, mu, sg, 0);  // (2C, 1)
    GGML_ASSERT(pool->ne[0] == 2 * C);
    g.pool_out = pool;
    ggml_set_name(g.pool_out, "pool_out");
    ggml_set_output(g.pool_out);

    // emb_layer: BN(2C) folded → Conv1d(2C → emb_dim, k=1).
    ggml_tensor * e = apply_bn_fold(g.ctx, pool, w.dec.emb_bn_scale, w.dec.emb_bn_bias); // (2C, 1)
    e = pointwise(g.ctx, e, w.dec.emb_conv_w);          // (emb_dim, 1)
    e = add_bias(g.ctx, e, w.dec.emb_conv_b);
    g.embedding = e;
    ggml_set_name(g.embedding, "embedding");
    ggml_set_output(g.embedding);

    for (auto * t : g.block_out) ggml_build_forward_expand(g.graph, t);
    ggml_build_forward_expand(g.graph, g.pool_out);
    ggml_build_forward_expand(g.graph, g.embedding);
    return g;
}

bool spk_graph_compute(spk_graph & g, ggml_backend_t backend, ggml_gallocr_t alloc,
                       const float * mel_data, int lens) {
    if (!ggml_gallocr_alloc_graph(alloc, g.graph)) {
        fprintf(stderr, "spk_graph_compute: gallocr_alloc_graph failed\n");
        return false;
    }
    ggml_backend_tensor_set(g.mel_input, mel_data, 0, ggml_nbytes(g.mel_input));

    const int lens_clamped = (lens < 1) ? 1 : (lens > g.T ? g.T : lens);

    std::vector<float> mask((size_t)g.T, 0.0f);
    for (int t = 0; t < lens_clamped; t++) mask[t] = 1.0f;
    ggml_backend_tensor_set(g.mask_input, mask.data(), 0, mask.size() * sizeof(float));

    float inv_lens = 1.0f / (float)lens_clamped;
    ggml_backend_tensor_set(g.inv_lens, &inv_lens, 0, sizeof(float));

    if (g.neg_mask_inf) {
        std::vector<float> nm((size_t)g.T, 0.0f);
        for (int t = lens_clamped; t < g.T; t++) nm[t] = -1.0e9f;
        ggml_backend_tensor_set(g.neg_mask_inf, nm.data(), 0, nm.size() * sizeof(float));
    }

    ggml_status st = ggml_backend_graph_compute(backend, g.graph);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "spk_graph_compute: graph_compute returned %d\n", (int)st);
        return false;
    }
    return true;
}

void spk_graph_free(spk_graph & g) {
    if (g.ctx) { ggml_free(g.ctx); g.ctx = nullptr; }
    g.graph = nullptr;
    g.block_out.clear();
    g.encoder_out = g.pool_out = g.embedding = nullptr;
    g.mel_input = g.mask_input = g.inv_lens = nullptr;
}

// ---------------------------------------------------------------------------
// High-level session API
// ---------------------------------------------------------------------------

struct spk_session {
    diarize_model * m  = nullptr;
    const spk_weights * w = nullptr;
    spk_graph graph;
    ggml_gallocr_t alloc = nullptr;

    const float * fb = nullptr;
    const float * window = nullptr;
    diarize_audio_cfg pp_cfg;

    std::vector<float> mel_pp;
    std::vector<float> mel_chan;
};

spk_session * spk_session_init(diarize_model & m, const spk_weights & w) {
    auto * s = new spk_session;
    s->m = &m;
    s->w = &w;
    s->graph = spk_graph_build(w, kSpkMelPadded, kSpkNMels);
    s->alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));

    auto * fb_t  = diarize_model_get_tensor(m, "spk.preprocessor.featurizer.fb");
    auto * win_t = diarize_model_get_tensor(m, "spk.preprocessor.featurizer.window");
    GGML_ASSERT(fb_t && win_t);
    s->fb     = static_cast<const float *>(fb_t->data);
    s->window = static_cast<const float *>(win_t->data);

    s->pp_cfg.per_feature_normalize = true;
    s->mel_chan.resize((size_t)kSpkNMels * kSpkMelPadded);
    return s;
}

void spk_session_free(spk_session * s) {
    if (!s) return;
    if (s->alloc) ggml_gallocr_free(s->alloc);
    spk_graph_free(s->graph);
    delete s;
}

bool spk_session_run_chunk(spk_session * s, const float * audio,
                           int lens_samples, float * out_emb) {
    size_t t_valid = 0, t_padded = 0;
    t_padded = diarize_compute_logmel(audio, kSpkSubsegSamples, s->pp_cfg,
                                      s->fb, s->window, s->mel_pp, &t_valid);
    if ((int)t_padded != kSpkMelPadded || (int)t_valid != kSpkMelValid) {
        fprintf(stderr, "spk_session: unexpected mel shape t_valid=%zu t_padded=%zu\n",
                t_valid, t_padded);
        return false;
    }

    int lens_mel = lens_samples / 160;
    if (lens_mel > kSpkMelValid) lens_mel = kSpkMelValid;
    if (lens_mel < 1) lens_mel = 1;

    for (int c = 0; c < kSpkNMels; c++)
        for (int t = 0; t < kSpkMelPadded; t++)
            s->mel_chan[(size_t)t * kSpkNMels + c] = s->mel_pp[(size_t)c * kSpkMelPadded + t];

    if (!spk_graph_compute(s->graph, s->m->backend, s->alloc, s->mel_chan.data(), lens_mel))
        return false;

    ggml_backend_tensor_get(s->graph.embedding, out_emb, 0, kSpkEmbDim * sizeof(float));
    return true;
}
