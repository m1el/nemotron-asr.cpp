// Bisect the MarbleNet encoder mismatch by running ONLY block 0, with all
// intermediates exposed as graph outputs. Writes:
//   /tmp/cpp_block0_dw.f32      after depthwise
//   /tmp/cpp_block0_pw.f32      after pointwise
//   /tmp/cpp_block0_bn.f32      after BN
//   /tmp/cpp_block0_relu.f32    after ReLU (== block 0 final out)
// Each file is in (C, T) row-major to match the PyTorch fixtures.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include "diarize.h"
#include "diarize_vad.h"

static std::vector<float> read_f32(const std::string & p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "open %s failed\n", p.c_str()); std::exit(2); }
    size_t bytes = (size_t)f.tellg();
    f.seekg(0);
    std::vector<float> buf(bytes / sizeof(float));
    f.read(reinterpret_cast<char *>(buf.data()), bytes);
    return buf;
}

static void write_f32(const std::string & p, const std::vector<float> & v) {
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(float));
}

// (T, C) channels-innermost  ->  (C, T) row-major, like the PyTorch (B, C, T) fixture.
static std::vector<float> to_row_major(const float * in, int C, int T) {
    std::vector<float> out((size_t)C * T);
    for (int t = 0; t < T; t++)
        for (int c = 0; c < C; c++)
            out[(size_t)c * T + t] = in[(size_t)t * C + c];
    return out;
}

// Inverse: (C, T) row-major  ->  (C, T) channels-innermost.
static std::vector<float> to_chan_first(const float * in, int C, int T) {
    std::vector<float> out((size_t)C * T);
    for (int c = 0; c < C; c++)
        for (int t = 0; t < T; t++)
            out[(size_t)t * C + c] = in[(size_t)c * T + t];
    return out;
}

static void diff_against(const std::string & label,
                         const std::vector<float> & cpp,
                         const std::vector<float> & ref,
                         int C, int T, int t_max_compare) {
    if (cpp.size() != ref.size()) {
        fprintf(stderr, "%s: size mismatch cpp=%zu ref=%zu\n",
                label.c_str(), cpp.size(), ref.size());
        return;
    }
    double max_abs = 0, sum_abs = 0;
    int max_idx = -1, n = 0;
    for (int c = 0; c < C; c++) {
        for (int t = 0; t < t_max_compare; t++) {
            size_t i = (size_t)c * T + t;
            float d = std::fabs(cpp[i] - ref[i]);
            if (d > max_abs) { max_abs = d; max_idx = (int)i; }
            sum_abs += d;
            n++;
        }
    }
    fprintf(stdout, "%-12s  max_abs=%.6f  mean_abs=%.6f  cells=%d",
            label.c_str(), max_abs, sum_abs / n, n);
    if (max_idx >= 0) {
        int c_i = max_idx / T;
        int t_i = max_idx % T;
        fprintf(stdout, "   worst at C=%d t=%d (cpp=%.4f ref=%.4f)",
                c_i, t_i, cpp[max_idx], ref[max_idx]);
    }
    fprintf(stdout, "\n");
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <diarize.gguf> <ref_dir>\n", argv[0]);
        return 1;
    }
    std::string ref_dir = argv[2];
    if (!ref_dir.empty() && ref_dir.back() != '/') ref_dir += "/";

    diarize_model m;
    if (!diarize_model_load(argv[1], m)) return 2;

    vad_weights w;
    if (!vad_weights_resolve(m, m.backend, w)) return 3;

    // Use the same mel input used by the PyTorch dump.
    auto mel_ref = read_f32(ref_dir + "mel.f32");
    const int n_mels = 80;
    const int T = (int)mel_ref.size() / n_mels;
    auto mel_in = to_chan_first(mel_ref.data(), n_mels, T);

    // Build a small graph: only block 0, with intermediates set as outputs.
    ggml_init_params p = { .mem_size = 1 * 1024 * 1024, .mem_buffer = nullptr, .no_alloc = true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 1024, false);

    ggml_tensor * mel = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_mels, T);
    ggml_set_input(mel);

    const vad_subconv & s = w.blocks[0].subs[0];
    GGML_ASSERT(s.separable);

    // Depthwise.
    const int pad = s.dilation * (s.kernel - 1) / 2;
    ggml_tensor * padded = ggml_pad_ext(ctx, mel, 0, 0, pad, pad, 0, 0, 0, 0);
    ggml_tensor * dw_out = nullptr;
    for (int i = 0; i < s.kernel; i++) {
        ggml_tensor * slice = ggml_view_2d(ctx, padded, n_mels, T, padded->nb[1],
                                           (size_t)(i * s.dilation) * padded->nb[1]);
        ggml_tensor * kcol_1d = ggml_view_1d(ctx, s.dw_w, n_mels, (size_t)i * s.dw_w->nb[1]);
        ggml_tensor * kcol = ggml_reshape_2d(ctx, kcol_1d, n_mels, 1);
        ggml_tensor * prod = ggml_mul(ctx, slice, kcol);
        dw_out = (dw_out == nullptr) ? prod : ggml_add(ctx, dw_out, prod);
    }
    ggml_set_name(dw_out, "dw_out");
    ggml_set_output(dw_out);

    // Pointwise.
    ggml_tensor * pw_out = ggml_mul_mat(ctx, s.pw_w, dw_out);
    ggml_set_name(pw_out, "pw_out");
    ggml_set_output(pw_out);

    // BN (folded scale + bias).
    const int64_t C_out = pw_out->ne[0];
    ggml_tensor * scale2 = ggml_reshape_2d(ctx, s.bn_scale, C_out, 1);
    ggml_tensor * bias2  = ggml_reshape_2d(ctx, s.bn_bias,  C_out, 1);
    ggml_tensor * bn_out = ggml_mul(ctx, pw_out, scale2);
    bn_out = ggml_add(ctx, bn_out, bias2);
    ggml_set_name(bn_out, "bn_out");
    ggml_set_output(bn_out);

    // ReLU.
    ggml_tensor * relu_out = ggml_relu(ctx, bn_out);
    ggml_set_name(relu_out, "relu_out");
    ggml_set_output(relu_out);

    ggml_build_forward_expand(graph, dw_out);
    ggml_build_forward_expand(graph, pw_out);
    ggml_build_forward_expand(graph, bn_out);
    ggml_build_forward_expand(graph, relu_out);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    if (!ggml_gallocr_alloc_graph(alloc, graph)) {
        fprintf(stderr, "alloc failed\n");
        return 4;
    }

    ggml_backend_tensor_set(mel, mel_in.data(), 0, ggml_nbytes(mel));
    if (ggml_backend_graph_compute(m.backend, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n");
        return 5;
    }
    // Note: this micro-graph does NOT mask. Block 0's bisection runs are
    // expected to diverge slightly near t=520 (vs the masked fixture); we
    // limit the diff window below.

    auto dump = [&](ggml_tensor * t) {
        std::vector<float> v(ggml_nelements(t));
        ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
        return v;
    };

    int C0 = (int)dw_out->ne[0];
    int C1 = (int)pw_out->ne[0];

    auto cpp_dw  = to_row_major(dump(dw_out).data(),  C0, T);
    auto cpp_pw  = to_row_major(dump(pw_out).data(),  C1, T);
    auto cpp_bn  = to_row_major(dump(bn_out).data(),  C1, T);
    auto cpp_relu= to_row_major(dump(relu_out).data(),C1, T);

    auto ref_dw   = read_f32(ref_dir + "block0_dw.f32");
    auto ref_pw   = read_f32(ref_dir + "block0_pw.f32");
    auto ref_bn   = read_f32(ref_dir + "block0_bn.f32");
    auto ref_relu = read_f32(ref_dir + "block0_relu.f32");

    write_f32("/tmp/cpp_block0_dw.f32",   cpp_dw);
    write_f32("/tmp/cpp_block0_pw.f32",   cpp_pw);
    write_f32("/tmp/cpp_block0_bn.f32",   cpp_bn);
    write_f32("/tmp/cpp_block0_relu.f32", cpp_relu);

    // The PyTorch fixture used mask_input(lens=520) which zeros out frames
    // 520..527 BEFORE the conv. We don't do that masking, so frames 515..527
    // will diverge in dw output (kernel radius 5 around the masked region).
    // Compare interior frames only, well away from the mask boundary.
    const int t_compare = 510;

    fprintf(stdout, "diff (frames 0..%d):\n", t_compare - 1);
    diff_against("dw",   cpp_dw,   ref_dw,   C0, T, t_compare);
    diff_against("pw",   cpp_pw,   ref_pw,   C1, T, t_compare);
    diff_against("bn",   cpp_bn,   ref_bn,   C1, T, t_compare);
    diff_against("relu", cpp_relu, ref_relu, C1, T, t_compare);

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    vad_weights_free(w);
    diarize_model_free(m);
    return 0;
}
