#include "nemo-ggml.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>

// Maximum nodes in computation graph
#define NEMO_MAX_NODES 8192

// Compute positional embeddings (sinusoidal)
// NeMo convention: pos[0] = most positive position, pos[total_len-1] = most negative
// i.e., stored in descending order: +max_len-1, +max_len-2, ..., 0, ..., -(max_len-1)
static void compute_pos_emb(float * data, int max_len, int d_model) {
    int total_len = 2 * max_len - 1;

    for (int pos = 0; pos < total_len; pos++) {
        // Reversed: p goes from +(max_len-1) down to -(max_len-1)
        float p = (float)(max_len - 1) - (float)pos;

        for (int i = 0; i < d_model; i += 2) {
            float div_term = std::exp(-(float)i * std::log(10000.0f) / (float)d_model);
            data[pos * d_model + i] = std::sin(p * div_term);
            if (i + 1 < d_model) {
                data[pos * d_model + i + 1] = std::cos(p * div_term);
            }
        }
    }
}

bool nemo_model_load(const std::string & path, nemo_model & model) {
    // printf("%s: loading model from '%s'\n", __func__, path.c_str());

    // Initialize backend first
    model.backend = ggml_backend_cpu_init();
    if (!model.backend) {
        fprintf(stderr, "%s: failed to init CPU backend\n", __func__);
        return false;
    }

    // Open GGUF file
    struct ggml_context * ctx_meta = nullptr;
    struct gguf_init_params params = {
        .no_alloc = true,
        .ctx      = &ctx_meta,
    };

    struct gguf_context * gguf_ctx = gguf_init_from_file(path.c_str(), params);
    if (!gguf_ctx) {
        fprintf(stderr, "%s: failed to open GGUF file\n", __func__);
        return false;
    }

    // Read hyperparameters from GGUF metadata
    int64_t kv_idx;

    kv_idx = gguf_find_key(gguf_ctx, "nemo.n_mels");
    if (kv_idx >= 0) model.hparams.n_mels = gguf_get_val_u32(gguf_ctx, kv_idx);

    kv_idx = gguf_find_key(gguf_ctx, "nemo.d_model");
    if (kv_idx >= 0) model.hparams.d_model = gguf_get_val_u32(gguf_ctx, kv_idx);

    kv_idx = gguf_find_key(gguf_ctx, "nemo.n_heads");
    if (kv_idx >= 0) model.hparams.n_heads = gguf_get_val_u32(gguf_ctx, kv_idx);

    kv_idx = gguf_find_key(gguf_ctx, "nemo.d_head");
    if (kv_idx >= 0) model.hparams.d_head = gguf_get_val_u32(gguf_ctx, kv_idx);

    kv_idx = gguf_find_key(gguf_ctx, "nemo.d_ff");
    if (kv_idx >= 0) model.hparams.d_ff = gguf_get_val_u32(gguf_ctx, kv_idx);

    kv_idx = gguf_find_key(gguf_ctx, "nemo.n_layers");
    if (kv_idx >= 0) model.hparams.n_layers = gguf_get_val_u32(gguf_ctx, kv_idx);

    kv_idx = gguf_find_key(gguf_ctx, "nemo.vocab_size");
    if (kv_idx >= 0) model.hparams.vocab_size = gguf_get_val_u32(gguf_ctx, kv_idx);

    kv_idx = gguf_find_key(gguf_ctx, "nemo.decoder_dim");
    if (kv_idx >= 0) model.hparams.decoder_dim = gguf_get_val_u32(gguf_ctx, kv_idx);

    kv_idx = gguf_find_key(gguf_ctx, "nemo.joint_dim");
    if (kv_idx >= 0) model.hparams.joint_dim = gguf_get_val_u32(gguf_ctx, kv_idx);

    // Load vocabulary
    // vocab is stored as a string containing vocab_size * 8 bytes (each token is a char8)
    kv_idx = gguf_find_key(gguf_ctx, "tokenizer.vocab");
    if (kv_idx >= 0) {
        const char * vocab_data = gguf_get_val_str(gguf_ctx, kv_idx);
        // Use vocab_size from hparams (already loaded), each entry is 8 bytes
        size_t n_tokens = model.hparams.vocab_size;  // 1025
        size_t vocab_entry_size = 8;  // char8
        size_t vocab_bytes = n_tokens * vocab_entry_size;
        model.vocab.resize(n_tokens, {0});
        memcpy(model.vocab.data(), vocab_data, vocab_bytes);
    }

    // printf("%s: n_mels     = %d\n", __func__, model.hparams.n_mels);
    // printf("%s: d_model    = %d\n", __func__, model.hparams.d_model);
    // printf("%s: n_heads    = %d\n", __func__, model.hparams.n_heads);
    // printf("%s: n_layers   = %d\n", __func__, model.hparams.n_layers);
    // printf("%s: vocab_size = %d\n", __func__, model.hparams.vocab_size);
    // printf("%s: vocab tokens = %zu\n", __func__, model.vocab.size());

    // Get tensor count
    int64_t n_tensors = gguf_get_n_tensors(gguf_ctx);
    // printf("%s: n_tensors  = %lld\n", __func__, (long long)n_tensors);

    // Create weight context with no_alloc=true
    size_t ctx_size = ggml_tensor_overhead() * (n_tensors + 10);  // +10 for pos_emb
    struct ggml_init_params ctx_params = {
        .mem_size   = ctx_size,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };

    model.ctx_w = ggml_init(ctx_params);
    if (!model.ctx_w) {
        fprintf(stderr, "%s: failed to init ggml context\n", __func__);
        gguf_free(gguf_ctx);
        ggml_free(ctx_meta);
        return false;
    }

    // Create tensors from GGUF metadata
    model.encoder.layers.resize(model.hparams.n_layers);

    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        struct ggml_tensor * meta_tensor = ggml_get_tensor(ctx_meta, name);

        if (!meta_tensor) {
            fprintf(stderr, "%s: tensor '%s' not found in meta context\n", __func__, name);
            continue;
        }

        // Create tensor in our context with same dimensions
        struct ggml_tensor * tensor = ggml_dup_tensor(model.ctx_w, meta_tensor);
        ggml_set_name(tensor, name);
        model.tensors[name] = tensor;
    }

    // Add positional embedding tensor (precomputed)
    const int max_pos_len = 512;
    model.pos_emb = ggml_new_tensor_2d(model.ctx_w, GGML_TYPE_F32,
                                        model.hparams.d_model, 2 * max_pos_len - 1);
    ggml_set_name(model.pos_emb, "pos_emb");

    // Allocate backend buffer
    model.buffer_w = ggml_backend_alloc_ctx_tensors(model.ctx_w, model.backend);
    if (!model.buffer_w) {
        fprintf(stderr, "%s: failed to allocate backend buffer\n", __func__);
        gguf_free(gguf_ctx);
        ggml_free(ctx_meta);
        return false;
    }

    // Load tensor data from GGUF file
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "%s: failed to reopen file for data loading\n", __func__);
        gguf_free(gguf_ctx);
        ggml_free(ctx_meta);
        return false;
    }

    size_t data_offset = gguf_get_data_offset(gguf_ctx);

    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        size_t tensor_offset = gguf_get_tensor_offset(gguf_ctx, i);
        size_t tensor_size = gguf_get_tensor_size(gguf_ctx, i);

        auto it = model.tensors.find(name);
        if (it == model.tensors.end()) {
            continue;
        }

        struct ggml_tensor * tensor = it->second;

        // Read data from file
        std::vector<char> buf(tensor_size);
        fseek(f, data_offset + tensor_offset, SEEK_SET);
        if (fread(buf.data(), 1, tensor_size, f) != tensor_size) {
            fprintf(stderr, "%s: failed to read tensor '%s'\n", __func__, name);
            fclose(f);
            gguf_free(gguf_ctx);
            ggml_free(ctx_meta);
            return false;
        }

        // Set tensor data in backend
        ggml_backend_tensor_set(tensor, buf.data(), 0, tensor_size);
    }

    fclose(f);

    // Compute and set positional embeddings
    {
        std::vector<float> pos_data((2 * max_pos_len - 1) * model.hparams.d_model);
        compute_pos_emb(pos_data.data(), max_pos_len, model.hparams.d_model);
        ggml_backend_tensor_set(model.pos_emb, pos_data.data(), 0, pos_data.size() * sizeof(float));
    }

    // Map tensors to model structure
    // ConvSubsampling
    model.encoder.subsampling.conv0_w = model.tensors["encoder.pre_encode.conv.0.weight"];
    model.encoder.subsampling.conv0_b = model.tensors["encoder.pre_encode.conv.0.bias"];
    model.encoder.subsampling.conv2_w = model.tensors["encoder.pre_encode.conv.2.weight"];
    model.encoder.subsampling.conv2_b = model.tensors["encoder.pre_encode.conv.2.bias"];
    model.encoder.subsampling.conv3_w = model.tensors["encoder.pre_encode.conv.3.weight"];
    model.encoder.subsampling.conv3_b = model.tensors["encoder.pre_encode.conv.3.bias"];
    model.encoder.subsampling.conv5_w = model.tensors["encoder.pre_encode.conv.5.weight"];
    model.encoder.subsampling.conv5_b = model.tensors["encoder.pre_encode.conv.5.bias"];
    model.encoder.subsampling.conv6_w = model.tensors["encoder.pre_encode.conv.6.weight"];
    model.encoder.subsampling.conv6_b = model.tensors["encoder.pre_encode.conv.6.bias"];
    model.encoder.subsampling.out_w   = model.tensors["encoder.pre_encode.out.weight"];
    model.encoder.subsampling.out_b   = model.tensors["encoder.pre_encode.out.bias"];

    // Encoder layers
    for (int i = 0; i < model.hparams.n_layers; i++) {
        auto & layer = model.encoder.layers[i];
        std::string prefix = "encoder.layers." + std::to_string(i);

        layer.norm_ff1_w = model.tensors[prefix + ".norm_feed_forward1.weight"];
        layer.norm_ff1_b = model.tensors[prefix + ".norm_feed_forward1.bias"];
        layer.ffn1_linear1_w = model.tensors[prefix + ".feed_forward1.linear1.weight"];
        layer.ffn1_linear2_w = model.tensors[prefix + ".feed_forward1.linear2.weight"];

        layer.norm_attn_w = model.tensors[prefix + ".norm_self_att.weight"];
        layer.norm_attn_b = model.tensors[prefix + ".norm_self_att.bias"];
        layer.attn_q_w = model.tensors[prefix + ".self_attn.linear_q.weight"];
        layer.attn_k_w = model.tensors[prefix + ".self_attn.linear_k.weight"];
        layer.attn_v_w = model.tensors[prefix + ".self_attn.linear_v.weight"];
        layer.attn_pos_w = model.tensors[prefix + ".self_attn.linear_pos.weight"];
        layer.attn_out_w = model.tensors[prefix + ".self_attn.linear_out.weight"];
        layer.pos_bias_u = model.tensors[prefix + ".self_attn.pos_bias_u"];
        layer.pos_bias_v = model.tensors[prefix + ".self_attn.pos_bias_v"];

        layer.norm_conv_w = model.tensors[prefix + ".norm_conv.weight"];
        layer.norm_conv_b = model.tensors[prefix + ".norm_conv.bias"];
        layer.conv_pw1_w = model.tensors[prefix + ".conv.pointwise_conv1.weight"];
        layer.conv_dw_w = model.tensors[prefix + ".conv.depthwise_conv.weight"];
        layer.conv_ln_w = model.tensors[prefix + ".conv.batch_norm.weight"];
        layer.conv_ln_b = model.tensors[prefix + ".conv.batch_norm.bias"];
        layer.conv_pw2_w = model.tensors[prefix + ".conv.pointwise_conv2.weight"];

        layer.norm_ff2_w = model.tensors[prefix + ".norm_feed_forward2.weight"];
        layer.norm_ff2_b = model.tensors[prefix + ".norm_feed_forward2.bias"];
        layer.ffn2_linear1_w = model.tensors[prefix + ".feed_forward2.linear1.weight"];
        layer.ffn2_linear2_w = model.tensors[prefix + ".feed_forward2.linear2.weight"];

        layer.norm_final_w = model.tensors[prefix + ".norm_out.weight"];
        layer.norm_final_b = model.tensors[prefix + ".norm_out.bias"];

        // Infer kernel_size from first layer's depthwise conv weight
        if (i == 0 && layer.conv_dw_w) {
            // Weight shape in GGML is [kernel_size, 1, d_model]
            model.hparams.kernel_size = layer.conv_dw_w->ne[0];
        }
    }

    // Encoder final (note: this model doesn't have separate final norm or fc after conformer layers)
    // The encoder output goes directly through joint.enc for projection
    model.encoder.final_norm_w = nullptr;
    model.encoder.final_norm_b = nullptr;
    model.encoder.fc_w = nullptr;

    // Decoder (2-layer LSTM)
    model.decoder.embedding = model.tensors["decoder.prediction.embed.weight"];
    // LSTM layer 0
    model.decoder.lstm_w_ih_l0 = model.tensors["decoder.prediction.dec_rnn.lstm.weight_ih_l0"];
    model.decoder.lstm_w_hh_l0 = model.tensors["decoder.prediction.dec_rnn.lstm.weight_hh_l0"];
    model.decoder.lstm_b_ih_l0 = model.tensors["decoder.prediction.dec_rnn.lstm.bias_ih_l0"];
    model.decoder.lstm_b_hh_l0 = model.tensors["decoder.prediction.dec_rnn.lstm.bias_hh_l0"];
    // LSTM layer 1
    model.decoder.lstm_w_ih_l1 = model.tensors["decoder.prediction.dec_rnn.lstm.weight_ih_l1"];
    model.decoder.lstm_w_hh_l1 = model.tensors["decoder.prediction.dec_rnn.lstm.weight_hh_l1"];
    model.decoder.lstm_b_ih_l1 = model.tensors["decoder.prediction.dec_rnn.lstm.bias_ih_l1"];
    model.decoder.lstm_b_hh_l1 = model.tensors["decoder.prediction.dec_rnn.lstm.bias_hh_l1"];

    // Joint
    model.joint.enc_w = model.tensors["joint.enc.weight"];
    model.joint.enc_b = model.tensors["joint.enc.bias"];
    model.joint.dec_w = model.tensors["joint.pred.weight"];
    model.joint.dec_b = model.tensors["joint.pred.bias"];
    model.joint.out_w = model.tensors["joint.joint_net.2.weight"];
    model.joint.out_b = model.tensors["joint.joint_net.2.bias"];

    // Cleanup GGUF context (we've copied all data)
    gguf_free(gguf_ctx);
    ggml_free(ctx_meta);

    // Verify critical tensors
    bool missing = false;
    auto check = [&](struct ggml_tensor * t, const char * name) {
        if (!t) {
            fprintf(stderr, "%s: missing tensor: %s\n", __func__, name);
            missing = true;
        }
    };

    check(model.encoder.subsampling.conv0_w, "encoder.pre_encode.conv.0.weight");
    check(model.encoder.subsampling.out_w, "encoder.pre_encode.out.weight");
    check(model.decoder.embedding, "decoder.prediction.embed.weight");
    check(model.decoder.lstm_w_ih_l0, "decoder.prediction.dec_rnn.lstm.weight_ih_l0");
    check(model.decoder.lstm_w_ih_l1, "decoder.prediction.dec_rnn.lstm.weight_ih_l1");
    check(model.joint.enc_w, "joint.enc.weight");
    check(model.joint.dec_w, "joint.pred.weight");
    check(model.joint.out_w, "joint.joint_net.2.weight");

    if (missing) {
        return false;
    }

    // printf("%s: model loaded successfully\n", __func__);
    return true;
}

struct nemo_context * nemo_init(const char * model_path) {
    struct nemo_context * ctx = new nemo_context();

    if (!nemo_model_load(model_path, ctx->model)) {
        delete ctx;
        return nullptr;
    }

    // Initialize allocator for compute graphs
    ctx->state.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->model.backend));

    return ctx;
}

void nemo_free(struct nemo_context * ctx) {
    if (ctx) {
        if (ctx->state.allocr) {
            ggml_gallocr_free(ctx->state.allocr);
        }
        if (ctx->model.buffer_w) {
            ggml_backend_buffer_free(ctx->model.buffer_w);
        }
        if (ctx->model.ctx_w) {
            ggml_free(ctx->model.ctx_w);
        }
        if (ctx->model.backend) {
            ggml_backend_free(ctx->model.backend);
        }
        delete ctx;
    }
}

// ============================================================================
// Graph building helpers
// ============================================================================

// Layer normalization: norm(x) * weight + bias
static struct ggml_tensor * build_layer_norm(
    struct ggml_context * ctx,
    struct ggml_tensor * input,
    struct ggml_tensor * weight,
    struct ggml_tensor * bias,
    float eps = 1e-5f
) {
    struct ggml_tensor * cur = ggml_norm(ctx, input, eps);
    cur = ggml_mul(ctx, cur, weight);
    cur = ggml_add(ctx, cur, bias);
    return cur;
}

// Feed-forward module: Linear -> Swish -> Linear
static struct ggml_tensor * build_ffn(
    struct ggml_context * ctx,
    struct ggml_tensor * input,
    struct ggml_tensor * linear1_w,
    struct ggml_tensor * linear2_w
) {
    // Linear1: [batch, time, d_model] -> [batch, time, d_ff]
    struct ggml_tensor * cur = ggml_mul_mat(ctx, linear1_w, input);

    // Swish activation (SiLU in ggml)
    cur = ggml_silu(ctx, cur);

    // Linear2: [batch, time, d_ff] -> [batch, time, d_model]
    cur = ggml_mul_mat(ctx, linear2_w, cur);

    return cur;
}

// GLU activation: split input in half, multiply first half by sigmoid of second half
static struct ggml_tensor * build_glu(
    struct ggml_context * ctx,
    struct ggml_tensor * input  // [channels*2, ...]
) {
    int64_t half_channels = input->ne[0] / 2;
    int64_t ne1 = input->ne[1];
    int64_t ne2 = input->ne[2];
    int64_t ne3 = input->ne[3];

    // First half
    struct ggml_tensor * a = ggml_view_4d(ctx, input,
        half_channels, ne1, ne2, ne3,
        input->nb[1], input->nb[2], input->nb[3], 0);

    // Second half
    struct ggml_tensor * b = ggml_view_4d(ctx, input,
        half_channels, ne1, ne2, ne3,
        input->nb[1], input->nb[2], input->nb[3],
        half_channels * ggml_element_size(input));

    // a * sigmoid(b)
    return ggml_mul(ctx, a, ggml_sigmoid(ctx, b));
}

// LSTM cell: returns (h_out, c_out)
static void build_lstm_cell(
    struct ggml_context * ctx,
    struct ggml_tensor * input,      // [batch, input_size]
    struct ggml_tensor * h_prev,     // [batch, hidden_size]
    struct ggml_tensor * c_prev,     // [batch, hidden_size]
    struct ggml_tensor * w_ih,       // [4*hidden_size, input_size]
    struct ggml_tensor * w_hh,       // [4*hidden_size, hidden_size]
    struct ggml_tensor * b_ih,       // [4*hidden_size]
    struct ggml_tensor * b_hh,       // [4*hidden_size]
    struct ggml_tensor ** h_out,
    struct ggml_tensor ** c_out
) {
    int64_t hidden_size = h_prev->ne[0];

    // gates = input @ W_ih.T + h_prev @ W_hh.T + b_ih + b_hh
    struct ggml_tensor * gates_i = ggml_mul_mat(ctx, w_ih, input);
    struct ggml_tensor * gates_h = ggml_mul_mat(ctx, w_hh, h_prev);
    struct ggml_tensor * gates = ggml_add(ctx, gates_i, gates_h);
    gates = ggml_add(ctx, gates, b_ih);
    gates = ggml_add(ctx, gates, b_hh);

    // Split gates: [i, f, g, o] each of size hidden_size
    struct ggml_tensor * i_gate = ggml_view_1d(ctx, gates, hidden_size, 0 * hidden_size * sizeof(float));
    struct ggml_tensor * f_gate = ggml_view_1d(ctx, gates, hidden_size, 1 * hidden_size * sizeof(float));
    struct ggml_tensor * g_gate = ggml_view_1d(ctx, gates, hidden_size, 2 * hidden_size * sizeof(float));
    struct ggml_tensor * o_gate = ggml_view_1d(ctx, gates, hidden_size, 3 * hidden_size * sizeof(float));

    // Apply activations
    i_gate = ggml_sigmoid(ctx, i_gate);
    f_gate = ggml_sigmoid(ctx, f_gate);
    g_gate = ggml_tanh(ctx, g_gate);
    o_gate = ggml_sigmoid(ctx, o_gate);

    // c = f * c_prev + i * g
    *c_out = ggml_add(ctx, ggml_mul(ctx, f_gate, c_prev), ggml_mul(ctx, i_gate, g_gate));

    // h = o * tanh(c)
    *h_out = ggml_mul(ctx, o_gate, ggml_tanh(ctx, *c_out));
}

// Relative position shift: out[i,j] = input[i, j + qlen - 1 - i]
// input: [batch, heads, qlen, pos_len] where pos_len = 2*qlen - 1
// output: [batch, heads, qlen, qlen]
// Implements NeMo's rel_shift operation for relative position attention
static struct ggml_tensor * build_rel_shift(
    struct ggml_context * ctx,
    struct ggml_tensor * input,  // [pos_len, qlen, heads, batch] (ggml reversed)
    int qlen
) {
    // In ggml layout: input is [pos_len=2*qlen-1, qlen, heads, batch]
    // We need to slice out the correct diagonal from pos_len for each query position
    // out[i,j] = input[i, j + qlen - 1 - i] => for each row i, we start from column (qlen-1-i)

    int64_t pos_len = input->ne[0];   // 2*qlen - 1
    int64_t heads = input->ne[2];
    int64_t batch = input->ne[3];

    // Implementation: pad left with one zero, reshape, drop first, slice
    // Step 1: Pad with zero column on the left
    // After pad: [pos_len+1, qlen, heads, batch]
    struct ggml_tensor * padded = ggml_pad_ext(ctx, input, 1, 0, 0, 0, 0, 0, 0, 0);

    // Step 2: Make contiguous and reshape to [qlen, pos_len+1, heads, batch]
    struct ggml_tensor * reshaped = ggml_reshape_4d(ctx, ggml_cont(ctx, padded), qlen, pos_len + 1, heads, batch);

    // Step 3: Drop first row (slice from row 1 onward)
    // After: [qlen, pos_len, heads, batch]
    struct ggml_tensor * dropped = ggml_view_4d(ctx, reshaped,
        qlen, pos_len, heads, batch,
        reshaped->nb[1], reshaped->nb[2], reshaped->nb[3],
        qlen * ggml_element_size(reshaped));  // offset by one row

    // Step 4: Make contiguous and reshape back to [pos_len, qlen, heads, batch]
    struct ggml_tensor * back = ggml_reshape_4d(ctx, ggml_cont(ctx, dropped), pos_len, qlen, heads, batch);

    // Step 5: Slice to [qlen, qlen, heads, batch] - take first qlen columns
    struct ggml_tensor * out = ggml_view_4d(ctx, back,
        qlen, qlen, heads, batch,
        back->nb[1], back->nb[2], back->nb[3], 0);

    return ggml_cont(ctx, out);  // Make contiguous for subsequent ops
}

// Full relative position multi-head attention
// input: [d_model, time, batch]
// pos_emb: [d_model, pos_len] where pos_len = 2*time - 1
// Returns: [d_model, time, batch]
static struct ggml_tensor * build_rel_pos_mha(
    struct ggml_context * ctx,
    struct ggml_tensor * input,    // [d_model, time, batch]
    struct ggml_tensor * pos_emb,  // [d_model, pos_len] precomputed
    struct ggml_tensor * q_w,      // [d_model, d_model]
    struct ggml_tensor * k_w,      // [d_model, d_model]
    struct ggml_tensor * v_w,      // [d_model, d_model]
    struct ggml_tensor * pos_w,    // [d_model, d_model]
    struct ggml_tensor * out_w,    // [d_model, d_model]
    struct ggml_tensor * bias_u,   // [d_model] = [heads * d_head]
    struct ggml_tensor * bias_v,   // [d_model] = [heads * d_head]
    int n_heads,
    int d_head
) {
    int64_t d_model = input->ne[0];
    int64_t time = input->ne[1];
    int64_t batch = input->ne[2];
    int64_t pos_len = pos_emb->ne[1];  // 2*time - 1

    // Q, K, V projections: [d_model, time, batch]
    struct ggml_tensor * q = ggml_mul_mat(ctx, q_w, input);
    struct ggml_tensor * k = ggml_mul_mat(ctx, k_w, input);
    struct ggml_tensor * v = ggml_mul_mat(ctx, v_w, input);

    // Position projection: [d_model, pos_len]
    struct ggml_tensor * pos = ggml_mul_mat(ctx, pos_w, pos_emb);

    // Reshape Q, K, V to [d_head, heads, time, batch]
    q = ggml_reshape_4d(ctx, q, d_head, n_heads, time, batch);
    k = ggml_reshape_4d(ctx, k, d_head, n_heads, time, batch);
    v = ggml_reshape_4d(ctx, v, d_head, n_heads, time, batch);

    // Reshape pos to [d_head, heads, pos_len] (no batch dimension)
    pos = ggml_reshape_3d(ctx, pos, d_head, n_heads, pos_len);

    // Permute to [d_head, time, heads, batch] for matmul
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [d_head, time, heads, batch]
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));  // [d_head, time, heads, batch]
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));  // [d_head, time, heads, batch]
    pos = ggml_cont(ctx, ggml_permute(ctx, pos, 0, 2, 1, 3));  // [d_head, pos_len, heads, 1]

    // Reshape bias_u, bias_v to [d_head, 1, heads, 1] for broadcasting
    // bias_u/v is stored as [d_head, heads] in GGML, reshape to [d_head, 1, heads, 1]
    struct ggml_tensor * bias_u_4d = ggml_reshape_4d(ctx, bias_u, d_head, 1, n_heads, 1);
    struct ggml_tensor * bias_v_4d = ggml_reshape_4d(ctx, bias_v, d_head, 1, n_heads, 1);

    // q_u = q + bias_u, q_v = q + bias_v
    struct ggml_tensor * q_u = ggml_add(ctx, q, bias_u_4d);
    struct ggml_tensor * q_v = ggml_add(ctx, q, bias_v_4d);

    // Content attention: q_u @ k^T -> [time, time, heads, batch]
    // ggml_mul_mat(A, B) = B @ A^T, so mul_mat(k, q_u) = q_u @ k^T
    struct ggml_tensor * content_attn = ggml_mul_mat(ctx, k, q_u);

    // Position attention: q_v @ pos^T -> [pos_len, time, heads, batch]
    // Broadcast pos across batch dimension
    struct ggml_tensor * pos_attn_raw = ggml_mul_mat(ctx, pos, q_v);

    // Rel shift to align position indices
    struct ggml_tensor * pos_attn = build_rel_shift(ctx, pos_attn_raw, time);

    // Combine: attn_scores = (content + pos) * scale
    float scale = 1.0f / std::sqrt((float)d_head);
    struct ggml_tensor * attn_scores = ggml_add(ctx, content_attn, pos_attn);
    attn_scores = ggml_scale(ctx, attn_scores, scale);

    // Softmax over key dimension (ne[0] = time)
    struct ggml_tensor * attn_weights = ggml_soft_max(ctx, attn_scores);

    // Apply attention to values: context = attn_weights @ v
    // attn_weights: [time(j), time(i), heads, batch], v: [d_head, time(j), heads, batch]
    // Want: context[d, i, h, b] = sum_j attn_weights[j, i, h, b] * v[d, j, h, b]
    // Permute v to [time, d_head, heads, batch], then mul_mat(v_perm, attn_weights)
    struct ggml_tensor * v_perm = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
    // Result: [d_head, time, heads, batch]
    struct ggml_tensor * context = ggml_mul_mat(ctx, v_perm, attn_weights);

    // Permute to [d_head, heads, time, batch] for output reshape
    context = ggml_cont(ctx, ggml_permute(ctx, context, 0, 2, 1, 3));

    // Reshape to [d_model, time, batch]
    context = ggml_reshape_3d(ctx, context, d_model, time, batch);

    // Output projection
    struct ggml_tensor * out = ggml_mul_mat(ctx, out_w, context);

    return out;
}

// Conformer Convolution Module
// input: [d_model, time, batch]
// Returns: [d_model, time, batch]
static struct ggml_tensor * build_conformer_conv(
    struct ggml_context * ctx,
    struct ggml_tensor * input,     // [d_model, time, batch]
    struct ggml_tensor * pw1_w,     // [1, d_model, 2*d_model] pointwise conv1
    struct ggml_tensor * dw_w,      // [kernel_size, 1, d_model] depthwise conv
    struct ggml_tensor * ln_w,      // [d_model] layer norm weight
    struct ggml_tensor * ln_b,      // [d_model] layer norm bias
    struct ggml_tensor * pw2_w,     // [1, d_model, d_model] pointwise conv2
    int kernel_size
) {
    int64_t d_model = input->ne[0];
    int64_t seq_len = input->ne[1];
    int64_t batch = input->ne[2];

    // Pointwise Conv1: [d_model, seq_len, batch] -> [2*d_model, seq_len, batch]
    // pw1_w is [1, d_model, 2*d_model], reshape to [d_model, 2*d_model] for matmul
    struct ggml_tensor * pw1_w_2d = ggml_reshape_2d(ctx, pw1_w, d_model, 2 * d_model);
    struct ggml_tensor * cur = ggml_mul_mat(ctx, pw1_w_2d, input);

    // GLU: split in half, multiply first half by sigmoid of second half
    // cur: [2*d_model, seq_len, batch] -> [d_model, seq_len, batch]
    int64_t half_ch = d_model;
    int64_t full_ch = 2 * d_model;
    size_t nb1 = full_ch * sizeof(float);
    size_t nb2 = full_ch * seq_len * sizeof(float);

    struct ggml_tensor * glu_a = ggml_view_3d(ctx, cur, half_ch, seq_len, batch, nb1, nb2, 0);
    struct ggml_tensor * glu_b = ggml_view_3d(ctx, cur, half_ch, seq_len, batch, nb1, nb2, half_ch * sizeof(float));
    cur = ggml_mul(ctx, glu_a, ggml_sigmoid(ctx, glu_b));
    cur = ggml_cont(ctx, cur);

    // Transpose to [seq_len, d_model, batch] for conv1d
    cur = ggml_cont(ctx, ggml_permute(ctx, cur, 1, 0, 2, 3));

    // Causal padding: pad left by kernel_size-1
    cur = ggml_pad_ext(ctx, cur, kernel_size - 1, 0, 0, 0, 0, 0, 0, 0);

    // Depthwise conv1d: for each position t in output, sum over k:
    // output[c, t] = sum_k input[c, t+k] * weight[c, k]
    // Weight is [kernel_size, 1, d_model], reshape to [kernel_size, d_model]
    struct ggml_tensor * dw_w_2d = ggml_reshape_2d(ctx, dw_w, kernel_size, d_model);
    struct ggml_tensor * dw_w_t = ggml_cont(ctx, ggml_transpose(ctx, dw_w_2d));
    // dw_w_t: [d_model, kernel_size]

    struct ggml_tensor * conv_result = nullptr;
    for (int k = 0; k < kernel_size; k++) {
        // Extract slice at offset k
        struct ggml_tensor * input_slice = ggml_view_3d(ctx, cur,
            seq_len, d_model, batch,
            cur->nb[1], cur->nb[2],
            k * sizeof(float));

        // Get k-th kernel element for each channel
        struct ggml_tensor * kernel_k = ggml_view_1d(ctx, dw_w_t, d_model, k * d_model * sizeof(float));
        kernel_k = ggml_reshape_3d(ctx, kernel_k, 1, d_model, 1);

        // Multiply and accumulate
        struct ggml_tensor * product = ggml_mul(ctx, input_slice, kernel_k);
        if (conv_result == nullptr) {
            conv_result = product;
        } else {
            conv_result = ggml_add(ctx, conv_result, product);
        }
    }
    cur = conv_result;

    // Transpose back to [d_model, seq_len, batch]
    cur = ggml_cont(ctx, ggml_permute(ctx, cur, 1, 0, 2, 3));

    // Layer norm
    cur = ggml_norm(ctx, cur, 1e-5f);
    cur = ggml_mul(ctx, cur, ln_w);
    cur = ggml_add(ctx, cur, ln_b);

    // Swish
    cur = ggml_silu(ctx, cur);

    // Pointwise Conv2: [d_model, seq_len, batch] -> [d_model, seq_len, batch]
    struct ggml_tensor * pw2_w_2d = ggml_reshape_2d(ctx, pw2_w, d_model, d_model);
    cur = ggml_mul_mat(ctx, pw2_w_2d, cur);

    return cur;
}

// Full Conformer Layer
// Structure: x -> LN -> FFN1 -> +x*0.5 -> LN -> Attn -> +x -> LN -> Conv -> +x -> LN -> FFN2 -> +x*0.5 -> LN
// input: [d_model, time, batch]
// pos_emb: [d_model, pos_len]
// Returns: [d_model, time, batch]
struct ggml_tensor * build_conformer_layer(
    struct ggml_context * ctx,
    struct ggml_tensor * input,     // [d_model, time, batch]
    struct ggml_tensor * pos_emb,   // [d_model, pos_len] precomputed
    nemo_conformer_layer * layer,   // layer weights
    int n_heads,
    int d_head,
    int kernel_size
) {
    struct ggml_tensor * residual = input;
    struct ggml_tensor * cur;

    // 1. FFN1 path: LN -> FFN1 -> residual * 0.5
    cur = build_layer_norm(ctx, residual, layer->norm_ff1_w, layer->norm_ff1_b);
    cur = build_ffn(ctx, cur, layer->ffn1_linear1_w, layer->ffn1_linear2_w);
    cur = ggml_scale(ctx, cur, 0.5f);
    residual = ggml_add(ctx, residual, cur);

    // 2. Self-attention path: LN -> Attn -> residual
    cur = build_layer_norm(ctx, residual, layer->norm_attn_w, layer->norm_attn_b);
    cur = build_rel_pos_mha(ctx, cur, pos_emb,
                            layer->attn_q_w, layer->attn_k_w, layer->attn_v_w,
                            layer->attn_pos_w, layer->attn_out_w,
                            layer->pos_bias_u, layer->pos_bias_v,
                            n_heads, d_head);
    residual = ggml_add(ctx, residual, cur);

    // 3. Conv path: LN -> Conv -> residual
    cur = build_layer_norm(ctx, residual, layer->norm_conv_w, layer->norm_conv_b);
    cur = build_conformer_conv(ctx, cur,
                               layer->conv_pw1_w, layer->conv_dw_w,
                               layer->conv_ln_w, layer->conv_ln_b,
                               layer->conv_pw2_w, kernel_size);
    residual = ggml_add(ctx, residual, cur);

    // 4. FFN2 path: LN -> FFN2 -> residual * 0.5
    cur = build_layer_norm(ctx, residual, layer->norm_ff2_w, layer->norm_ff2_b);
    cur = build_ffn(ctx, cur, layer->ffn2_linear1_w, layer->ffn2_linear2_w);
    cur = ggml_scale(ctx, cur, 0.5f);
    residual = ggml_add(ctx, residual, cur);

    // 5. Final layer norm
    cur = build_layer_norm(ctx, residual, layer->norm_final_w, layer->norm_final_b);

    return cur;
}

// ============================================================================
// ConvSubsampling
// ============================================================================

// Helper: build causal conv2d with asymmetric padding
static struct ggml_tensor * build_causal_conv2d(
    struct ggml_context * ctx,
    struct ggml_tensor * input,   // [W, H, C, N]
    struct ggml_tensor * weight,  // [KW, KH, IC, OC]
    struct ggml_tensor * bias,    // [OC]
    int stride_w, int stride_h
) {
    // Causal padding: pad_left = kW-1, pad_right = stride-1, pad_top = kH-1, pad_bottom = stride-1
    int kW = weight->ne[0];
    int kH = weight->ne[1];
    int pad_left = kW - 1;
    int pad_right = stride_w - 1;
    int pad_top = kH - 1;
    int pad_bottom = stride_h - 1;

    struct ggml_tensor * padded = ggml_pad_ext(ctx, input, pad_left, pad_right, pad_top, pad_bottom, 0, 0, 0, 0);
    struct ggml_tensor * conv_out = ggml_conv_2d(ctx, weight, padded, stride_w, stride_h, 0, 0, 1, 1);

    // Add bias
    int out_channels = weight->ne[3];
    struct ggml_tensor * bias_reshaped = ggml_reshape_4d(ctx, bias, 1, 1, out_channels, 1);
    return ggml_add(ctx, conv_out, bias_reshaped);
}

// Helper: build causal depthwise conv2d using ggml_conv_2d_dw_direct (F32)
static struct ggml_tensor * build_causal_dw_conv2d(
    struct ggml_context * ctx,
    struct ggml_tensor * input,   // [W, H, C, N]
    struct ggml_tensor * weight,  // [KW, KH, 1, C]
    struct ggml_tensor * bias,    // [C]
    int stride_w, int stride_h
) {
    int kW = weight->ne[0];
    int kH = weight->ne[1];
    int pad_left = kW - 1;
    int pad_right = stride_w - 1;
    int pad_top = kH - 1;
    int pad_bottom = stride_h - 1;

    struct ggml_tensor * padded = ggml_pad_ext(ctx, input, pad_left, pad_right, pad_top, pad_bottom, 0, 0, 0, 0);
    struct ggml_tensor * conv_out = ggml_conv_2d_dw_direct(ctx, weight, padded, stride_w, stride_h, 0, 0, 1, 1);

    // Add bias
    int channels = weight->ne[3];
    struct ggml_tensor * bias_reshaped = ggml_reshape_4d(ctx, bias, 1, 1, channels, 1);
    return ggml_add(ctx, conv_out, bias_reshaped);
}

// Build ConvSubsampling module (depthwise-separable architecture)
// mel: [n_mels, time, batch] -> [d_model, time/8, batch]
struct ggml_tensor * build_conv_subsampling(
    struct ggml_context * ctx,
    struct ggml_tensor * mel,           // [n_mels, time, batch]
    nemo_conv_subsampling * sub         // weights
) {
    int64_t n_mels = mel->ne[0];
    int64_t time_in = mel->ne[1];
    int64_t batch = mel->ne[2];

    // Reshape to 4D for conv2d: [n_mels, time, 1, batch]
    struct ggml_tensor * cur = ggml_reshape_4d(ctx, mel, n_mels, time_in, 1, batch);

    // Conv0: CausalConv2D(1, 256, k=3, s=2) + ReLU
    cur = build_causal_conv2d(ctx, cur, sub->conv0_w, sub->conv0_b, 2, 2);
    cur = ggml_relu(ctx, cur);

    // Conv2: Depthwise CausalConv2D(256, k=3, s=2, groups=256)
    cur = build_causal_dw_conv2d(ctx, cur, sub->conv2_w, sub->conv2_b, 2, 2);

    // Conv3: Pointwise Conv2D(256, 256, k=1, s=1) + ReLU
    cur = ggml_conv_2d(ctx, sub->conv3_w, cur, 1, 1, 0, 0, 1, 1);
    struct ggml_tensor * conv3_b_reshaped = ggml_reshape_4d(ctx, sub->conv3_b, 1, 1, sub->conv3_b->ne[0], 1);
    cur = ggml_add(ctx, cur, conv3_b_reshaped);
    cur = ggml_relu(ctx, cur);

    // Conv5: Depthwise CausalConv2D(256, k=3, s=2, groups=256)
    cur = build_causal_dw_conv2d(ctx, cur, sub->conv5_w, sub->conv5_b, 2, 2);

    // Conv6: Pointwise Conv2D(256, 256, k=1, s=1) + ReLU
    cur = ggml_conv_2d(ctx, sub->conv6_w, cur, 1, 1, 0, 0, 1, 1);
    struct ggml_tensor * conv6_b_reshaped = ggml_reshape_4d(ctx, sub->conv6_b, 1, 1, sub->conv6_b->ne[0], 1);
    cur = ggml_add(ctx, cur, conv6_b_reshaped);
    cur = ggml_relu(ctx, cur);

    // cur shape: [W_out, H_out, 256, batch] where W_out ~= n_mels/8, H_out ~= time/8
    int64_t w_out = cur->ne[0];  // ~17 for n_mels=128
    int64_t h_out = cur->ne[1];  // ~time/8
    int64_t c_out = cur->ne[2];  // 256

    // Permute to [W, C, H, N] then reshape to [W*C, H, N] for linear projection
    // This matches original C++ flatten order: flat[c * W + w]
    struct ggml_tensor * permuted = ggml_cont(ctx, ggml_permute(ctx, cur, 0, 2, 1, 3));
    permuted = ggml_reshape_3d(ctx, permuted, w_out * c_out, h_out, batch);

    // Linear projection: [W*C, H, N] -> [d_model, H, N]
    cur = ggml_mul_mat(ctx, sub->out_w, permuted);

    // Add bias
    cur = ggml_add(ctx, cur, sub->out_b);

    return cur;  // [d_model, time_out, batch]
}

// ============================================================================
// Full Encoder
// ============================================================================

// Build full encoder: ConvSubsampling + 24 Conformer layers
// mel: [n_mels, time, batch]
// Returns: [d_model, time/8, batch]
struct ggml_tensor * build_encoder(
    struct ggml_context * ctx,
    struct ggml_tensor * mel,       // [n_mels, time, batch]
    nemo_model * model              // model with all weights
) {
    const int n_heads = model->hparams.n_heads;     // 8
    const int d_head = model->hparams.d_head;       // 128
    const int kernel_size = model->hparams.kernel_size;  // 31
    const int n_layers = model->hparams.n_layers;   // 24
    const int d_model = model->hparams.d_model;     // 1024

    // 1. ConvSubsampling: [n_mels, time, batch] -> [d_model, time/8, batch]
    struct ggml_tensor * cur = build_conv_subsampling(ctx, mel, &model->encoder.subsampling);

    int64_t seq_len = cur->ne[1];  // time/8
    int64_t batch = cur->ne[2];

    // 2. Get positional embeddings for this sequence length
    // pos_emb is precomputed as [d_model, 2*max_len-1], we need to slice for current seq_len
    // pos_len needed = 2*seq_len - 1
    int64_t pos_len = 2 * seq_len - 1;
    int64_t max_pos_len = model->pos_emb->ne[1];  // 2*512-1 = 1023

    // Calculate offset to center the slice (for positions [-(seq_len-1), seq_len-1])
    int64_t pos_offset = (max_pos_len - pos_len) / 2;

    struct ggml_tensor * pos_emb = ggml_view_2d(ctx, model->pos_emb,
        d_model, pos_len,
        model->pos_emb->nb[1],
        pos_offset * model->pos_emb->nb[1]);

    // 3. Process through all conformer layers
    for (int i = 0; i < n_layers; i++) {
        cur = build_conformer_layer(ctx, cur, pos_emb,
                                    &model->encoder.layers[i],
                                    n_heads, d_head, kernel_size);
    }

    return cur;  // [d_model, time/8, batch]
}

// ============================================================================
// Decoder (RNNT Prediction Network)
// ============================================================================

// Build decoder step: embedding + 2-layer LSTM
// token: input token id
// h_in: [2 * hidden_size] concatenated hidden states for both layers
// c_in: [2 * hidden_size] concatenated cell states for both layers
// Returns: decoder output [hidden_size], and updated h_out, c_out
struct ggml_tensor * build_decoder_step(
    struct ggml_context * ctx,
    struct ggml_tensor * token_emb,     // [hidden_size] token embedding
    struct ggml_tensor * h_in,          // [2 * hidden_size]
    struct ggml_tensor * c_in,          // [2 * hidden_size]
    nemo_decoder * decoder,
    struct ggml_tensor ** h_out,        // output: [2 * hidden_size]
    struct ggml_tensor ** c_out         // output: [2 * hidden_size]
) {
    const int64_t hidden_size = nemo_decoder::HIDDEN_SIZE;  // 640

    // Split input states into per-layer states
    struct ggml_tensor * h0_in = ggml_view_1d(ctx, h_in, hidden_size, 0);
    struct ggml_tensor * c0_in = ggml_view_1d(ctx, c_in, hidden_size, 0);
    struct ggml_tensor * h1_in = ggml_view_1d(ctx, h_in, hidden_size, hidden_size * sizeof(float));
    struct ggml_tensor * c1_in = ggml_view_1d(ctx, c_in, hidden_size, hidden_size * sizeof(float));

    // Layer 0: input is token embedding
    struct ggml_tensor * h0_out = nullptr;
    struct ggml_tensor * c0_out = nullptr;
    build_lstm_cell(ctx, token_emb, h0_in, c0_in,
                    decoder->lstm_w_ih_l0, decoder->lstm_w_hh_l0,
                    decoder->lstm_b_ih_l0, decoder->lstm_b_hh_l0,
                    &h0_out, &c0_out);

    // Layer 1: input is h0_out
    struct ggml_tensor * h1_out = nullptr;
    struct ggml_tensor * c1_out = nullptr;
    build_lstm_cell(ctx, h0_out, h1_in, c1_in,
                    decoder->lstm_w_ih_l1, decoder->lstm_w_hh_l1,
                    decoder->lstm_b_ih_l1, decoder->lstm_b_hh_l1,
                    &h1_out, &c1_out);

    // Concatenate output states
    *h_out = ggml_concat(ctx, h0_out, h1_out, 0);  // [2 * hidden_size]
    *c_out = ggml_concat(ctx, c0_out, c1_out, 0);  // [2 * hidden_size]

    // Decoder output is the hidden state of the last layer
    return h1_out;  // [hidden_size]
}

// Embedding lookup using ggml_get_rows
// token_id: scalar token ID
// Returns: [hidden_size] embedding vector
static struct ggml_tensor * build_embedding_lookup(
    struct ggml_context * ctx,
    int token_id,
    struct ggml_tensor * embedding   // [hidden_size, vocab_size] in GGML layout
) {
    // Create index tensor
    struct ggml_tensor * idx = ggml_new_i32(ctx, token_id);

    // ggml_get_rows expects embedding in [hidden_size, vocab_size] layout
    // and returns [hidden_size, n_rows]
    struct ggml_tensor * emb = ggml_get_rows(ctx, embedding, idx);

    // Reshape to [hidden_size]
    return ggml_reshape_1d(ctx, emb, embedding->ne[0]);
}

// ============================================================================
// Joint Network
// ============================================================================

// Build joint network: encoder_out + decoder_out -> logits
// encoder_out: [d_model] or [d_model, time] single frame or multiple frames
// decoder_out: [hidden_size]
// Returns: [vocab_size] logits
struct ggml_tensor * build_joint(
    struct ggml_context * ctx,
    struct ggml_tensor * encoder_out,   // [d_model] or [d_model, 1]
    struct ggml_tensor * decoder_out,   // [hidden_size]
    nemo_joint * joint
) {
    // Ensure encoder_out is 1D [d_model]
    struct ggml_tensor * enc = encoder_out;
    if (enc->ne[1] > 1 || enc->ne[2] > 1) {
        // Multiple time steps or batch, reshape to just the first element
        enc = ggml_view_1d(ctx, enc, enc->ne[0], 0);
    } else if (enc->ne[1] == 1) {
        // [d_model, 1] -> [d_model]
        enc = ggml_reshape_1d(ctx, enc, enc->ne[0]);
    }

    // Project encoder: [d_model] -> [joint_dim]
    // enc_w is [joint_dim, d_model], mul_mat computes enc @ enc_w.T = [joint_dim]
    struct ggml_tensor * enc_proj = ggml_mul_mat(ctx, joint->enc_w, enc);
    enc_proj = ggml_add(ctx, enc_proj, joint->enc_b);

    // Project decoder: [hidden_size] -> [joint_dim]
    struct ggml_tensor * dec_proj = ggml_mul_mat(ctx, joint->dec_w, decoder_out);
    dec_proj = ggml_add(ctx, dec_proj, joint->dec_b);

    // Add and ReLU
    struct ggml_tensor * joint_out = ggml_add(ctx, enc_proj, dec_proj);
    joint_out = ggml_relu(ctx, joint_out);

    // Output projection: [joint_dim] -> [vocab_size]
    struct ggml_tensor * logits = ggml_mul_mat(ctx, joint->out_w, joint_out);
    logits = ggml_add(ctx, logits, joint->out_b);

    return logits;  // [vocab_size]
}

// ============================================================================
// Greedy Decoding
// ============================================================================

// Run greedy RNN-T decoding
// encoder_out: [d_model, time, batch] - output from build_encoder
// Returns: vector of token IDs
std::vector<int> greedy_decode(
    struct nemo_context * nctx,
    struct ggml_tensor * encoder_out,  // [d_model, time, batch]
    ggml_backend_t backend
) {
    std::vector<int> tokens;

    const int d_model = encoder_out->ne[0];       // 1024
    const int time_steps = encoder_out->ne[1];
    const int hidden_size = nemo_decoder::HIDDEN_SIZE;  // 640
    const int num_layers = nemo_decoder::NUM_LAYERS;    // 2
    const int vocab_size = nctx->model.hparams.vocab_size;  // 1025
    const int blank_token = vocab_size - 1;  // 1024

    // Get encoder data to CPU
    std::vector<float> enc_data(d_model * time_steps);
    ggml_backend_tensor_get(encoder_out, enc_data.data(), 0, enc_data.size() * sizeof(float));

    // Initialize decoder state
    std::vector<float> h_state(num_layers * hidden_size, 0.0f);
    std::vector<float> c_state(num_layers * hidden_size, 0.0f);

    int prev_token = blank_token;

    // Max symbols per step (prevent infinite loops)
    const int MAX_SYMBOLS_PER_STEP = 10;

    bool debug = false;  // Set to true for debug output

    for (int t = 0; t < time_steps; t++) {
        // Extract encoder frame at time t
        const float * enc_frame = enc_data.data() + t * d_model;

        for (int sym = 0; sym < MAX_SYMBOLS_PER_STEP; sym++) {
            // Create compute context for this step
            size_t buf_size = ggml_tensor_overhead() * 100 + ggml_graph_overhead();
            std::vector<uint8_t> compute_buf(buf_size);

            struct ggml_init_params params = {
                /*.mem_size   =*/ buf_size,
                /*.mem_buffer =*/ compute_buf.data(),
                /*.no_alloc   =*/ true,
            };

            struct ggml_context * ctx0 = ggml_init(params);
            if (!ctx0) break;

            // Create input tensors
            struct ggml_tensor * h_in = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, num_layers * hidden_size);
            struct ggml_tensor * c_in = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, num_layers * hidden_size);
            struct ggml_tensor * token_emb = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hidden_size);
            struct ggml_tensor * enc_in = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, d_model);
            ggml_set_input(h_in);
            ggml_set_input(c_in);
            ggml_set_input(token_emb);
            ggml_set_input(enc_in);

            // Build decoder step
            struct ggml_tensor * h_out = nullptr;
            struct ggml_tensor * c_out = nullptr;
            struct ggml_tensor * dec_out = build_decoder_step(ctx0, token_emb, h_in, c_in,
                                                              &nctx->model.decoder, &h_out, &c_out);

            // Build joint
            struct ggml_tensor * logits = build_joint(ctx0, enc_in, dec_out, &nctx->model.joint);
            ggml_set_output(logits);
            ggml_set_output(h_out);
            ggml_set_output(c_out);

            // Build graph
            struct ggml_cgraph * gf = ggml_new_graph(ctx0);
            ggml_build_forward_expand(gf, logits);
            ggml_build_forward_expand(gf, h_out);
            ggml_build_forward_expand(gf, c_out);

            // Allocate
            ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(allocr, gf)) {
                ggml_gallocr_free(allocr);
                ggml_free(ctx0);
                break;
            }

            // Set inputs
            ggml_backend_tensor_set(h_in, h_state.data(), 0, h_state.size() * sizeof(float));
            ggml_backend_tensor_set(c_in, c_state.data(), 0, c_state.size() * sizeof(float));

            // Get embedding for prev_token
            std::vector<float> emb_data(hidden_size);
            size_t emb_offset = prev_token * hidden_size * sizeof(float);
            ggml_backend_tensor_get(nctx->model.decoder.embedding, emb_data.data(), emb_offset, hidden_size * sizeof(float));
            ggml_backend_tensor_set(token_emb, emb_data.data(), 0, hidden_size * sizeof(float));

            ggml_backend_tensor_set(enc_in, enc_frame, 0, d_model * sizeof(float));

            // Compute
            ggml_backend_graph_compute(backend, gf);

            // Get logits and find argmax
            std::vector<float> logits_data(vocab_size);
            ggml_backend_tensor_get(logits, logits_data.data(), 0, vocab_size * sizeof(float));

            int best_token = 0;
            float best_score = logits_data[0];
            for (int v = 1; v < vocab_size; v++) {
                if (logits_data[v] > best_score) {
                    best_score = logits_data[v];
                    best_token = v;
                }
            }

            // Get updated state BEFORE freeing (we'll decide whether to keep it)
            std::vector<float> new_h_state(h_state.size());
            std::vector<float> new_c_state(c_state.size());
            ggml_backend_tensor_get(h_out, new_h_state.data(), 0, new_h_state.size() * sizeof(float));
            ggml_backend_tensor_get(c_out, new_c_state.data(), 0, new_c_state.size() * sizeof(float));

            ggml_gallocr_free(allocr);
            ggml_free(ctx0);

            if (debug && t < 5) {
                printf("  t=%d sym=%d: prev_token=%d, best_token=%d (score=%.4f), blank_score=%.4f\n",
                       t, sym, prev_token, best_token, best_score, logits_data[blank_token]);
            }

            if (best_token == blank_token) {
                // Move to next time step - DON'T update state
                break;
            }

            // Emit non-blank token
            tokens.push_back(best_token);
            prev_token = best_token;

            // Only update LSTM state when emitting a non-blank token
            h_state = std::move(new_h_state);
            c_state = std::move(new_c_state);
        }
    }

    if (debug) {
        printf("Total tokens emitted: %zu\n", tokens.size());
    }

    return tokens;
}

// Convert token IDs to text
std::string tokens_to_text(const std::vector<int> & tokens, const std::vector<char8> & vocab) {
    std::string result;
    for (int token : tokens) {
        if (token >= 0 && token < (int)vocab.size()) {
            std::string piece(vocab[token].data);
            // SentencePiece convention: _ means space at start of word
            if (!piece.empty() && piece[0] == '\xe2' && piece.size() >= 3 &&
                piece[1] == '\x96' && piece[2] == '\x81') {
                // UTF-8 for ▁ (U+2581)
                if (!result.empty()) result += ' ';
                result += piece.substr(3);
            } else {
                result += piece;
            }
        }
    }
    return result;
}
