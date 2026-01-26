#include "nemo-ggml.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>

// Maximum nodes in computation graph
#define NEMO_MAX_NODES 8192

// Compute positional embeddings (sinusoidal)
static void compute_pos_emb(float * data, int max_len, int d_model) {
    int total_len = 2 * max_len - 1;

    for (int pos = 0; pos < total_len; pos++) {
        float p = (float)pos - (float)(max_len - 1);

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
    printf("%s: loading model from '%s'\n", __func__, path.c_str());

    // Initialize backend first
    model.backend = ggml_backend_cpu_init();
    if (!model.backend) {
        fprintf(stderr, "%s: failed to init CPU backend\n", __func__);
        return false;
    }

    // Open GGUF file
    struct ggml_context * ctx_meta = nullptr;
    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &ctx_meta,
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
    kv_idx = gguf_find_key(gguf_ctx, "tokenizer.vocab");
    if (kv_idx >= 0) {
        const char * vocab_data = gguf_get_val_str(gguf_ctx, kv_idx);
        size_t vocab_bytes = gguf_get_arr_n(gguf_ctx, kv_idx);
        size_t vocab_entry_size = 8; // char8
        size_t n_tokens = vocab_bytes / vocab_entry_size;
        model.vocab.resize(n_tokens + 1, {0}); // +1 for empty token
        memcpy(model.vocab.data(), vocab_data, vocab_bytes);
    }

    printf("%s: n_mels     = %d\n", __func__, model.hparams.n_mels);
    printf("%s: d_model    = %d\n", __func__, model.hparams.d_model);
    printf("%s: n_heads    = %d\n", __func__, model.hparams.n_heads);
    printf("%s: n_layers   = %d\n", __func__, model.hparams.n_layers);
    printf("%s: vocab_size = %d\n", __func__, model.hparams.vocab_size);
    printf("%s: vocab tokens = %zu\n", __func__, model.vocab.size());

    // Get tensor count
    int64_t n_tensors = gguf_get_n_tensors(gguf_ctx);
    printf("%s: n_tensors  = %lld\n", __func__, (long long)n_tensors);

    // Create weight context with no_alloc=true
    size_t ctx_size = ggml_tensor_overhead() * (n_tensors + 10);  // +10 for pos_emb
    struct ggml_init_params ctx_params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
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
    model.encoder.subsampling.conv1_w = model.tensors["encoder.pre_encode.conv.0.weight"];
    model.encoder.subsampling.conv1_b = model.tensors["encoder.pre_encode.conv.0.bias"];
    model.encoder.subsampling.conv2_w = model.tensors["encoder.pre_encode.conv.2.weight"];
    model.encoder.subsampling.conv2_b = model.tensors["encoder.pre_encode.conv.2.bias"];
    model.encoder.subsampling.conv3_w = model.tensors["encoder.pre_encode.conv.4.weight"];
    model.encoder.subsampling.conv3_b = model.tensors["encoder.pre_encode.conv.4.bias"];
    model.encoder.subsampling.out_w   = model.tensors["encoder.pre_encode.out.weight"];

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
        layer.conv_pw1_w = model.tensors[prefix + ".conv_module.pointwise_conv1.weight"];
        layer.conv_dw_w = model.tensors[prefix + ".conv_module.depthwise_conv.weight"];
        layer.conv_ln_w = model.tensors[prefix + ".conv_module.batch_norm.weight"];
        layer.conv_ln_b = model.tensors[prefix + ".conv_module.batch_norm.bias"];
        layer.conv_pw2_w = model.tensors[prefix + ".conv_module.pointwise_conv2.weight"];

        layer.norm_ff2_w = model.tensors[prefix + ".norm_feed_forward2.weight"];
        layer.norm_ff2_b = model.tensors[prefix + ".norm_feed_forward2.bias"];
        layer.ffn2_linear1_w = model.tensors[prefix + ".feed_forward2.linear1.weight"];
        layer.ffn2_linear2_w = model.tensors[prefix + ".feed_forward2.linear2.weight"];

        layer.norm_final_w = model.tensors[prefix + ".norm_out.weight"];
        layer.norm_final_b = model.tensors[prefix + ".norm_out.bias"];
    }

    // Encoder final (note: this model doesn't have separate final norm or fc after conformer layers)
    // The encoder output goes directly through joint.enc for projection
    model.encoder.final_norm_w = nullptr;
    model.encoder.final_norm_b = nullptr;
    model.encoder.fc_w = nullptr;

    // Decoder
    model.decoder.embedding = model.tensors["decoder.prediction.embed.weight"];
    model.decoder.lstm_w_ih = model.tensors["decoder.prediction.dec_rnn.lstm.weight_ih_l0"];
    model.decoder.lstm_w_hh = model.tensors["decoder.prediction.dec_rnn.lstm.weight_hh_l0"];
    model.decoder.lstm_b_ih = model.tensors["decoder.prediction.dec_rnn.lstm.bias_ih_l0"];
    model.decoder.lstm_b_hh = model.tensors["decoder.prediction.dec_rnn.lstm.bias_hh_l0"];
    model.decoder.fc_w = model.tensors["decoder.prediction.fc.weight"];

    // Joint
    model.joint.enc_w = model.tensors["joint.enc.weight"];
    model.joint.dec_w = model.tensors["joint.pred.weight"];
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

    check(model.encoder.subsampling.conv1_w, "encoder.pre_encode.conv.0.weight");
    check(model.encoder.subsampling.out_w, "encoder.pre_encode.out.weight");
    check(model.decoder.embedding, "decoder.prediction.embed.weight");
    check(model.joint.out_w, "joint.joint_net.2.weight");

    if (missing) {
        return false;
    }

    printf("%s: model loaded successfully\n", __func__);
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
