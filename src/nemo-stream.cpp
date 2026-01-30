#include "nemo-stream.h"
#include "preprocessor.h"

#include <cassert>
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>

// Forward declaration from nemo-ggml.cpp
std::string tokens_to_text(const std::vector<int> & tokens, const std::vector<char8> & vocab);

// nemo_decoder_state::init and reset are now inline in nemo-ggml.h

// =============================================================================
// Pre-built Encoder Graph
// =============================================================================

nemo_encoder_graph::~nemo_encoder_graph() {
    if (allocr) {
        ggml_gallocr_free(allocr);
        allocr = nullptr;
    }
    if (ctx) {
        ggml_free(ctx);
        ctx = nullptr;
    }
}

void nemo_encoder_graph::reset() {
    // Reset cache inputs to zero (don't need to rebuild graph)
    initialized = false;
}

void nemo_stream_context::init(struct nemo_context* ctx, const nemo_cache_config& cfg) {
    nctx = ctx;
    config = cfg;

    // Initialize decoder state
    decoder_state.init(cfg.decoder_layers, cfg.decoder_hidden);
    decoder_state.prev_token = cfg.blank_token;

    // Pre-build encoder graph for streaming
    // Chunk size depends on att_right_context (latency mode):
    //   [70, 0]  -> 8 mel frames  -> 80ms  latency (pure causal)
    //   [70, 1]  -> 16 mel frames -> 160ms latency
    //   [70, 6]  -> 56 mel frames -> 560ms latency
    //   [70, 13] -> 112 mel frames -> 1.12s latency
    encoder_graph.init(ctx, cfg);

    // Initialize decode graph as nullptr (built on first use)
    decode_ctx = nullptr;
    decode_graph = nullptr;
    decode_allocr = nullptr;
    decode_graph_initialized = false;

    // init mel buffer with zeros (pre_encode_cache_size frames of overlap)
    mel_buffer.clear();
    mel_buffer.resize(cfg.pre_encode_cache_size * cfg.n_mels, 0.0f);

    // Clear tokens and transcript
    tokens.clear();
    transcript.clear();

    // Reset cache validity tracking
    cache_valid_len = 0;

    // Reset chunk counter
    total_chunks_processed = 0;

    // Reset timing
    total_audio_seconds = 0;
    total_compute_seconds = 0;
    encoder_seconds = 0;
    decoder_seconds = 0;
    transfer_seconds = 0;
    total_decode_iterations = 0;
}

void nemo_stream_context::reset() {
    encoder_graph.reset();
    decoder_state.reset();
    decoder_state.prev_token = config.blank_token;

    // init mel buffer with zeros (pre_encode_cache_size frames of overlap)
    mel_buffer.clear();
    mel_buffer.resize(config.pre_encode_cache_size * config.n_mels, 0.0f);

    tokens.clear();
    transcript.clear();
    total_audio_seconds = 0;
    total_compute_seconds = 0;
    encoder_seconds = 0;
    decoder_seconds = 0;
    transfer_seconds = 0;
    total_decode_iterations = 0;
    cache_valid_len = 0;
    total_chunks_processed = 0;
    // Don't reset decode_graph - it can be reused
}

nemo_stream_context::~nemo_stream_context() {
    if (decode_allocr) {
        ggml_gallocr_free(decode_allocr);
        decode_allocr = nullptr;
    }
    if (decode_ctx) {
        ggml_free(decode_ctx);
        decode_ctx = nullptr;
    }
}

void nemo_encoder_graph::build_streaming_encoder(
    struct ggml_context * ctx,
    struct nemo_context* nctx,
    const nemo_cache_config& cfg,
    size_t drop_extra_preencoded
) {
    const int d_model = cfg.d_model;
    const int n_layers = cfg.n_layers;
    const int n_mels = cfg.n_mels;
    const int cache_len = cfg.att_left_context;
    const int conv_cache_len = cfg.conv_kernel_size - 1;
    size_t mel_chunk_frames = cfg.get_chunk_mel_frames();
    // Create input tensor for mel chunk
    mel_input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_mels, mel_chunk_frames, 1);
    ggml_set_name(mel_input, "mel_input");
    ggml_set_input(mel_input);

    // Run subsampling
    struct ggml_tensor* subsampled = build_conv_subsampling(ctx, mel_input, &nctx->model.encoder.subsampling);
    
    // Drop extra pre-encoded frames from the START (overlap with cache)
    // NeMo: audio_signal = audio_signal[:, drop_extra_pre_encoded:, :]
    if (drop_extra_preencoded > 0) {
        struct ggml_tensor * s = subsampled;
        // printf("%ld %ld %ld %ld\n",  s->ne[0], s->ne[1], s->ne[2], s->ne[3]);
        subsampled = ggml_view_4d(ctx, s,
            s->ne[0], s->ne[1] - drop_extra_preencoded, s->ne[2], s->ne[3],
            /*s->nb[0],*/ s->nb[1], s->nb[2], s->nb[3],
            drop_extra_preencoded * s->nb[1]);  // offset to skip first frames
        // printf("%ld %ld %ld %ld\n", s->ne[0], s->ne[1], s->ne[2], s->ne[3]);
    }

    // Expected chunk_len after subsampling (approximately mel_chunk_frames/8)
    // For 8 mel frames, this should give ~1 encoder frame
    int64_t chunk_len = subsampled->ne[1];

    // Get positional embeddings for cached attention
    // pos_len = 2 * (cache_len + chunk_len) - 1
    int64_t pos_len = 2 * (cache_len + chunk_len) - 1;
    int64_t max_pos_len = nctx->model.pos_emb->ne[1];
    int64_t pos_offset = (max_pos_len - pos_len) / 2;
    
    struct ggml_tensor* pos_emb = ggml_view_2d(ctx, nctx->model.pos_emb,
        d_model, pos_len,
        nctx->model.pos_emb->nb[1],
        pos_offset * nctx->model.pos_emb->nb[1]);
    
    // Create cache input/output tensors for all layers
    k_cache_ins.resize(n_layers);
    v_cache_ins.resize(n_layers);
    conv_cache_ins.resize(n_layers);
    k_cache_outs.resize(n_layers);
    v_cache_outs.resize(n_layers);
    conv_cache_outs.resize(n_layers);
    
    for (int l = 0; l < n_layers; l++) {
        k_cache_ins[l] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, cache_len);
        v_cache_ins[l] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, cache_len);
        conv_cache_ins[l] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, conv_cache_len);
    }

    // Create attention mask for invalid cache positions
    // Shape: [kv_len, 1] - will be broadcast across query positions and heads
    // Values: 0.0 for valid positions, -1e9 for masked (invalid cache) positions
    // Updated before each compute based on cache_valid_len
    int64_t kv_len = cache_len + chunk_len;
    attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kv_len, 1);
    ggml_set_name(attn_mask, "attn_mask");
    ggml_set_input(attn_mask);

    // Process through all conformer layers with caching
    struct ggml_tensor* cur = subsampled;

    for (int l = 0; l < n_layers; l++) {
        cur = build_cached_conformer_layer(
            ctx, cur,
            k_cache_ins[l], v_cache_ins[l], conv_cache_ins[l],
            pos_emb, attn_mask,
            &nctx->model.encoder.layers[l],
            &cfg,
            &k_cache_outs[l], &v_cache_outs[l], &conv_cache_outs[l],
            l  // Pass layer index for debugging
        );
    }
    
    encoder_out = cur;
    // ggml_set_name(encoder_out, "encoder_out");
    // ggml_set_output(encoder_out);
    
    for (int l = 0; l < n_layers; l++) {
        k_cache_ins[l] = ggml_cpy(ctx, k_cache_outs[l], k_cache_ins[l]);
        ggml_set_output(k_cache_ins[l]);
        v_cache_ins[l] = ggml_cpy(ctx, v_cache_outs[l], v_cache_ins[l]);
        ggml_set_output(v_cache_ins[l]);
        conv_cache_ins[l] = ggml_cpy(ctx, conv_cache_outs[l], conv_cache_ins[l]);
        ggml_set_output(conv_cache_ins[l]);
    }
    
    // Build the compute graph
    graph = ggml_new_graph_custom(ctx, 16384, false);
    ggml_build_forward_expand(graph, encoder_out);
    for (int l = 0; l < n_layers; l++) {
        ggml_build_forward_expand(graph, k_cache_ins[l]);
        ggml_build_forward_expand(graph, v_cache_ins[l]);
        ggml_build_forward_expand(graph, conv_cache_ins[l]);
    }
}

struct ggml_tensor* build_cached_conformer_layer(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* k_cache_in,
    struct ggml_tensor* v_cache_in,
    struct ggml_tensor* conv_cache_in,
    struct ggml_tensor* pos_emb,
    struct ggml_tensor* attn_mask,
    nemo_conformer_layer* layer,
    const nemo_cache_config* config,
    struct ggml_tensor** k_cache_out,
    struct ggml_tensor** v_cache_out,
    struct ggml_tensor** conv_cache_out,
    int layer_idx
);

void nemo_encoder_graph::init(struct nemo_context* nctx, const nemo_cache_config& cfg) {
    if (initialized) return;
    // Allocate context for the graph (large enough for 24-layer conformer)
    size_t buf_size = ggml_tensor_overhead() * 8000 + ggml_graph_overhead() * 2;
    
    struct ggml_init_params params = {
        .mem_size = buf_size,
        .mem_buffer = nullptr,  // Let ggml allocate
        .no_alloc = true,
    };
    
    ctx = ggml_init(params);

    if (!ctx) {
        fprintf(stderr, "[ERROR] Failed to create ggml context for encoder graph\n");
        return;
    }

    build_streaming_encoder(ctx, nctx, cfg, 2);  // Build graph structure
    // Allocate memory for the graph
    allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(nctx->model.backend));
    if (!ggml_gallocr_alloc_graph(allocr, graph)) {
        fprintf(stderr, "[ERROR] Failed to allocate encoder graph\n");
        ggml_free(ctx);
        ctx = nullptr;
        return;
    }

    // Initialize all cache tensors to zero using backend-compatible method
    const int n_layers = cfg.n_layers;
    const int d_model = cfg.d_model;
    const int cache_len = cfg.att_left_context;
    const int conv_cache_len = cfg.conv_kernel_size - 1;

    std::vector<float> zeros_attn(d_model * cache_len, 0.0f);
    std::vector<float> zeros_conv(d_model * conv_cache_len, 0.0f);

    for (int l = 0; l < n_layers; l++) {
        ggml_backend_tensor_set(k_cache_ins[l], zeros_attn.data(), 0, zeros_attn.size() * sizeof(float));
        ggml_backend_tensor_set(v_cache_ins[l], zeros_attn.data(), 0, zeros_attn.size() * sizeof(float));
        ggml_backend_tensor_set(conv_cache_ins[l], zeros_conv.data(), 0, zeros_conv.size() * sizeof(float));
    }
    // Sync to ensure initialization completes before compute
    // ggml_backend_synchronize(nctx->model.backend);

    initialized = true;
}

// =============================================================================
// Graph Building: Cached Causal Conv1d
// =============================================================================

struct ggml_tensor* build_cached_causal_conv1d(
    struct ggml_context* ctx,
    struct ggml_tensor* x,              // [d_model, seq_len, batch]
    struct ggml_tensor* cache_in,       // [d_model, kernel_size-1] or nullptr
    struct ggml_tensor* weight,         // [kernel_size, 1, d_model]
    int kernel_size,
    struct ggml_tensor** cache_out      // Output: updated cache
) {
    int64_t d_model = x->ne[0];
    int64_t seq_len = x->ne[1];
    int64_t batch = x->ne[2];
    int64_t cache_len = kernel_size - 1;
    
    struct ggml_tensor* x_padded;
    
    if (cache_in != nullptr) {
        // Prepend cache to input: [d_model, cache_len + seq_len, batch]
        // First, expand cache to have batch dimension
        struct ggml_tensor* cache_expanded = ggml_reshape_3d(ctx, cache_in, d_model, cache_len, 1);
        // Repeat for batch size (simplified: assume batch=1 for now)
        x_padded = ggml_concat(ctx, cache_expanded, x, 1);  // Concat along seq dim
    } else {
        // First chunk: zero-pad left
        x_padded = ggml_pad_ext(ctx, x, 0, 0, cache_len, 0, 0, 0, 0, 0);
    }
    
    // x_padded: [d_model, cache_len + seq_len, batch]
    // Permute to [seq_len + cache_len, d_model, batch] for conv
    struct ggml_tensor* x_perm = ggml_cont(ctx, ggml_permute(ctx, x_padded, 1, 0, 2, 3));
    
    // Reshape weight: [kernel_size, 1, d_model] -> [kernel_size, d_model]
    struct ggml_tensor* w_2d = ggml_reshape_2d(ctx, weight, kernel_size, d_model);
    struct ggml_tensor* w_t = ggml_cont(ctx, ggml_transpose(ctx, w_2d));  // [d_model, kernel_size]
    
    // Manual depthwise conv1d
    struct ggml_tensor* conv_result = nullptr;
    for (int k = 0; k < kernel_size; k++) {
        // Extract slice at offset k: [seq_len, d_model, batch]
        struct ggml_tensor* input_slice = ggml_view_3d(ctx, x_perm,
            seq_len, d_model, batch,
            x_perm->nb[1], x_perm->nb[2],
            k * sizeof(float));
        
        // Get k-th kernel element for each channel: [d_model]
        struct ggml_tensor* kernel_k = ggml_view_1d(ctx, w_t, d_model, k * d_model * sizeof(float));
        kernel_k = ggml_reshape_3d(ctx, kernel_k, 1, d_model, 1);
        
        // Multiply and accumulate
        struct ggml_tensor* product = ggml_mul(ctx, input_slice, kernel_k);
        if (conv_result == nullptr) {
            conv_result = product;
        } else {
            conv_result = ggml_add(ctx, conv_result, product);
        }
    }
    
    // Permute back to [d_model, seq_len, batch]
    struct ggml_tensor* output = ggml_cont(ctx, ggml_permute(ctx, conv_result, 1, 0, 2, 3));
    
    // Output cache: last (kernel_size - 1) frames of the FULL padded input
    // This includes the previous cache concatenated with current input
    // x_padded has shape [d_model, cache_len + seq_len, batch]
    if (cache_out != nullptr) {
        int64_t padded_len = cache_len + seq_len;
        if (padded_len >= cache_len) {
            // Extract last cache_len frames from x_padded
            *cache_out = ggml_view_2d(ctx, x_padded, 
                d_model, cache_len,
                x_padded->nb[1],
                (padded_len - cache_len) * x_padded->nb[1]);
        } else {
            // This should not happen since padded_len = cache_len + seq_len >= cache_len
            *cache_out = ggml_view_2d(ctx, x_padded, d_model, padded_len, x_padded->nb[1], 0);
        }
        *cache_out = ggml_cont(ctx, *cache_out);
    }
    
    return output;
}

// =============================================================================
// Graph Building: Cached Relative Position MHA
// =============================================================================

// Helper: build relative shift for cached attention
static struct ggml_tensor* build_cached_rel_shift(
    struct ggml_context* ctx,
    struct ggml_tensor* input,  // [pos_len, qlen, heads, batch]
    int qlen,
    int cache_len
) {
    // For cached attention, we need to shift to align with the full K sequence
    // The query positions are [0, qlen) in the current chunk
    // The key positions are [0, cache_len + qlen) 
    // Relative position for q[i] to k[j] is: j - (cache_len + i)
    
    // Standard rel_shift implementation adapted for cached case
    int64_t pos_len = input->ne[0];
    int64_t heads = input->ne[2];
    int64_t batch = input->ne[3];
    
    // Pad left with one zero column
    struct ggml_tensor* padded = ggml_pad_ext(ctx, input, 1, 0, 0, 0, 0, 0, 0, 0);
    
    // Reshape to [qlen, pos_len+1, heads, batch]
    struct ggml_tensor* reshaped = ggml_reshape_4d(ctx, ggml_cont(ctx, padded), 
        qlen, pos_len + 1, heads, batch);
    
    // Drop first row: slice from row 1
    struct ggml_tensor* dropped = ggml_view_4d(ctx, reshaped,
        qlen, pos_len, heads, batch,
        reshaped->nb[1], reshaped->nb[2], reshaped->nb[3],
        qlen * ggml_element_size(reshaped));
    
    // Reshape back to [pos_len, qlen, heads, batch]
    struct ggml_tensor* back = ggml_reshape_4d(ctx, ggml_cont(ctx, dropped), 
        pos_len, qlen, heads, batch);
    
    // For cached attention, we need [cache_len + qlen] keys
    int klen = cache_len + qlen;
    
    // Slice to [klen, qlen, heads, batch]
    struct ggml_tensor* out = ggml_view_4d(ctx, back,
        klen, qlen, heads, batch,
        back->nb[1], back->nb[2], back->nb[3], 0);
    
    return ggml_cont(ctx, out);
}

struct ggml_tensor* build_cached_rel_pos_mha(
    struct ggml_context* ctx,
    struct ggml_tensor* x,              // [d_model, chunk_len, batch]
    struct ggml_tensor* k_cache_in,     // [d_model, cache_len] or nullptr
    struct ggml_tensor* v_cache_in,     // [d_model, cache_len] or nullptr
    struct ggml_tensor* pos_emb,        // [d_model, pos_len]
    struct ggml_tensor* attn_mask,      // [kv_len, 1] attention mask (0=valid, -1e9=masked)
    nemo_conformer_layer* layer,
    int n_heads,
    int d_head,
    int left_context,
    [[maybe_unused]] int right_context, // TODO: implement attention mask for right context
    struct ggml_tensor** k_cache_out,
    struct ggml_tensor** v_cache_out
) {
    int64_t d_model = x->ne[0];
    int64_t chunk_len = x->ne[1];
    int64_t batch = x->ne[2];
    int64_t cache_len = k_cache_in ? k_cache_in->ne[1] : 0;
    int64_t kv_len = cache_len + chunk_len;  // Full K/V sequence length
    
    // Q, K, V projections on current chunk
    struct ggml_tensor* q = ggml_mul_mat(ctx, layer->attn_q_w, x);  // [d_model, chunk_len, batch]
    struct ggml_tensor* k_new = ggml_mul_mat(ctx, layer->attn_k_w, x);
    struct ggml_tensor* v_new = ggml_mul_mat(ctx, layer->attn_v_w, x);
    
    // Concatenate with cache if available
    struct ggml_tensor* k;
    struct ggml_tensor* v;
    
    if (k_cache_in != nullptr && cache_len > 0) {
        // Expand cache to 3D: [d_model, cache_len, 1] and concat
        struct ggml_tensor* k_cache_3d = ggml_reshape_3d(ctx, k_cache_in, d_model, cache_len, 1);
        struct ggml_tensor* v_cache_3d = ggml_reshape_3d(ctx, v_cache_in, d_model, cache_len, 1);
        k = ggml_concat(ctx, k_cache_3d, k_new, 1);  // [d_model, kv_len, batch]
        v = ggml_concat(ctx, v_cache_3d, v_new, 1);
    } else {
        k = k_new;
        v = v_new;
    }
    
    // Output new cache: last left_context frames of K/V
    int64_t new_cache_len = std::min(kv_len, (int64_t)left_context);
    int64_t offset = kv_len - new_cache_len;
    *k_cache_out = ggml_cont(ctx, ggml_view_2d(ctx, k, 
        d_model, new_cache_len,
        k->nb[1], offset * k->nb[1]));
    *v_cache_out = ggml_cont(ctx, ggml_view_2d(ctx, v,
        d_model, new_cache_len,
        v->nb[1], offset * v->nb[1]));
    
    // Position projection
    int64_t pos_len = pos_emb->ne[1];
    struct ggml_tensor* pos = ggml_mul_mat(ctx, layer->attn_pos_w, pos_emb);
    
    // Reshape Q, K, V to multi-head format
    q = ggml_reshape_4d(ctx, q, d_head, n_heads, chunk_len, batch);
    k = ggml_reshape_4d(ctx, k, d_head, n_heads, kv_len, batch);
    v = ggml_reshape_4d(ctx, v, d_head, n_heads, kv_len, batch);
    pos = ggml_reshape_3d(ctx, pos, d_head, n_heads, pos_len);
    
    // Permute for attention computation
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));    // [d_head, chunk_len, heads, batch]
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));    // [d_head, kv_len, heads, batch]
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));    // [d_head, kv_len, heads, batch]
    pos = ggml_cont(ctx, ggml_permute(ctx, pos, 0, 2, 1, 3)); // [d_head, pos_len, heads, 1]
    
    // Add position biases
    struct ggml_tensor* bias_u_4d = ggml_reshape_4d(ctx, layer->pos_bias_u, d_head, 1, n_heads, 1);
    struct ggml_tensor* bias_v_4d = ggml_reshape_4d(ctx, layer->pos_bias_v, d_head, 1, n_heads, 1);
    
    struct ggml_tensor* q_u = ggml_add(ctx, q, bias_u_4d);
    struct ggml_tensor* q_v = ggml_add(ctx, q, bias_v_4d);
    
    // Content attention: Q @ K^T -> [kv_len, chunk_len, heads, batch]
    struct ggml_tensor* content_attn = ggml_mul_mat(ctx, k, q_u);
    
    // Position attention: Q @ pos^T -> needs rel_shift
    struct ggml_tensor* pos_attn_raw = ggml_mul_mat(ctx, pos, q_v);
    struct ggml_tensor* pos_attn = build_cached_rel_shift(ctx, pos_attn_raw, chunk_len, cache_len);
    
    // Combine and scale
    float scale = 1.0f / std::sqrt((float)d_head);
    struct ggml_tensor* attn_scores = ggml_add(ctx, content_attn, pos_attn);
    attn_scores = ggml_scale(ctx, attn_scores, scale);

    // Apply attention mask for invalid cache positions
    // attn_mask is [kv_len, 1] with 0 for valid, -1e9 for masked
    // Need to broadcast to [kv_len, chunk_len, heads, batch]
    if (attn_mask != nullptr) {
        // Reshape mask to [kv_len, 1, 1, 1] for broadcasting
        struct ggml_tensor* mask_4d = ggml_reshape_4d(ctx, attn_mask, kv_len, 1, 1, 1);
        attn_scores = ggml_add(ctx, attn_scores, mask_4d);
    }

    // Softmax
    struct ggml_tensor* attn_weights = ggml_soft_max(ctx, attn_scores);
    
    // Apply to values: V @ attn_weights
    struct ggml_tensor* v_perm = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
    struct ggml_tensor* context = ggml_mul_mat(ctx, v_perm, attn_weights);
    
    // Reshape back
    context = ggml_cont(ctx, ggml_permute(ctx, context, 0, 2, 1, 3));
    context = ggml_reshape_3d(ctx, context, d_model, chunk_len, batch);
    
    // Output projection
    struct ggml_tensor* out = ggml_mul_mat(ctx, layer->attn_out_w, context);
    
    return out;
}

// =============================================================================
// Graph Building: Cached Conformer Layer
// =============================================================================

// Forward declaration of helper functions from nemo-ggml.cpp
static struct ggml_tensor* build_layer_norm(
    struct ggml_context* ctx,
    struct ggml_tensor* input,
    struct ggml_tensor* weight,
    struct ggml_tensor* bias,
    float eps = 1e-5f
) {
    struct ggml_tensor* cur = ggml_norm(ctx, input, eps);
    cur = ggml_mul(ctx, cur, weight);
    cur = ggml_add(ctx, cur, bias);
    return cur;
}

static struct ggml_tensor* build_ffn(
    struct ggml_context* ctx,
    struct ggml_tensor* input,
    struct ggml_tensor* linear1_w,
    struct ggml_tensor* linear2_w
) {
    struct ggml_tensor* cur = ggml_mul_mat(ctx, linear1_w, input);
    cur = ggml_silu(ctx, cur);
    cur = ggml_mul_mat(ctx, linear2_w, cur);
    return cur;
}

struct ggml_tensor* build_cached_conformer_layer(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* k_cache_in,
    struct ggml_tensor* v_cache_in,
    struct ggml_tensor* conv_cache_in,
    struct ggml_tensor* pos_emb,
    struct ggml_tensor* attn_mask,
    nemo_conformer_layer* layer,
    const nemo_cache_config* config,
    struct ggml_tensor** k_cache_out,
    struct ggml_tensor** v_cache_out,
    struct ggml_tensor** conv_cache_out,
    int layer_idx
) {
    (void)layer_idx;
    int n_heads = config->n_heads;
    int d_head = config->d_head;
    int kernel_size = config->conv_kernel_size;
    int left_context = config->att_left_context;
    int right_context = config->att_right_context;
    
    struct ggml_tensor* residual = x;
    struct ggml_tensor* cur;
    
    // 1. FFN1: LN -> FFN -> *0.5 + residual
    cur = build_layer_norm(ctx, residual, layer->norm_ff1_w, layer->norm_ff1_b);
    cur = build_ffn(ctx, cur, layer->ffn1_linear1_w, layer->ffn1_linear2_w);
    cur = ggml_scale(ctx, cur, 0.5f);
    residual = ggml_add(ctx, residual, cur);
    
    // 2. Self-attention with caching
    cur = build_layer_norm(ctx, residual, layer->norm_attn_w, layer->norm_attn_b);

    cur = build_cached_rel_pos_mha(ctx, cur, k_cache_in, v_cache_in, pos_emb, attn_mask,
                                    layer, n_heads, d_head, left_context, right_context,
                                    k_cache_out, v_cache_out);

    residual = ggml_add(ctx, residual, cur);
    
    // 3. Conv module with caching
    cur = build_layer_norm(ctx, residual, layer->norm_conv_w, layer->norm_conv_b);
    
    // Pointwise conv1 + GLU
    int64_t d_model = cur->ne[0];
    int64_t seq_len = cur->ne[1];
    int64_t batch = cur->ne[2];
    
    struct ggml_tensor* pw1_w_2d = ggml_reshape_2d(ctx, layer->conv_pw1_w, d_model, 2 * d_model);
    struct ggml_tensor* conv_cur = ggml_mul_mat(ctx, pw1_w_2d, cur);
    
    // GLU
    int64_t half_ch = d_model;
    int64_t full_ch = 2 * d_model;
    size_t nb1 = full_ch * sizeof(float);
    size_t nb2 = full_ch * seq_len * sizeof(float);
    struct ggml_tensor* glu_a = ggml_cont(ctx, ggml_view_3d(ctx, conv_cur, half_ch, seq_len, batch, nb1, nb2, 0));
    struct ggml_tensor* glu_b = ggml_cont(ctx, ggml_view_3d(ctx, conv_cur, half_ch, seq_len, batch, nb1, nb2, half_ch * sizeof(float)));
    conv_cur = ggml_mul(ctx, glu_a, ggml_sigmoid(ctx, glu_b));
    conv_cur = ggml_cont(ctx, conv_cur);

    // Cached depthwise conv1d
    conv_cur = build_cached_causal_conv1d(ctx, conv_cur, conv_cache_in,
                                           layer->conv_dw_w, kernel_size, conv_cache_out);

    // Layer norm + Swish + Pointwise conv2
    conv_cur = ggml_norm(ctx, conv_cur, 1e-5f);
    conv_cur = ggml_mul(ctx, conv_cur, layer->conv_ln_w);
    conv_cur = ggml_add(ctx, conv_cur, layer->conv_ln_b);
    conv_cur = ggml_silu(ctx, conv_cur);
    
    struct ggml_tensor* pw2_w_2d = ggml_reshape_2d(ctx, layer->conv_pw2_w, d_model, d_model);
    conv_cur = ggml_mul_mat(ctx, pw2_w_2d, conv_cur);
    
    residual = ggml_add(ctx, residual, conv_cur);
    
    // 4. FFN2: LN -> FFN -> *0.5 + residual
    cur = build_layer_norm(ctx, residual, layer->norm_ff2_w, layer->norm_ff2_b);
    cur = build_ffn(ctx, cur, layer->ffn2_linear1_w, layer->ffn2_linear2_w);
    cur = ggml_scale(ctx, cur, 0.5f);
    residual = ggml_add(ctx, residual, cur);
    
    cur = build_layer_norm(ctx, residual, layer->norm_final_w, layer->norm_final_b);

    return cur;
}

// =============================================================================
// Public API Implementation
// =============================================================================

struct nemo_stream_context* nemo_stream_init(
    struct nemo_context* ctx,
    const nemo_cache_config* config
) {
    if (!ctx) return nullptr;
    
    nemo_stream_context* sctx = new nemo_stream_context();
    
    // Use provided config or create default from model
    nemo_cache_config cfg;
    if (config) {
        cfg = *config;
    } else {
        // Default config based on model
        cfg.d_model = ctx->model.hparams.d_model;
        cfg.n_layers = ctx->model.hparams.n_layers;
        cfg.n_heads = ctx->model.hparams.n_heads;
        cfg.d_head = ctx->model.hparams.d_head;
        cfg.conv_kernel_size = ctx->model.hparams.kernel_size;
        cfg.conv_cache_size = cfg.conv_kernel_size - 1;
        cfg.vocab_size = ctx->model.hparams.vocab_size;
        cfg.blank_token = cfg.vocab_size - 1;
        cfg.decoder_hidden = nemo_decoder::HIDDEN_SIZE;
        cfg.decoder_layers = nemo_decoder::NUM_LAYERS;
    }
    
    sctx->init(ctx, cfg);
    
    return sctx;
}

// Forward declaration for token to text conversion (from nemo-ggml.cpp)
std::string tokens_to_text(const std::vector<timed_token>& tokens, const std::vector<char8>& vocab, bool timestamp_words);

// Helper to convert plain token IDs to text (no timestamps)
static std::string tokens_to_text_simple(const std::vector<int>& token_ids, const std::vector<char8>& vocab) {
    std::vector<timed_token> tokens;
    tokens.reserve(token_ids.size());
    for (int id : token_ids) {
        tokens.push_back({id, 0});
    }
    return tokens_to_text(tokens, vocab, false);
}

// Forward declaration for dump function
void append_dump_array(const float* data, int64_t *ne, size_t n_elements, const char* filename);

// Helper: Initialize the decode graph once (called on first decode)
static void init_decode_graph(nemo_stream_context* sctx) {
    nemo_context* nctx = sctx->nctx;
    const int d_model = sctx->config.d_model;
    const int hidden_size = sctx->config.decoder_hidden;
    const int num_layers = sctx->config.decoder_layers;

    // Create compute context for the graph
    size_t buf_size = ggml_tensor_overhead() * 100 + ggml_graph_overhead();
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    sctx->decode_ctx = ggml_init(params);
    if (!sctx->decode_ctx) {
        fprintf(stderr, "[ERROR] Failed to create decode context\n");
        return;
    }

    struct ggml_context* ctx0 = sctx->decode_ctx;

    // Create input tensors
    sctx->decode_h_in = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, num_layers * hidden_size);
    sctx->decode_c_in = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, num_layers * hidden_size);
    sctx->decode_token_emb = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hidden_size);
    sctx->decode_enc_in = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, d_model);
    ggml_set_input(sctx->decode_h_in);
    ggml_set_input(sctx->decode_c_in);
    ggml_set_input(sctx->decode_token_emb);
    ggml_set_input(sctx->decode_enc_in);

    // Build decoder step
    struct ggml_tensor* h_out = nullptr;
    struct ggml_tensor* c_out = nullptr;
    struct ggml_tensor* dec_out = build_decoder_step(ctx0, sctx->decode_token_emb,
                                                      sctx->decode_h_in, sctx->decode_c_in,
                                                      &nctx->model.decoder, &h_out, &c_out);

    // Build joint
    struct ggml_tensor* logits = build_joint(ctx0, sctx->decode_enc_in, dec_out, &nctx->model.joint);
    ggml_set_output(logits);
    ggml_set_output(h_out);
    ggml_set_output(c_out);

    // Store output tensor pointers
    sctx->decode_logits = logits;
    sctx->decode_h_out = h_out;
    sctx->decode_c_out = c_out;

    // Build graph
    sctx->decode_graph = ggml_new_graph(ctx0);
    ggml_build_forward_expand(sctx->decode_graph, logits);
    ggml_build_forward_expand(sctx->decode_graph, h_out);
    ggml_build_forward_expand(sctx->decode_graph, c_out);

    // Allocate
    sctx->decode_allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(nctx->model.backend));
    if (!ggml_gallocr_alloc_graph(sctx->decode_allocr, sctx->decode_graph)) {
        fprintf(stderr, "[ERROR] Failed to allocate decode graph\n");
        ggml_gallocr_free(sctx->decode_allocr);
        ggml_free(sctx->decode_ctx);
        sctx->decode_allocr = nullptr;
        sctx->decode_ctx = nullptr;
        return;
    }

    sctx->decode_graph_initialized = true;
}

// Helper: Run one step of greedy decode for a single encoder frame
// Returns all tokens emitted for this frame (can be 0 to MAX_SYMBOLS_PER_STEP)
static std::vector<int> decode_one_step(
    nemo_stream_context* sctx,
    const float* enc_frame  // [d_model]
) {
    nemo_context* nctx = sctx->nctx;
    const int d_model = sctx->config.d_model;
    const int hidden_size = sctx->config.decoder_hidden;
    const int vocab_size = sctx->config.vocab_size;
    const int blank_token = sctx->config.blank_token;
    const int MAX_SYMBOLS_PER_STEP = 10;

    // Initialize decode graph on first use
    if (!sctx->decode_graph_initialized) {
        init_decode_graph(sctx);
        if (!sctx->decode_graph_initialized) {
            return {};  // Initialization failed
        }
    }

    std::vector<int> emitted_tokens;  // All tokens emitted for this encoder frame
    std::vector<float> emb_data(hidden_size);
    std::vector<float> logits_data(vocab_size);
    std::vector<float> new_h_state(sctx->decoder_state.h.size());
    std::vector<float> new_c_state(sctx->decoder_state.c.size());

    for (int sym = 0; sym < MAX_SYMBOLS_PER_STEP; sym++) {
        sctx->total_decode_iterations++;

        // Set inputs (measure transfer time)
        auto t_transfer_start = std::chrono::high_resolution_clock::now();

        ggml_backend_tensor_set(sctx->decode_h_in, sctx->decoder_state.h.data(), 0,
                                 sctx->decoder_state.h.size() * sizeof(float));
        ggml_backend_tensor_set(sctx->decode_c_in, sctx->decoder_state.c.data(), 0,
                                 sctx->decoder_state.c.size() * sizeof(float));

        // Get embedding for prev_token
        size_t emb_offset = sctx->decoder_state.prev_token * hidden_size * sizeof(float);
        ggml_backend_tensor_get(nctx->model.decoder.embedding, emb_data.data(), emb_offset,
                                 hidden_size * sizeof(float));
        ggml_backend_tensor_set(sctx->decode_token_emb, emb_data.data(), 0, hidden_size * sizeof(float));

        ggml_backend_tensor_set(sctx->decode_enc_in, enc_frame, 0, d_model * sizeof(float));

        auto t_transfer_end = std::chrono::high_resolution_clock::now();
        sctx->transfer_seconds += std::chrono::duration<double>(t_transfer_end - t_transfer_start).count();

        // Compute (measure decoder time)
        auto t_decode_start = std::chrono::high_resolution_clock::now();
        ggml_backend_graph_compute(nctx->model.backend, sctx->decode_graph);
        auto t_decode_end = std::chrono::high_resolution_clock::now();
        sctx->decoder_seconds += std::chrono::duration<double>(t_decode_end - t_decode_start).count();

        // Get logits and find argmax (measure transfer time)
        auto t_transfer2_start = std::chrono::high_resolution_clock::now();
        ggml_backend_tensor_get(sctx->decode_logits, logits_data.data(), 0, vocab_size * sizeof(float));
        auto t_transfer2_end = std::chrono::high_resolution_clock::now();
        sctx->transfer_seconds += std::chrono::duration<double>(t_transfer2_end - t_transfer2_start).count();

        int best_token = 0;
        float best_score = logits_data[0];
        for (int v = 1; v < vocab_size; v++) {
            if (logits_data[v] > best_score) {
                best_score = logits_data[v];
                best_token = v;
            }
        }

        if (best_token == blank_token) {
            // Move to next time step - DON'T update state
            break;
        }

        // Get updated state (measure transfer time)
        auto t_transfer3_start = std::chrono::high_resolution_clock::now();
        ggml_backend_tensor_get(sctx->decode_h_out, new_h_state.data(), 0, new_h_state.size() * sizeof(float));
        ggml_backend_tensor_get(sctx->decode_c_out, new_c_state.data(), 0, new_c_state.size() * sizeof(float));
        auto t_transfer3_end = std::chrono::high_resolution_clock::now();
        sctx->transfer_seconds += std::chrono::duration<double>(t_transfer3_end - t_transfer3_start).count();

        // Emit non-blank token
        emitted_tokens.push_back(best_token);
        sctx->decoder_state.prev_token = best_token;

        // Only update LSTM state when emitting a non-blank token
        sctx->decoder_state.h = new_h_state;
        sctx->decoder_state.c = new_c_state;
    }

    return emitted_tokens;
}

struct dump_file_info {
    bool cleared;
    int64_t shape[4];
};
static std::map<std::string, dump_file_info> file_info;

void append_dump_array(
    const float* data,
    int64_t *ne,
    size_t n_elements,
    const char* filename 
) {
    if (file_info.find(filename) == file_info.end()) {
        // First time: clear file
        FILE *f = fopen(filename, "wb");
        if (!f) {
            fprintf(stderr, "[ERROR] Failed to open dump file: %s\n", filename);
            return;
        }
        size_t bwrite = fwrite(&ne[0], 1, 32, f);
        assert(sizeof(int64_t) * 4 == 32);
        assert(bwrite == 32);
        struct dump_file_info info {
            .cleared = true,
            .shape = {ne[0], ne[1], ne[2], ne[3]}
        };
        file_info[filename] = info;
        fclose(f);
    }
    struct dump_file_info info = file_info[filename];
    for (int i = 0; i < 4; i++) {
        if (info.shape[i] != ne[i]) {
            fprintf(stderr, "[ERROR] Shape mismatch for dump file: %s\n", filename);
            fprintf(stderr, "Expected shape: [%ld, %ld, %ld, %ld]\n", info.shape[0], info.shape[1], info.shape[2], info.shape[3]);
            fprintf(stderr, "Actual shape:   [%ld, %ld, %ld, %ld]\n", ne[0], ne[1], ne[2], ne[3]);
            return;
        }
    }
    FILE *f = fopen(filename, "ab");
    if (!f) {
        fprintf(stderr, "[ERROR] Failed to open dump file: %s\n", filename);
        return;
    }
    size_t bwrite = fwrite(data, sizeof(float), n_elements, f);
    if (bwrite != n_elements) {
        fprintf(stderr, "[ERROR] Failed to write all elements to dump file: %s\n", filename);
    }
    fclose(f);
}

void append_dump_tensor(
    struct ggml_context* ctx,
    const char* name,
    const char* filename
) {
    struct ggml_tensor* tensor = ggml_get_tensor(ctx, name);
    if (!tensor) {
        fprintf(stderr, "[ERROR] Tensor not found for dump: %s\n", name);
        return;
    }

    size_t n_elements = ggml_nelements(tensor);
    std::vector<float> data(n_elements);
    size_t ndim = ggml_n_dims(tensor);
    bool do_print = file_info.find(filename) == file_info.end();
    if (do_print) {
        printf("Dumping tensor '%s' to %s, shape: ", name, filename);
        const char * sep = "";
        for (size_t i = 0; i < ndim; i++) {
            printf("%s%ld", sep, tensor->ne[i]);
            sep = ", ";
            size_t dim_size = tensor->ne[i];
            (void)dim_size;
        }
        printf("\n");
    }
    ggml_backend_tensor_get(tensor, data.data(), 0, n_elements * sizeof(float));
    append_dump_array(data.data(), &tensor->ne[0], n_elements, filename);
}

// Helper: Process encoder output through cached conformer and decode
static std::string process_mel_chunk_streaming(
    nemo_stream_context* sctx,
    const float* mel_data,  // [n_mels, n_frames] column-major
    size_t n_mel_frames
) {
    nemo_context* nctx = sctx->nctx;
    const size_t d_model = sctx->config.d_model;
    const size_t n_mels = sctx->config.n_mels;

    ggml_backend_tensor_set(sctx->encoder_graph.mel_input, mel_data, 0,
                             n_mel_frames * n_mels * sizeof(float));

    auto mel_input = sctx->encoder_graph.mel_input;

    // torch.Size([1, 17, 128])
    assert((size_t)mel_input->ne[0] == n_mels);
    assert((size_t)mel_input->ne[1] == n_mel_frames);
    assert((size_t)mel_input->ne[2] == 1);

    // Prepare attention mask for invalid cache positions
    // offset = cache_len - cache_valid_len: positions [0, offset) are masked
    const int cache_len = sctx->config.att_left_context;
    const int64_t kv_len = sctx->encoder_graph.attn_mask->ne[0];
    const int offset = cache_len - sctx->cache_valid_len;

    std::vector<float> mask_data(kv_len);
    for (int64_t i = 0; i < kv_len; i++) {
        // Mask positions [0, offset) - these are invalid cache positions
        mask_data[i] = (i < offset) ? -1e9f : 0.0f;
    }
    ggml_backend_tensor_set(sctx->encoder_graph.attn_mask, mask_data.data(), 0,
                            kv_len * sizeof(float));

    // Run encoder graph (measure encoder time)
    auto t_encoder_start = std::chrono::high_resolution_clock::now();
    ggml_backend_graph_compute(nctx->model.backend, sctx->encoder_graph.graph);
    // ggml_backend_synchronize(nctx->model.backend);
    auto t_encoder_end = std::chrono::high_resolution_clock::now();
    sctx->encoder_seconds += std::chrono::duration<double>(t_encoder_end - t_encoder_start).count();

    // Get encoder output
    size_t enc_out_frames = sctx->encoder_graph.encoder_out->ne[1];

    // long *ne = sctx->encoder_graph.encoder_out->ne;
    // printf("Encoder output shape: %ld %ld %ld\n", ne[0], ne[1], ne[2]);
    std::vector<float> enc_out(ggml_nelements(sctx->encoder_graph.encoder_out));
    auto subsampled = sctx->encoder_graph.encoder_out;
    ggml_backend_tensor_get(subsampled, enc_out.data(), 0, enc_out.size() * sizeof(float));
    // append_dump_tensor(sctx->encoder_graph.ctx, "encoder_out", "my_bin/ggml_subsampling_output.bin");
    // torch.Size([1, 1, 1024])
    assert((size_t)subsampled->ne[0] == d_model);
    // assert((size_t)subsampled->ne[1] == 1); TODO: depends on chunk size
    assert((size_t)subsampled->ne[2] == 1);
    // ggml_backend_synchronize(nctx->model.backend);

    // Update cache validity tracking
    // cache_valid_len grows by enc_out_frames (chunk_len), capped at cache_len
    sctx->cache_valid_len = std::min(sctx->cache_valid_len + (int)enc_out_frames, cache_len);

    // =========================================================================
    // STREAMING POST-PROCESS (matches NeMo streaming_post_process)
    // =========================================================================
    // 1. Truncate encoder output to valid_out_len frames
    //    NeMo: encoded = encoded[:, :, :valid_out_len]
    int32_t valid_out_len = sctx->config.valid_out_len;
    if (enc_out_frames > (size_t)valid_out_len) {
        // Only use first valid_out_len frames
        enc_out_frames = valid_out_len;
        enc_out.resize(enc_out_frames * d_model);
    }

    // 2. Cache truncation is already handled in build_cached_rel_pos_mha
    //    which takes the last left_context (70) frames

    // Run greedy decode on each encoder frame
    std::vector<int> new_tokens;
    for (size_t t = 0; t < enc_out_frames; t++) {
        const float* enc_frame = enc_out.data() + t * d_model;
        std::vector<int> frame_tokens = decode_one_step(sctx, enc_frame);
        // Collect all tokens emitted for this encoder frame
        for (int token : frame_tokens) {
            if (token >= 0 && token != sctx->config.blank_token) {
                new_tokens.push_back(token);
                sctx->tokens.push_back(token);
            }
        }
    }

    // Convert new tokens to text
    if (new_tokens.empty()) {
        return "";
    }

    std::string new_text = tokens_to_text_simple(new_tokens, nctx->model.vocab);
    sctx->transcript += new_text;
    return new_text;
}

// Forward declaration
std::string nemo_transcribe_audio_with_state(
    struct nemo_context* ctx,
    std::vector<float> & mel,
    nemo_decoder_state* decoder_state
);

// Forward declaration
std::string nemo_transcribe_audio(
    struct nemo_context* ctx,
    std::vector<int16_t> & audio_data
);

// True incremental streaming using cached encoder
// This processes audio incrementally without re-transcribing
std::string nemo_stream_process_incremental(
    struct nemo_stream_context* sctx,
    const int16_t* audio,
    int n_samples
) {
    if (!sctx || !audio || n_samples <= 0) return "";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    sctx->total_audio_seconds += (double)n_samples / sctx->config.sample_rate;
    
    // Convert audio to mel spectrogram
    struct nemo_preprocessor* pp = sctx->nctx->preprocessor;
    std::vector<float> mel;
    size_t n_mel_frames = nemo_preprocessor_process(pp, audio, n_samples, mel);
    (void)n_mel_frames;
    sctx->mel_buffer.insert(sctx->mel_buffer.end(), mel.begin(), mel.end());
    
    // Process when we have enough mel frames for the configured chunk size
    // Add to mel buffer if not enough
    size_t total_mels = sctx->mel_buffer.size() / sctx->config.n_mels;
    size_t graph_mel_frames = sctx->config.get_chunk_mel_frames();
    if (total_mels < graph_mel_frames) {
        return "";
    }

    // Process all available chunks
    std::string all_text;
    while (total_mels >= graph_mel_frames) {
        // The pre-built graph expects the chunk size based on latency mode:
        //   [70, 0]  -> 17 mel frames (9 overlap + 8 new)
        //   [70, 13] -> 121 mel frames (9 overlap + 112 new)
        std::string chunk_text = process_mel_chunk_streaming(
            sctx,
            sctx->mel_buffer.data(),
            graph_mel_frames
        );
        all_text += chunk_text;

        // Track chunk count
        sctx->total_chunks_processed++;

        // Advance buffer by shift_mel_frames (remove consumed frames, keep overlap)
        size_t shift_frames = sctx->shift_mel_frames();
        assert(total_mels >= sctx->overlap_mel_frames()
            && graph_mel_frames > sctx->overlap_mel_frames());
        sctx->mel_buffer.erase(
            sctx->mel_buffer.begin(),
            sctx->mel_buffer.begin() + shift_frames * sctx->config.n_mels
        );

        // Update total_mels for next iteration
        total_mels = sctx->mel_buffer.size() / sctx->config.n_mels;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    sctx->total_compute_seconds += elapsed.count();

    return all_text;
}

// Finalize streaming and return final transcript
std::string nemo_stream_finalize(struct nemo_stream_context* sctx) {
    if (!sctx) return "";

    // Print streaming statistics
    printf("\n[STREAMING STATS] Total chunks processed: %d\n", sctx->total_chunks_processed);
    printf("[STREAMING STATS] Total audio: %.2f seconds\n", sctx->total_audio_seconds);
    printf("[STREAMING STATS] Total compute: %.4f seconds\n", sctx->total_compute_seconds);
    printf("[STREAMING STATS] RTF (compute/audio): %.4f\n", sctx->rtf());

    // Detailed timing breakdown
    printf("\n[PROFILING] Timing breakdown:\n");
    printf("[PROFILING]   Encoder compute:    %.4f sec (%.1f%%)\n",
           sctx->encoder_seconds, 100.0 * sctx->encoder_seconds / sctx->total_compute_seconds);
    printf("[PROFILING]   Decoder compute:    %.4f sec (%.1f%%)\n",
           sctx->decoder_seconds, 100.0 * sctx->decoder_seconds / sctx->total_compute_seconds);
    printf("[PROFILING]   CPU-GPU transfers:  %.4f sec (%.1f%%)\n",
           sctx->transfer_seconds, 100.0 * sctx->transfer_seconds / sctx->total_compute_seconds);
    double other_seconds = sctx->total_compute_seconds - sctx->encoder_seconds - sctx->decoder_seconds - sctx->transfer_seconds;
    printf("[PROFILING]   Other (CPU work):   %.4f sec (%.1f%%)\n",
           other_seconds, 100.0 * other_seconds / sctx->total_compute_seconds);
    printf("[PROFILING]   Decode iterations:  %d (avg %.1f per chunk)\n",
           sctx->total_decode_iterations,
           (double)sctx->total_decode_iterations / sctx->total_chunks_processed);

    printf("\n[STREAMING STATS] Config: chunk_mel_frames=%zu, shift_mel_frames=%zu, valid_out_len=%d\n",
           sctx->config.get_chunk_mel_frames(),
           sctx->config.get_shift_mel_frames(),
           sctx->config.valid_out_len);
    printf("[STREAMING STATS] Config: att_left_context=%d, att_right_context=%d, drop_extra_pre_encoded=%d\n",
           sctx->config.att_left_context,
           sctx->config.att_right_context,
           sctx->config.drop_extra_pre_encoded);

    return sctx->transcript;
}

// Get accumulated transcript
std::string nemo_stream_get_transcript(struct nemo_stream_context* sctx) {
    if (!sctx) return "";
    return sctx->transcript;
}

const std::vector<int>& nemo_stream_get_tokens(struct nemo_stream_context* sctx) {
    static std::vector<int> empty;
    if (!sctx) return empty;
    return sctx->tokens;
}

void nemo_stream_reset(struct nemo_stream_context* sctx) {
    if (sctx) {
        sctx->reset();
    }
}

void nemo_stream_free(struct nemo_stream_context* sctx) {
    if (sctx) {
        delete sctx;
    }
}
