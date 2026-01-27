#include "nemo-stream.h"
#include "preprocessor.h"

#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>

// Forward declaration from nemo-ggml.cpp
std::string tokens_to_text(const std::vector<int> & tokens, const std::vector<char8> & vocab);

// =============================================================================
// Cache Structure Implementations
// =============================================================================

void nemo_layer_attn_cache::init(int32_t max_len, int32_t dim) {
    max_cache_len = max_len;
    d_model = dim;
    cache_len = 0;
    k_cache.resize(max_cache_len * d_model, 0.0f);
    v_cache.resize(max_cache_len * d_model, 0.0f);
}

void nemo_layer_attn_cache::reset() {
    cache_len = 0;
    std::fill(k_cache.begin(), k_cache.end(), 0.0f);
    std::fill(v_cache.begin(), v_cache.end(), 0.0f);
}

void nemo_layer_attn_cache::update(const float* k_new, const float* v_new, int32_t new_len) {
    // New cache = [old_cache[trim:], new_data]
    // trim = max(0, cache_len + new_len - max_cache_len)
    int32_t total_len = cache_len + new_len;
    int32_t trim = std::max(0, total_len - max_cache_len);
    int32_t keep_len = cache_len - trim;
    
    if (keep_len > 0 && trim > 0) {
        // Shift old data left
        memmove(k_cache.data(), k_cache.data() + trim * d_model, keep_len * d_model * sizeof(float));
        memmove(v_cache.data(), v_cache.data() + trim * d_model, keep_len * d_model * sizeof(float));
    }
    
    // Append new data
    int32_t new_cache_len = std::min(total_len, max_cache_len);
    int32_t copy_offset = new_cache_len - new_len;
    memcpy(k_cache.data() + copy_offset * d_model, k_new, new_len * d_model * sizeof(float));
    memcpy(v_cache.data() + copy_offset * d_model, v_new, new_len * d_model * sizeof(float));
    
    cache_len = new_cache_len;
}

void nemo_layer_conv_cache::init(int32_t kernel_size, int32_t dim) {
    cache_len = kernel_size - 1;
    d_model = dim;
    cache.resize(d_model * cache_len, 0.0f);
}

void nemo_layer_conv_cache::reset() {
    std::fill(cache.begin(), cache.end(), 0.0f);
}

void nemo_layer_conv_cache::update(const float* new_data, int32_t seq_len) {
    // Keep the last (kernel_size - 1) frames
    // new_data is [d_model, seq_len] in channels-first layout
    // cache is [d_model, cache_len] in channels-first layout
    
    if (seq_len >= cache_len) {
        // Take last cache_len frames from new_data
        int32_t offset = seq_len - cache_len;
        for (int32_t c = 0; c < d_model; c++) {
            memcpy(cache.data() + c * cache_len, 
                   new_data + c * seq_len + offset,
                   cache_len * sizeof(float));
        }
    } else {
        // Shift old cache left, append new data
        int32_t keep = cache_len - seq_len;
        for (int32_t c = 0; c < d_model; c++) {
            memmove(cache.data() + c * cache_len,
                    cache.data() + c * cache_len + seq_len,
                    keep * sizeof(float));
            memcpy(cache.data() + c * cache_len + keep,
                   new_data + c * seq_len,
                   seq_len * sizeof(float));
        }
    }
}

void nemo_encoder_cache::init(const nemo_cache_config& cfg) {
    config = cfg;
    
    // Initialize per-layer caches
    attn_caches.resize(cfg.n_layers);
    conv_caches.resize(cfg.n_layers);
    
    for (int i = 0; i < cfg.n_layers; i++) {
        attn_caches[i].init(cfg.att_left_context, cfg.d_model);
        conv_caches[i].init(cfg.conv_kernel_size, cfg.d_model);
    }
    
    // Initialize mel buffer
    mel_buffer.clear();
    mel_buffer_len = 0;
    
    // Initialize audio buffer
    audio_buffer.clear();
}

void nemo_encoder_cache::reset() {
    for (auto& cache : attn_caches) cache.reset();
    for (auto& cache : conv_caches) cache.reset();
    mel_buffer.clear();
    mel_buffer_len = 0;
    audio_buffer.clear();
}

size_t nemo_encoder_cache::memory_usage_bytes() const {
    size_t total = 0;
    
    // Attention caches: 2 * n_layers * max_cache_len * d_model * sizeof(float)
    total += 2 * config.n_layers * config.att_left_context * config.d_model * sizeof(float);
    
    // Conv caches: n_layers * d_model * (kernel_size - 1) * sizeof(float)
    total += config.n_layers * config.d_model * (config.conv_kernel_size - 1) * sizeof(float);
    
    // Buffers (approximate max)
    total += config.n_mels * config.subsampling_factor * sizeof(float);  // mel buffer
    total += config.chunk_samples * sizeof(int16_t);  // audio buffer
    
    return total;
}

void nemo_decoder_state::init(int32_t layers, int32_t hidden) {
    n_layers = layers;
    hidden_size = hidden;
    h.resize(layers * hidden, 0.0f);
    c.resize(layers * hidden, 0.0f);
    prev_token = -1;  // Will use blank token from config
}

void nemo_decoder_state::reset() {
    std::fill(h.begin(), h.end(), 0.0f);
    std::fill(c.begin(), c.end(), 0.0f);
    prev_token = -1;
}

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
    
    // Initialize encoder cache
    encoder_cache.init(cfg);
    
    // Initialize decoder state
    decoder_state.init(cfg.decoder_layers, cfg.decoder_hidden);
    decoder_state.prev_token = cfg.blank_token;
    
    // Pre-build encoder graph for streaming (8 mel frames -> 1+ encoder frames)
    encoder_graph.init(ctx, cfg, cfg.subsampling_factor);
    
    // Clear tokens
    tokens.clear();
    
    // Reset timing
    total_audio_seconds = 0;
    total_compute_seconds = 0;
}

void nemo_stream_context::reset() {
    encoder_cache.reset();
    encoder_graph.reset();
    decoder_state.reset();
    decoder_state.prev_token = config.blank_token;
    tokens.clear();
    total_audio_seconds = 0;
    total_compute_seconds = 0;
}

// Forward declarations for graph building
struct ggml_tensor* build_conv_subsampling(
    struct ggml_context* ctx,
    struct ggml_tensor* mel,
    nemo_conv_subsampling* sub
);

struct ggml_tensor* build_cached_conformer_layer(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* k_cache_in,
    struct ggml_tensor* v_cache_in,
    struct ggml_tensor* conv_cache_in,
    struct ggml_tensor* pos_emb,
    nemo_conformer_layer* layer,
    const nemo_cache_config* cfg,
    struct ggml_tensor** k_cache_out,
    struct ggml_tensor** v_cache_out,
    struct ggml_tensor** conv_cache_out
);

void nemo_encoder_graph::init(struct nemo_context* nctx, const nemo_cache_config& cfg, int mel_chunk_frames) {
    if (initialized) return;
    
    const int d_model = cfg.d_model;
    const int n_layers = cfg.n_layers;
    const int n_mels = cfg.n_mels;
    const int cache_len = cfg.att_left_context;
    const int conv_cache_len = cfg.conv_kernel_size - 1;
    
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
    
    // Create input tensor for mel chunk
    mel_input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_mels, mel_chunk_frames, 1);
    ggml_set_name(mel_input, "mel_input");
    ggml_set_input(mel_input);
    
    // Run subsampling
    struct ggml_tensor* subsampled = build_conv_subsampling(ctx, mel_input, &nctx->model.encoder.subsampling);
    
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
        ggml_set_input(k_cache_ins[l]);
        ggml_set_input(v_cache_ins[l]);
        ggml_set_input(conv_cache_ins[l]);
    }
    
    // Process through all conformer layers with caching
    struct ggml_tensor* cur = subsampled;
    
    for (int l = 0; l < n_layers; l++) {
        cur = build_cached_conformer_layer(
            ctx, cur,
            k_cache_ins[l], v_cache_ins[l], conv_cache_ins[l],
            pos_emb,
            &nctx->model.encoder.layers[l],
            &cfg,
            &k_cache_outs[l], &v_cache_outs[l], &conv_cache_outs[l]
        );
    }
    
    encoder_out = cur;
    ggml_set_name(encoder_out, "encoder_out");
    ggml_set_output(encoder_out);
    
    for (int l = 0; l < n_layers; l++) {
        if (k_cache_outs[l]) ggml_set_output(k_cache_outs[l]);
        if (v_cache_outs[l]) ggml_set_output(v_cache_outs[l]);
        if (conv_cache_outs[l]) ggml_set_output(conv_cache_outs[l]);
    }
    
    // Build the compute graph
    graph = ggml_new_graph_custom(ctx, 16384, false);
    ggml_build_forward_expand(graph, encoder_out);
    for (int l = 0; l < n_layers; l++) {
        if (k_cache_outs[l]) ggml_build_forward_expand(graph, k_cache_outs[l]);
        if (v_cache_outs[l]) ggml_build_forward_expand(graph, v_cache_outs[l]);
        if (conv_cache_outs[l]) ggml_build_forward_expand(graph, conv_cache_outs[l]);
    }
    
    // Allocate memory for the graph
    allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(nctx->model.backend));
    if (!ggml_gallocr_alloc_graph(allocr, graph)) {
        fprintf(stderr, "[ERROR] Failed to allocate encoder graph\n");
        ggml_free(ctx);
        ctx = nullptr;
        return;
    }
    
    initialized = true;
    fprintf(stderr, "[INFO] Pre-built encoder graph: %d mel frames -> %lld encoder frames\n",
            mel_chunk_frames, (long long)chunk_len);
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
    if (k_cache_out != nullptr) {
        int64_t new_cache_len = std::min(kv_len, (int64_t)left_context);
        int64_t offset = kv_len - new_cache_len;
        *k_cache_out = ggml_cont(ctx, ggml_view_2d(ctx, k, 
            d_model, new_cache_len,
            k->nb[1], offset * k->nb[1]));
        *v_cache_out = ggml_cont(ctx, ggml_view_2d(ctx, v,
            d_model, new_cache_len,
            v->nb[1], offset * v->nb[1]));
    }
    
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
    
    // Apply attention mask for limited context
    // TODO: For right_context > 0, need to mask future positions
    // For pure causal (right_context = 0), lower triangular is automatic from cache structure
    
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
    nemo_conformer_layer* layer,
    const nemo_cache_config* config,
    struct ggml_tensor** k_cache_out,
    struct ggml_tensor** v_cache_out,
    struct ggml_tensor** conv_cache_out
) {
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
    cur = build_cached_rel_pos_mha(ctx, cur, k_cache_in, v_cache_in, pos_emb,
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
    
    // 5. Final layer norm
    cur = build_layer_norm(ctx, residual, layer->norm_final_w, layer->norm_final_b);
    
    return cur;
}

// =============================================================================
// Streaming Encoder Step (Cached)
// =============================================================================

// Process a single chunk through the cached encoder pipeline using pre-built graph
// mel_chunk: [n_mels, chunk_frames] 
// Returns: encoder output [d_model, chunk_len]
static std::vector<float> process_encoder_chunk_cached(
    struct nemo_stream_context* sctx,
    const float* mel_data,
    int n_mel_frames
) {
    struct nemo_context* nctx = sctx->nctx;
    nemo_encoder_graph& g = sctx->encoder_graph;
    
    if (!g.initialized) {
        fprintf(stderr, "[ERROR] Encoder graph not initialized\n");
        return {};
    }
    
    const int d_model = sctx->config.d_model;
    const int n_layers = sctx->config.n_layers;
    const int n_mels = sctx->config.n_mels;
    const int cache_len = sctx->config.att_left_context;
    const int conv_cache_len = sctx->config.conv_kernel_size - 1;
    
    static int call_count = 0;
    call_count++;
    if (call_count <= 3) {
        fprintf(stderr, "[DEBUG] process_encoder_chunk_cached (reusing graph): mel_frames=%d, call #%d\n", 
                n_mel_frames, call_count);
    }
    
    // Set mel input data
    ggml_backend_tensor_set(g.mel_input, mel_data, 0, n_mels * n_mel_frames * sizeof(float));
    
    // Set cache inputs from current state
    for (int l = 0; l < n_layers; l++) {
        // Zero-pad if cache isn't full yet
        std::vector<float> k_buf(d_model * cache_len, 0.0f);
        std::vector<float> v_buf(d_model * cache_len, 0.0f);
        
        int actual_cache = sctx->encoder_cache.attn_caches[l].cache_len;
        if (actual_cache > 0) {
            // Copy existing cache to end of buffer (right-aligned)
            int offset = cache_len - actual_cache;
            memcpy(k_buf.data() + offset * d_model, 
                   sctx->encoder_cache.attn_caches[l].k_data(),
                   actual_cache * d_model * sizeof(float));
            memcpy(v_buf.data() + offset * d_model,
                   sctx->encoder_cache.attn_caches[l].v_data(),
                   actual_cache * d_model * sizeof(float));
        }
        
        ggml_backend_tensor_set(g.k_cache_ins[l], k_buf.data(), 0, k_buf.size() * sizeof(float));
        ggml_backend_tensor_set(g.v_cache_ins[l], v_buf.data(), 0, v_buf.size() * sizeof(float));
        ggml_backend_tensor_set(g.conv_cache_ins[l], 
            sctx->encoder_cache.conv_caches[l].data(), 0,
            d_model * conv_cache_len * sizeof(float));
    }
    
    // Compute the graph (reusing pre-built structure)
    ggml_backend_graph_compute(nctx->model.backend, g.graph);
    
    // Get encoder output
    int64_t chunk_len = g.encoder_out->ne[1];
    std::vector<float> enc_out(d_model * chunk_len);
    ggml_backend_tensor_get(g.encoder_out, enc_out.data(), 0, enc_out.size() * sizeof(float));
    
    if (call_count <= 3) {
        fprintf(stderr, "[DEBUG] encoder output chunk_len=%lld\n", (long long)chunk_len);
    }
    
    // Update caches from outputs
    for (int l = 0; l < n_layers; l++) {
        if (g.k_cache_outs[l]) {
            int new_cache_len = g.k_cache_outs[l]->ne[1];
            std::vector<float> k_new(d_model * new_cache_len);
            std::vector<float> v_new(d_model * new_cache_len);
            ggml_backend_tensor_get(g.k_cache_outs[l], k_new.data(), 0, k_new.size() * sizeof(float));
            ggml_backend_tensor_get(g.v_cache_outs[l], v_new.data(), 0, v_new.size() * sizeof(float));
            
            sctx->encoder_cache.attn_caches[l].cache_len = std::min(new_cache_len, cache_len);
            memcpy((void*)sctx->encoder_cache.attn_caches[l].k_data(), k_new.data(),
                   sctx->encoder_cache.attn_caches[l].cache_len * d_model * sizeof(float));
            memcpy((void*)sctx->encoder_cache.attn_caches[l].v_data(), v_new.data(),
                   sctx->encoder_cache.attn_caches[l].cache_len * d_model * sizeof(float));
        }
        
        if (g.conv_cache_outs[l]) {
            std::vector<float> conv_new(d_model * conv_cache_len);
            ggml_backend_tensor_get(g.conv_cache_outs[l], conv_new.data(), 0, conv_new.size() * sizeof(float));
            memcpy((void*)sctx->encoder_cache.conv_caches[l].data(), conv_new.data(), conv_new.size() * sizeof(float));
        }
    }
    
    return enc_out;
}

// Run greedy decoding for a single encoder frame
// Returns: new tokens emitted for this frame
static std::vector<int> decode_encoder_frame(
    struct nemo_stream_context* sctx,
    const float* enc_frame
) {
    struct nemo_context* nctx = sctx->nctx;
    std::vector<int> new_tokens;
    
    const int d_model = sctx->config.d_model;
    const int hidden_size = sctx->config.decoder_hidden;
    const int num_layers = sctx->config.decoder_layers;
    const int vocab_size = sctx->config.vocab_size;
    const int blank_token = sctx->config.blank_token;
    const int MAX_SYMBOLS_PER_STEP = 10;
    
    for (int sym = 0; sym < MAX_SYMBOLS_PER_STEP; sym++) {
        // Create compute context
        size_t buf_size = ggml_tensor_overhead() * 100 + ggml_graph_overhead();
        std::vector<uint8_t> compute_buf(buf_size);
        
        struct ggml_init_params params = {
            .mem_size = buf_size,
            .mem_buffer = compute_buf.data(),
            .no_alloc = true,
        };
        
        struct ggml_context* ctx0 = ggml_init(params);
        if (!ctx0) break;
        
        // Create input tensors
        struct ggml_tensor* h_in = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, num_layers * hidden_size);
        struct ggml_tensor* c_in = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, num_layers * hidden_size);
        struct ggml_tensor* token_emb = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hidden_size);
        struct ggml_tensor* enc_in = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, d_model);
        ggml_set_input(h_in);
        ggml_set_input(c_in);
        ggml_set_input(token_emb);
        ggml_set_input(enc_in);
        
        // Build decoder step
        struct ggml_tensor* h_out = nullptr;
        struct ggml_tensor* c_out = nullptr;
        struct ggml_tensor* dec_out = build_decoder_step(ctx0, token_emb, h_in, c_in,
                                                          &nctx->model.decoder, &h_out, &c_out);
        
        // Build joint
        struct ggml_tensor* logits = build_joint(ctx0, enc_in, dec_out, &nctx->model.joint);
        ggml_set_output(logits);
        ggml_set_output(h_out);
        ggml_set_output(c_out);
        
        // Build graph
        struct ggml_cgraph* gf = ggml_new_graph(ctx0);
        ggml_build_forward_expand(gf, logits);
        ggml_build_forward_expand(gf, h_out);
        ggml_build_forward_expand(gf, c_out);
        
        // Allocate
        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(nctx->model.backend));
        if (!ggml_gallocr_alloc_graph(allocr, gf)) {
            ggml_gallocr_free(allocr);
            ggml_free(ctx0);
            break;
        }
        
        // Set inputs
        ggml_backend_tensor_set(h_in, sctx->decoder_state.h.data(), 0, 
                                sctx->decoder_state.h.size() * sizeof(float));
        ggml_backend_tensor_set(c_in, sctx->decoder_state.c.data(), 0,
                                sctx->decoder_state.c.size() * sizeof(float));
        
        // Get embedding for prev_token
        std::vector<float> emb_data(hidden_size);
        size_t emb_offset = sctx->decoder_state.prev_token * hidden_size * sizeof(float);
        ggml_backend_tensor_get(nctx->model.decoder.embedding, emb_data.data(), 
                                emb_offset, hidden_size * sizeof(float));
        ggml_backend_tensor_set(token_emb, emb_data.data(), 0, hidden_size * sizeof(float));
        
        ggml_backend_tensor_set(enc_in, enc_frame, 0, d_model * sizeof(float));
        
        // Compute
        ggml_backend_graph_compute(nctx->model.backend, gf);
        
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
        
        if (best_token == blank_token) {
            // Blank: move to next time step
            ggml_gallocr_free(allocr);
            ggml_free(ctx0);
            break;
        }
        
        // Non-blank: emit token and update state
        new_tokens.push_back(best_token);
        sctx->tokens.push_back(best_token);
        sctx->decoder_state.prev_token = best_token;
        
        // Update LSTM state
        ggml_backend_tensor_get(h_out, sctx->decoder_state.h.data(), 0,
                                sctx->decoder_state.h.size() * sizeof(float));
        ggml_backend_tensor_get(c_out, sctx->decoder_state.c.data(), 0,
                                sctx->decoder_state.c.size() * sizeof(float));
        
        ggml_gallocr_free(allocr);
        ggml_free(ctx0);
    }
    
    return new_tokens;
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

std::string nemo_stream_process(
    struct nemo_stream_context* sctx,
    const int16_t* audio,
    int n_samples
) {
    if (!sctx || !audio || n_samples <= 0) return "";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Add to audio buffer
    sctx->encoder_cache.audio_buffer.insert(
        sctx->encoder_cache.audio_buffer.end(),
        audio, audio + n_samples
    );
    
    sctx->total_audio_seconds += (double)n_samples / sctx->config.sample_rate;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    sctx->total_compute_seconds += elapsed.count();
    
    // For now, just accumulate audio - O(1) per chunk
    // Processing happens at finalize or get_transcript - O(N) total
    return "";
}

// Process audio in streaming chunks using the cached encoder
// This supports arbitrary-length audio since each chunk only needs
// position embeddings for the local window (cache_len + chunk_len)
// Note: Currently unused - batch encoder is faster on CPU. Keep for future GPU use.
[[maybe_unused]]
static std::string process_audio_streaming(struct nemo_stream_context* sctx) {
    struct nemo_context* nctx = sctx->nctx;
    const int n_mels = sctx->config.n_mels;
    const int subsample_factor = sctx->config.subsampling_factor;  // 8
    
    // Convert accumulated audio to mel spectrogram
    std::vector<float> mel_data;
    size_t n_mel_frames = nemo_preprocessor_process(
        nctx->preprocessor,
        sctx->encoder_cache.audio_buffer.data(),
        sctx->encoder_cache.audio_buffer.size(),
        mel_data
    );
    
    if (n_mel_frames == 0) {
        return "";
    }
    
    // Process mel frames in chunks of 8 (producing 1 encoder frame per chunk)
    // This avoids the 512-frame limit in the batch encoder
    std::vector<float> encoder_outputs;
    const int d_model = sctx->config.d_model;
    
    for (size_t mel_offset = 0; mel_offset + subsample_factor <= n_mel_frames; mel_offset += subsample_factor) {
        // Extract chunk of mel frames (8 frames -> 1 encoder frame)
        std::vector<float> mel_chunk(n_mels * subsample_factor);
        for (int t = 0; t < subsample_factor; t++) {
            for (int m = 0; m < n_mels; m++) {
                mel_chunk[m + t * n_mels] = mel_data[(mel_offset + t) * n_mels + m];
            }
        }
        // Transpose to [n_mels, time] format
        std::vector<float> mel_chunk_transposed(n_mels * subsample_factor);
        for (int t = 0; t < subsample_factor; t++) {
            for (int m = 0; m < n_mels; m++) {
                mel_chunk_transposed[m * subsample_factor + t] = mel_chunk[t * n_mels + m];
            }
        }
        
        // Process through cached encoder
        auto enc_out = process_encoder_chunk_cached(sctx, mel_chunk_transposed.data(), subsample_factor);
        if (!enc_out.empty()) {
            encoder_outputs.insert(encoder_outputs.end(), enc_out.begin(), enc_out.end());
        }
    }
    
    if (encoder_outputs.empty()) {
        fprintf(stderr, "[DEBUG] No encoder outputs collected!\n");
        return "";
    }
    
    // Decode all encoder frames
    int n_enc_frames = encoder_outputs.size() / d_model;
    fprintf(stderr, "[DEBUG] Collected %d encoder frames, now decoding...\n", n_enc_frames);
    
    // Check if encoder output looks reasonable (not all zeros)
    float enc_sum = 0.0f;
    for (size_t i = 0; i < std::min(encoder_outputs.size(), (size_t)d_model); i++) {
        enc_sum += fabsf(encoder_outputs[i]);
    }
    fprintf(stderr, "[DEBUG] First encoder frame L1 norm: %.6f\n", enc_sum);
    
    for (int f = 0; f < n_enc_frames; f++) {
        auto new_toks = decode_encoder_frame(sctx, encoder_outputs.data() + f * d_model);
        for (int tok : new_toks) {
            sctx->tokens.push_back(tok);
        }
    }
    
    fprintf(stderr, "[DEBUG] Decoded %zu tokens\n", sctx->tokens.size());
    
    // Convert tokens to text
    return tokens_to_text(sctx->tokens, nctx->model.vocab);
}

std::string nemo_stream_finalize(struct nemo_stream_context* sctx) {
    if (!sctx) return "";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::string result;
    
    if (!sctx->encoder_cache.audio_buffer.empty()) {
        size_t n_samples = sctx->encoder_cache.audio_buffer.size();
        double audio_seconds = (double)n_samples / sctx->config.sample_rate;
        
        fprintf(stderr, "[DEBUG] nemo_stream_finalize: n_samples=%zu, audio_seconds=%.2f\n",
                n_samples, audio_seconds);
        
        // Use batch encoder - now supports up to ~164 seconds with max_pos_len=2048
        result = nemo_transcribe_audio(
            sctx->nctx,
            sctx->encoder_cache.audio_buffer.data(),
            sctx->encoder_cache.audio_buffer.size()
        );
        sctx->encoder_cache.audio_buffer.clear();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    sctx->total_compute_seconds += elapsed.count();
    
    return result;
}

std::string nemo_stream_get_transcript(struct nemo_stream_context* sctx) {
    if (!sctx) return "";
    
    // For progress display: run transcription on what we have so far
    // This is O(accumulated_audio) but only called periodically
    if (sctx->encoder_cache.audio_buffer.empty()) {
        return "";
    }
    
    return nemo_transcribe_audio(
        sctx->nctx,
        sctx->encoder_cache.audio_buffer.data(),
        sctx->encoder_cache.audio_buffer.size()
    );
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
