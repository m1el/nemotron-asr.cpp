#ifndef NEMO_STREAM_H
#define NEMO_STREAM_H

#include "nemo-ggml.h"
#include <cstdint>
#include <string>
#include <vector>

// =============================================================================
// Cache Configuration
// =============================================================================

// Latency mode presets for streaming ASR
// Determines how much lookahead (right context) the encoder sees
enum class nemo_latency_mode {
    PURE_CAUSAL   = 0,   // att_right_context=0,  80ms  latency, chunk=8 mel frames
    ULTRA_LOW     = 1,   // att_right_context=1,  160ms latency, chunk=16 mel frames  
    LOW           = 6,   // att_right_context=6,  560ms latency, chunk=56 mel frames
    DEFAULT       = 13,  // att_right_context=13, 1.12s latency, chunk=112 mel frames
};

// Streaming cache configuration for Nemotron-Speech model
struct nemo_cache_config {
    // Attention cache settings
    int32_t att_left_context   = 70;     // Number of past frames to cache for attention
    int32_t att_right_context  = 0;      // Lookahead frames (0=pure causal, 1/6/13 = other modes)
    int32_t cache_drop_size    = 0;      // Frames to drop from cache per step (0 for chunked_limited)
    
    // Convolution cache settings
    int32_t conv_kernel_size   = 9;      // Depthwise conv kernel size (from model config)
    int32_t conv_cache_size    = 8;      // kernel_size - 1
    
    // Model dimensions
    int32_t d_model            = 1024;   // Model dimension
    int32_t n_layers           = 24;     // Number of conformer layers
    int32_t n_heads            = 8;      // Number of attention heads
    int32_t d_head             = 128;    // Head dimension
    
    // Subsampling settings
    int32_t subsampling_factor = 8;      // Mel frames to encoder frames ratio
    int32_t n_mels             = 128;    // Number of mel features
    
    // Audio settings
    int32_t sample_rate        = 16000;  // Audio sample rate
    int32_t hop_length         = 160;    // Mel hop length (10ms at 16kHz)
    
    // Decoder settings
    int32_t decoder_hidden     = 640;    // LSTM hidden size
    int32_t decoder_layers     = 2;      // Number of LSTM layers
    int32_t vocab_size         = 1025;   // Vocabulary size (including blank)
    int32_t blank_token        = 1024;   // Blank token ID
    
    // Compute chunk_mel_frames based on att_right_context
    // Formula: chunk_size = subsampling_factor * (1 + att_right_context)
    // For pure causal (att_right_context=0): 8 * 1 = 8 mel frames = 80ms
    // For default (att_right_context=13): 8 * 14 = 112 mel frames = 1.12s
    int32_t get_chunk_mel_frames() const {
        return subsampling_factor * (1 + att_right_context);
    }
    
    // Compute chunk audio samples based on latency mode
    // chunk_samples = chunk_mel_frames * hop_length
    int32_t get_chunk_samples() const {
        return get_chunk_mel_frames() * hop_length;
    }
    
    // Get latency in milliseconds
    int32_t get_latency_ms() const {
        return get_chunk_mel_frames() * hop_length * 1000 / sample_rate;
    }
    
    // Factory methods for different latency modes
    static nemo_cache_config default_config() {
        return with_latency(nemo_latency_mode::PURE_CAUSAL);
    }
    
    static nemo_cache_config with_latency(nemo_latency_mode mode) {
        nemo_cache_config cfg;
        cfg.att_right_context = static_cast<int32_t>(mode);
        return cfg;
    }
    
    static nemo_cache_config pure_causal() {
        return with_latency(nemo_latency_mode::PURE_CAUSAL);
    }
    
    static nemo_cache_config ultra_low_latency() {
        return with_latency(nemo_latency_mode::ULTRA_LOW);
    }
    
    static nemo_cache_config low_latency() {
        return with_latency(nemo_latency_mode::LOW);
    }
    
    static nemo_cache_config balanced() {
        return with_latency(nemo_latency_mode::DEFAULT);
    }
};

// =============================================================================
// Per-Layer Cache Structures
// =============================================================================

// Attention cache for one conformer layer
// Stores K and V for left_context past frames
struct nemo_layer_attn_cache {
    std::vector<float> k_cache;   // [cache_len, d_model] flattened
    std::vector<float> v_cache;   // [cache_len, d_model] flattened
    int32_t cache_len;            // Current valid cache length (0 to max_cache_len)
    int32_t max_cache_len;        // Maximum cache size (= att_left_context)
    int32_t d_model;              // Model dimension
    
    void init(int32_t max_len, int32_t dim);
    void reset();
    
    // Append new K/V and trim to max_cache_len
    void update(const float* k_new, const float* v_new, int32_t new_len);
    
    // Get pointers for graph building
    const float* k_data() const { return k_cache.data(); }
    const float* v_data() const { return v_cache.data(); }
};

// Convolution cache for one conformer layer
// Stores the tail of the input for causal depthwise conv1d
struct nemo_layer_conv_cache {
    std::vector<float> cache;     // [d_model, kernel_size-1] flattened, channels-first
    int32_t cache_len;            // kernel_size - 1
    int32_t d_model;              // Model dimension
    
    void init(int32_t kernel_size, int32_t dim);
    void reset();
    
    // Update cache with new data (takes last kernel_size-1 frames)
    void update(const float* new_data, int32_t seq_len);
    
    const float* data() const { return cache.data(); }
};

// =============================================================================
// Full Encoder Cache
// =============================================================================

// Complete cache state for streaming encoder
struct nemo_encoder_cache {
    nemo_cache_config config;
    
    // Per-layer caches
    std::vector<nemo_layer_attn_cache> attn_caches;  // [n_layers]
    std::vector<nemo_layer_conv_cache> conv_caches;  // [n_layers]
    
    // Subsampling cache (for mel frames that don't fit complete chunk)
    std::vector<float> mel_buffer;   // Buffered mel frames
    int32_t mel_buffer_len;          // Number of buffered mel frames

    // Audio buffer (for samples that don't fit complete chunk)
    std::vector<int16_t> audio_buffer;

    // Streaming audio buffer for STFT overlap
    // We need to keep some audio history for proper STFT windowing
    std::vector<float> audio_history;  // Float audio with pre-emphasis applied
    int32_t audio_history_len;         // Valid samples in history
    float last_sample;                 // Last sample for pre-emphasis continuity

    // Frame tracking for true streaming
    int64_t total_encoder_frames;      // Total encoder frames processed so far
    
    // Initialization and reset
    void init(const nemo_cache_config& cfg);
    void reset();
    
    // Memory usage calculation
    size_t memory_usage_bytes() const;
};

// =============================================================================
// Decoder State - defined in nemo-ggml.h
// =============================================================================
// nemo_decoder_state is defined in nemo-ggml.h

// =============================================================================
// Streaming Context
// =============================================================================

// Pre-built encoder graph for reuse across chunks
struct nemo_encoder_graph {
    struct ggml_context* ctx;          // Persistent context
    struct ggml_cgraph* graph;         // Pre-built graph
    ggml_gallocr_t allocr;             // Persistent allocator
    
    // Input tensors (set data before compute)
    struct ggml_tensor* mel_input;     // [n_mels, chunk_frames, 1]
    std::vector<struct ggml_tensor*> k_cache_ins;   // [n_layers]
    std::vector<struct ggml_tensor*> v_cache_ins;   // [n_layers]
    std::vector<struct ggml_tensor*> conv_cache_ins; // [n_layers]
    
    // Output tensors (read data after compute)
    struct ggml_tensor* encoder_out;   // [d_model, chunk_len]
    std::vector<struct ggml_tensor*> k_cache_outs;  // [n_layers]
    std::vector<struct ggml_tensor*> v_cache_outs;  // [n_layers]
    std::vector<struct ggml_tensor*> conv_cache_outs; // [n_layers]
    
    bool initialized;
    
    nemo_encoder_graph() : ctx(nullptr), graph(nullptr), allocr(nullptr), 
                           mel_input(nullptr), encoder_out(nullptr), initialized(false) {}
    ~nemo_encoder_graph();
    
    void init(struct nemo_context* nctx, const nemo_cache_config& cfg, int mel_chunk_frames);
    void reset();
};

// Full streaming session context
struct nemo_stream_context {
    // Base model context (not owned, must outlive stream context)
    struct nemo_context* nctx;

    // Cache configuration
    nemo_cache_config config;

    // Encoder caches
    nemo_encoder_cache encoder_cache;

    // Decoder state
    nemo_decoder_state decoder_state;

    // Pre-built encoder graph (reused across chunks)
    nemo_encoder_graph encoder_graph;

    // Pre-built decoder/joint graph (reused across decode steps)
    struct ggml_context* decode_ctx;
    struct ggml_cgraph* decode_graph;
    ggml_gallocr_t decode_allocr;
    struct ggml_tensor* decode_h_in;
    struct ggml_tensor* decode_c_in;
    struct ggml_tensor* decode_token_emb;
    struct ggml_tensor* decode_enc_in;
    struct ggml_tensor* decode_logits;
    struct ggml_tensor* decode_h_out;
    struct ggml_tensor* decode_c_out;
    bool decode_graph_initialized;

    // CPU-side copies of preprocessor weights for streaming mel conversion
    std::vector<float> filterbank_cpu;   // [n_mels * n_bins]
    std::vector<float> window_cpu;       // [n_window_size]

    // Accumulated tokens from this session
    std::vector<int> tokens;

    // Accumulated transcript text
    std::string transcript;
    
    // Timing stats
    double total_audio_seconds;
    double total_compute_seconds;
    
    // Initialize from model context
    void init(struct nemo_context* ctx, const nemo_cache_config& cfg);
    void reset();
    
    // Real-time factor (compute time / audio time)
    double rtf() const { 
        return total_audio_seconds > 0 ? total_compute_seconds / total_audio_seconds : 0; 
    }
};

// =============================================================================
// Streaming API
// =============================================================================

// Initialize streaming context
// Returns nullptr on failure
// The nemo_context must outlive the stream context
struct nemo_stream_context* nemo_stream_init(
    struct nemo_context* ctx,
    const nemo_cache_config* config = nullptr  // Use default if nullptr
);

// Process audio chunk and return new tokens/text
// audio: int16_t samples at config.sample_rate (16kHz), mono
// n_samples: number of samples (can be any length, will buffer internally)
// Returns: transcription of newly decoded tokens (may be empty if no new tokens)
//
// This version uses batch re-transcription for higher quality but O(N^2) complexity
std::string nemo_stream_process(
    struct nemo_stream_context* sctx,
    const int16_t* audio,
    int n_samples
);

// True incremental streaming using cached encoder
// This processes audio incrementally without re-transcribing.
// Lower latency but may have slight quality differences at chunk boundaries.
std::string nemo_stream_process_incremental(
    struct nemo_stream_context* sctx,
    const int16_t* audio,
    int n_samples
);

// Process final audio and flush any buffered data
// Returns: final transcription including any remaining tokens
std::string nemo_stream_finalize(
    struct nemo_stream_context* sctx
);

// Get full accumulated transcript so far
std::string nemo_stream_get_transcript(
    struct nemo_stream_context* sctx
);

// Get list of all tokens emitted so far
const std::vector<int>& nemo_stream_get_tokens(
    struct nemo_stream_context* sctx
);

// Reset streaming state (keep model, clear caches and transcript)
void nemo_stream_reset(
    struct nemo_stream_context* sctx
);

// Free streaming context
void nemo_stream_free(
    struct nemo_stream_context* sctx
);

// =============================================================================
// Graph Building Helpers (for cached computation)
// =============================================================================

// Build cached relative position multi-head attention
// x: [d_model, chunk_len, batch] - current chunk input
// k_cache_in, v_cache_in: [d_model, cache_len] - cached K/V from previous chunks
// Returns: output [d_model, chunk_len, batch]
// Outputs k_cache_out, v_cache_out contain updated caches
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
    int right_context,
    struct ggml_tensor** k_cache_out,   // Output: updated K cache
    struct ggml_tensor** v_cache_out    // Output: updated V cache
);

// Build cached causal depthwise conv1d
// x: [d_model, seq_len, batch] - current chunk (channels first in dim 0)
// cache_in: [d_model, kernel_size-1] - cached state from previous chunk
// Returns: output [d_model, seq_len, batch]
// Outputs cache_out for next chunk
struct ggml_tensor* build_cached_causal_conv1d(
    struct ggml_context* ctx,
    struct ggml_tensor* x,              // [d_model, seq_len, batch]
    struct ggml_tensor* cache_in,       // [d_model, kernel_size-1] or nullptr
    struct ggml_tensor* weight,         // [kernel_size, 1, d_model]
    int kernel_size,
    struct ggml_tensor** cache_out      // Output: updated cache
);

// Build cached conformer layer
// x: [d_model, chunk_len, batch]
// attn_cache_in: attention K/V cache
// conv_cache_in: convolution state cache
// Returns: output [d_model, chunk_len, batch]
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
);

// Build full streaming encoder step
// mel_chunk: [n_mels, chunk_frames, batch]
// Uses and updates encoder_cache internally
// Returns: [d_model, valid_out_len, batch]
struct ggml_tensor* build_streaming_encoder(
    struct ggml_context* ctx,
    struct ggml_tensor* mel_chunk,
    nemo_model* model,
    nemo_encoder_cache* cache
);

#endif // NEMO_STREAM_H
