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

    // Streaming post-processing settings (from NeMo streaming_cfg)
    int32_t valid_out_len      = 1;      // Number of encoder frames to output per chunk
    int32_t drop_extra_pre_encoded = 2;  // Frames to drop from start after subsampling
    int32_t last_channel_cache_size = 70; // Max size for attention cache (same as att_left_context)
    int32_t pre_encode_cache_size = 9;   // Overlap mel frames for conv subsampling context
    int32_t shift_mel_frames   = 8;      // Mel frames to advance per chunk (NeMo shift_size)
    
    // Compute chunk_mel_frames based on att_right_context
    // This is the total mel frames needed for one encoder step (including overlap)
    // Formula: chunk_size = pre_encode_cache_size + shift_mel_frames
    // For pure causal: 9 + 8 = 17 mel frames = 170ms input
    // For default (att_right_context=13): 9 + 8*(1+13) = 121 mel frames
    size_t get_chunk_mel_frames() const {
        // NeMo formula: sampling_frames[1] + subsampling_factor * lookahead_steps
        // where lookahead_steps = att_right_context
        // Plus pre_encode_cache_size for the overlap
        int32_t lookahead_steps = att_right_context;
        int32_t chunk_without_cache = subsampling_factor + subsampling_factor * lookahead_steps;
        return pre_encode_cache_size + chunk_without_cache;
    }

    // Get the number of mel frames to shift/advance per chunk
    // This determines how many new mel frames are consumed each step
    size_t get_shift_mel_frames() const {
        // NeMo formula: shift_size[1] = sampling_frames[1] + sampling_frames[1] * (lookahead_steps - cache_drop_size)
        // For pure causal with cache_drop_size=0: 8 + 8*0 = 8
        int32_t lookahead_steps = att_right_context;
        return subsampling_factor + subsampling_factor * (lookahead_steps - cache_drop_size);
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
    struct ggml_tensor* attn_mask;     // [kv_len, 1] attention mask for invalid cache positions
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
                           mel_input(nullptr), attn_mask(nullptr), encoder_out(nullptr), initialized(false) {}
    ~nemo_encoder_graph();
    
    void init(struct nemo_context* nctx, const nemo_cache_config& cfg);
    void build_streaming_encoder(
        struct ggml_context * ctx,
        struct nemo_context* nctx,
        const nemo_cache_config& cfg,
        size_t drop_extra_preencoded
    );
    void reset();
};

// Full streaming session context
struct nemo_stream_context {
    // Base model context (not owned, must outlive stream context)
    struct nemo_context* nctx;

    // Cache configuration
    nemo_cache_config config;

    // Decoder state
    nemo_decoder_state decoder_state;

    // Pre-built encoder graph (reused across chunks)
    nemo_encoder_graph encoder_graph;

    // Overlap settings (derived from config)
    size_t overlap_mel_frames() const {
        return config.pre_encode_cache_size;
    }

    size_t shift_mel_frames() const {
        return config.get_shift_mel_frames();
    }

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

    struct ggml_tensor* cache_last_channel;
    struct ggml_tensor* cache_next_channel;
    struct ggml_tensor* cache_last_time;
    struct ggml_tensor* cache_next_time;
    struct ggml_tensor* cache_last_channel_len;
    struct ggml_tensor* cache_next_channel_len;
    bool decode_graph_initialized;

    // MEL buffer
    std::vector<float> mel_buffer;

    // Accumulated tokens from this session
    std::vector<int> tokens;

    // Accumulated transcript text
    std::string transcript;
    
    // Timing stats
    double total_audio_seconds;
    double total_compute_seconds;

    // Detailed timing breakdown (for profiling)
    double encoder_seconds;
    double decoder_seconds;
    double transfer_seconds;
    int total_decode_iterations;  // Total decoder loop iterations

    // Cache validity tracking (for attention masking)
    // Starts at 0, grows by chunk_len each chunk, capped at cache_len
    int cache_valid_len;

    // Chunk counter for debugging
    int total_chunks_processed;

    // Initialize from model context
    void init(struct nemo_context* ctx, const nemo_cache_config& cfg);
    void reset();
    ~nemo_stream_context();
    
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
    struct ggml_tensor* attn_mask,      // [kv_len, 1] attention mask (0=valid, -1e9=masked)
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
    struct ggml_tensor* attn_mask,      // [kv_len, 1] attention mask (0=valid, -1e9=masked)
    nemo_conformer_layer* layer,
    const nemo_cache_config* config,
    struct ggml_tensor** k_cache_out,
    struct ggml_tensor** v_cache_out,
    struct ggml_tensor** conv_cache_out,
    int layer_idx                       // Layer index for debugging
);

// Build full streaming encoder step
// mel_chunk: [n_mels, chunk_frames, batch]
// Returns: [d_model, valid_out_len, batch]
struct ggml_tensor* build_streaming_encoder(
    struct ggml_context* ctx,
    struct ggml_tensor* mel_chunk,
    nemo_model* model
);

#endif // NEMO_STREAM_H
