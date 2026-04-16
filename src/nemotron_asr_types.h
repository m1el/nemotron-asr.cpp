#ifndef NEMOTRON_ASR_TYPES_H
#define NEMOTRON_ASR_TYPES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Latency Mode Enum
// =============================================================================

// Latency mode presets for streaming ASR
// Determines how much lookahead (right context) the encoder sees
typedef enum {
    NEMO_LATENCY_PURE_CAUSAL = 0,   // att_right_context=0,  80ms  latency, chunk=8 mel frames
    NEMO_LATENCY_ULTRA_LOW   = 1,   // att_right_context=1,  160ms latency, chunk=16 mel frames
    NEMO_LATENCY_LOW         = 6,   // att_right_context=6,  560ms latency, chunk=56 mel frames
    NEMO_LATENCY_DEFAULT     = 13,  // att_right_context=13, 1.12s latency, chunk=112 mel frames
} nemo_latency_mode;

// =============================================================================
// Cache Configuration Struct
// =============================================================================

// Streaming cache configuration for Nemotron-Speech model
// This is a C-compatible POD struct with optional C++ convenience methods
struct nemo_cache_config {
    // Attention cache settings
    int32_t att_left_context;      // Number of past frames to cache for attention (default: 70)
    int32_t att_right_context;     // Lookahead frames (0=pure causal, 1/6/13 = other modes, default: 0)
    int32_t cache_drop_size;       // Frames to drop from cache per step (default: 0 for chunked_limited)

    // Convolution cache settings
    int32_t conv_kernel_size;      // Depthwise conv kernel size (default: 9)
    int32_t conv_cache_size;       // kernel_size - 1 (default: 8)

    // Model dimensions
    int32_t d_model;               // Model dimension (default: 1024)
    int32_t n_layers;              // Number of conformer layers (default: 24)
    int32_t n_heads;               // Number of attention heads (default: 8)
    int32_t d_head;                // Head dimension (default: 128)

    // Subsampling settings
    int32_t subsampling_factor;    // Mel frames to encoder frames ratio (default: 8)
    int32_t n_mels;                // Number of mel features (default: 128)

    // Audio settings
    int32_t sample_rate;           // Audio sample rate (default: 16000)
    int32_t hop_length;            // Mel hop length (default: 160, 10ms at 16kHz)

    // Decoder settings
    int32_t decoder_hidden;        // LSTM hidden size (default: 640)
    int32_t decoder_layers;        // Number of LSTM layers (default: 2)
    int32_t vocab_size;            // Vocabulary size (default: 1025, including blank)
    int32_t blank_token;           // Blank token ID (default: 1024)

    // Streaming post-processing settings (from NeMo streaming_cfg)
    int32_t drop_extra_pre_encoded;    // Frames to drop from start after subsampling (default: 2)
    int32_t last_channel_cache_size;   // Max size for attention cache (default: 70)
    int32_t pre_encode_cache_size;     // Overlap mel frames for conv subsampling context (default: 9)
    int32_t shift_mel_frames;          // Mel frames to advance per chunk (default: 8)
};

// =============================================================================
// Helper Functions
// =============================================================================

// Get chunk size in mel frames
size_t nemo_cache_config_get_chunk_mel_frames(const struct nemo_cache_config* config);

// Get shift mel frames value
size_t nemo_cache_config_get_shift_mel_frames_val(const struct nemo_cache_config* config);

// Get chunk size in audio samples
int32_t nemo_cache_config_get_chunk_samples(const struct nemo_cache_config* config);

// Get latency in milliseconds
int32_t nemo_cache_config_get_latency_ms(const struct nemo_cache_config* config);

// Get valid output length
int32_t nemo_cache_config_get_valid_out_len(const struct nemo_cache_config* config);

// Create default cache configuration
struct nemo_cache_config nemo_cache_config_default(void);

// Create cache configuration with specific latency mode
struct nemo_cache_config nemo_cache_config_with_latency(nemo_latency_mode mode);

#ifdef __cplusplus
}
#endif

#endif // NEMOTRON_ASR_TYPES_H
