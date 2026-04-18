#ifndef NEMOTRON_ASR_C_H
#define NEMOTRON_ASR_C_H

#include <stdint.h>
#include <stddef.h>

// Include the shared C-compatible types
#include "nemotron_asr_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Opaque Types
// =============================================================================

// Main model context (opaque pointer to C++ nemo_context)
// The _ffi suffix distinguishes the opaque C handle from the C++ type
typedef struct nemo_context nemo_context_ffi;

// Streaming context (opaque pointer to C++ nemo_stream_context)
typedef struct nemo_stream_context nemo_stream_context_ffi;

// Note: nemo_cache_config, nemo_latency_mode, and their helper functions
// are defined in nemotron_asr_types.h

// =============================================================================
// Model Initialization and Cleanup
// =============================================================================

// Initialize model context with specific backend
// backend_name: "CPU", "CUDA", "Vulkan", "Metal", etc., or NULL for auto-select
// Returns: context pointer on success, NULL on failure
nemo_context_ffi* c_nemo_init_with_backend(const char* model_path, const char* backend_name);

// Free model context
void c_nemo_free(nemo_context_ffi* ctx);

// Get current backend name
// Returns: backend name string (owned by context, do not free)
const char* c_nemo_get_backend_name(nemo_context_ffi* ctx);

// =============================================================================
// Streaming API
// =============================================================================

// Initialize streaming context
// config: cache configuration, or NULL to use default
// Returns: streaming context pointer on success, NULL on failure
nemo_stream_context_ffi* c_nemo_stream_init(
    nemo_context_ffi* ctx,
    const struct nemo_cache_config* config
);

// Process audio chunk incrementally
// audio: int16_t PCM samples at 16kHz mono
// n_samples: number of samples
// Returns: C string with new transcription (may be empty), caller must free with c_nemo_free_string
char* c_nemo_stream_process_incremental(
    nemo_stream_context_ffi* sctx,
    const int16_t* audio,
    int n_samples
);

// Finalize streaming and flush remaining audio
// Returns: C string with final transcription, caller must free with c_nemo_free_string
char* c_nemo_stream_finalize(nemo_stream_context_ffi* sctx);

// Get full accumulated transcript
// Returns: C string with full transcript, caller must free with c_nemo_free_string
char* c_nemo_stream_get_transcript(nemo_stream_context_ffi* sctx);

// Reset streaming state (clear caches and transcript)
void c_nemo_stream_reset(nemo_stream_context_ffi* sctx);

// Free streaming context
void c_nemo_stream_free(nemo_stream_context_ffi* sctx);

// =============================================================================
// Memory Management
// =============================================================================

// Free C string returned by library functions
void c_nemo_free_string(char* str);

#ifdef __cplusplus
}
#endif

#endif // NEMOTRON_ASR_C_H
