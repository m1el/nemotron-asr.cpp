// C wrapper implementation for nemotron-asr.cpp
// Provides C-compatible FFI interface for Rust and other languages

#include "nemotron_asr_c.h"
#include "nemo-ggml.h"
#include "nemo-stream.h"

#include <cstring>
#include <cstdlib>

// =============================================================================
// Internal Type Casts
// =============================================================================

// Cast between C FFI and C++ types
// These are safe because the FFI types are opaque pointers to the C++ types
static inline ::nemo_context* to_cpp_ctx(nemo_context_ffi* ctx) {
    return reinterpret_cast<::nemo_context*>(ctx);
}

static inline nemo_context_ffi* to_ffi_ctx(::nemo_context* ctx) {
    return reinterpret_cast<nemo_context_ffi*>(ctx);
}

static inline ::nemo_stream_context* to_cpp_sctx(nemo_stream_context_ffi* sctx) {
    return reinterpret_cast<::nemo_stream_context*>(sctx);
}

static inline nemo_stream_context_ffi* to_ffi_sctx(::nemo_stream_context* sctx) {
    return reinterpret_cast<nemo_stream_context_ffi*>(sctx);
}

// Allocate and copy C++ string to C string
static char* copy_string(const std::string& str) {
    if (str.empty()) {
        // Return empty string, not NULL
        char* result = (char*)malloc(1);
        if (result) {
            result[0] = '\0';
        }
        return result;
    }

    char* result = (char*)malloc(str.length() + 1);
    if (result) {
        std::memcpy(result, str.c_str(), str.length() + 1);
    }
    return result;
}

// =============================================================================
// Configuration Helper Functions
// =============================================================================

extern "C" {

// Note: nemo_cache_config helper functions are implemented in nemo-stream.cpp

// =============================================================================
// Model Initialization and Cleanup
// =============================================================================

nemo_context_ffi* c_nemo_init_with_backend(const char* model_path, const char* backend_name) {
    if (!model_path) return nullptr;

    ::nemo_context* ctx = ::nemo_init_with_backend(model_path, backend_name);
    return to_ffi_ctx(ctx);
}

void c_nemo_free(nemo_context_ffi* ctx) {
    if (ctx) {
        ::nemo_free(to_cpp_ctx(ctx));
    }
}

const char* c_nemo_get_backend_name(nemo_context_ffi* ctx) {
    if (!ctx) return nullptr;
    return ::nemo_get_backend_name(to_cpp_ctx(ctx));
}

// =============================================================================
// Streaming API
// =============================================================================

nemo_stream_context_ffi* c_nemo_stream_init(
    nemo_context_ffi* ctx,
    const nemo_cache_config* config
) {
    if (!ctx) return nullptr;

    // config is already C-compatible, can pass directly
    ::nemo_stream_context* sctx = ::nemo_stream_init(to_cpp_ctx(ctx), config);
    return to_ffi_sctx(sctx);
}

char* c_nemo_stream_process_incremental(
    nemo_stream_context_ffi* sctx,
    const int16_t* audio,
    int n_samples
) {
    if (!sctx || !audio) {
        return copy_string("");
    }

    std::string result = ::nemo_stream_process_incremental(
        to_cpp_sctx(sctx),
        audio,
        n_samples
    );

    return copy_string(result);
}

char* c_nemo_stream_finalize(nemo_stream_context_ffi* sctx) {
    if (!sctx) {
        return copy_string("");
    }

    std::string result = ::nemo_stream_finalize(to_cpp_sctx(sctx));
    return copy_string(result);
}

char* c_nemo_stream_get_transcript(nemo_stream_context_ffi* sctx) {
    if (!sctx) {
        return copy_string("");
    }

    std::string result = ::nemo_stream_get_transcript(to_cpp_sctx(sctx));
    return copy_string(result);
}

void c_nemo_stream_reset(nemo_stream_context_ffi* sctx) {
    if (sctx) {
        ::nemo_stream_reset(to_cpp_sctx(sctx));
    }
}

void c_nemo_stream_free(nemo_stream_context_ffi* sctx) {
    if (sctx) {
        ::nemo_stream_free(to_cpp_sctx(sctx));
    }
}

// =============================================================================
// Memory Management
// =============================================================================

void c_nemo_free_string(char* str) {
    if (str) {
        free(str);
    }
}

} // extern "C"
