# Cache-Aware Streaming ASR Implementation

**Last Updated**: 2026-01-27

## Overview

This document details the cache-aware streaming architecture for the Nemotron-Speech model (`nemotron-speech-streaming-en-0.6b`) and its implementation in GGML.

The model is specifically designed for streaming ASR with:
- **Causal convolutions** in subsampling and conformer layers
- **Limited attention context** (chunked_limited style)
- **K/V caching** for attention layers
- **Convolution state caching** for depthwise conv1d

## Architecture Summary

```
Audio Chunk (1280 samples = 80ms)
         ↓
   Preprocessor (Mel Spectrogram)
         ↓ [n_mels=128, ~8 frames per chunk]
   Conv Subsampling (8x)  + cache
         ↓ [d_model=1024, 1 frame per 80ms chunk]  
   24× Conformer Layers  + K/V cache + conv cache
         ↓ [d_model=1024, 1 frame]
   RNN-T Decoder (LSTM)  + hidden state
         ↓
   Joint Network → Tokens
```

## Model Configuration

From `model_config.yaml`:

```yaml
encoder:
  _target_: nemo.collections.asr.modules.ConformerEncoder
  feat_in: 128                    # mel features
  n_layers: 24                    # conformer layers
  d_model: 1024                   # model dimension
  n_heads: 8                      # attention heads (d_head = 128)
  
  # Streaming-specific settings:
  att_context_style: chunked_limited    # Limited context attention
  att_context_size:                     # Multiple supported configs
    - [70, 13]   # 70 left, 13 right (1.12s latency)
    - [70, 6]    # 70 left, 6 right  (0.56s latency)
    - [70, 1]    # 70 left, 1 right  (0.16s latency)
    - [70, 0]    # 70 left, 0 right  (pure causal, 80ms latency)
  
  conv_context_size: causal       # Causal convolutions
  causal_downsampling: true       # Causal subsampling
  conv_kernel_size: 9             # Depthwise conv kernel size

decoder:
  pred_hidden: 640                # LSTM hidden size
  pred_rnn_layers: 2              # Number of LSTM layers
  vocab_size: 1024                # + blank = 1025 total
```

## Attention Context Explained

Each 80ms audio chunk (after 8x subsampling) = 1 encoder frame. The `att_context_size=[L, R]` defines:
- **L (left context)**: Number of past frames each position can attend to
- **R (right context)**: Number of future frames (lookahead) each position can attend to

| Mode | Left | Right | Chunk Duration | Use Case |
|------|------|-------|----------------|----------|
| [70, 0] | 70 | 0 | 80ms | Ultra-low latency, pure causal |
| [70, 1] | 70 | 1 | 160ms | Very low latency |
| [70, 6] | 70 | 6 | 560ms | Low latency |
| [70, 13] | 70 | 13 | 1120ms | Higher quality |

The **left context of 70** means each frame can attend to the previous 70 frames = **5.6 seconds** of audio history.

## Cache Structures

### 1. Attention Cache (`cache_last_channel`)

**Purpose**: Store K/V states from past frames for attention context.

**Shape**: `[n_layers, batch, cache_size, d_model]` = `[24, 1, 70, 1024]`

**Size**: 24 × 70 × 1024 × 4 bytes = **6.9 MB** per stream

**NeMo Implementation** (from `MultiHeadAttention.update_cache()`):
```python
def update_cache(self, key, value, query, cache):
    if cache is not None:
        # Prepend cached K/V to current K/V
        key = value = torch.cat([cache, key], dim=1)
        
        # Update cache: keep query positions for next step
        q_keep_size = query.shape[1] - self.cache_drop_size
        cache = torch.cat([
            cache[:, q_keep_size:, :],      # Drop oldest entries
            query[:, :q_keep_size, :]       # Add new query positions
        ], dim=1)
    return key, value, query, cache
```

**Key insight**: In cache-aware mode with `cache_drop_size=0` (for chunked_limited), the cache is:
- Updated by appending new frames
- Trimmed to keep only `last_channel_cache_size=70` frames
- The full K/V sequence for attention becomes: `[cache_70_frames, current_chunk_frames]`

### 2. Convolution Cache (`cache_last_time`)

**Purpose**: Store input state for causal depthwise conv1d.

**Shape**: `[n_layers, batch, d_model, kernel_size-1]` = `[24, 1, 1024, 8]`

**Size**: 24 × 1024 × 8 × 4 bytes = **768 KB** per stream

**NeMo Implementation** (from `CausalConv1D.update_cache()`):
```python
def update_cache(self, x, cache=None):
    if cache is None:
        # First chunk: pad left
        new_x = F.pad(x, pad=(self._left_padding, self._right_padding))
        next_cache = None
    else:
        # Subsequent chunks: prepend cache
        new_x = F.pad(x, pad=(0, self._right_padding))
        new_x = torch.cat([cache, new_x], dim=-1)
        
        # Update cache: keep last kernel_size-1 elements
        if self.cache_drop_size > 0:
            next_cache = new_x[:, :, :-self.cache_drop_size]
        else:
            next_cache = new_x
        next_cache = next_cache[:, :, -cache.size(-1):]
    return new_x, next_cache
```

### 3. Subsampling Cache (Optional)

For pure streaming, the subsampling layer also needs caching:
- **Pre-encode cache size**: Number of mel frames to retain for next chunk
- Handles boundary conditions at 2D conv layer edges

**Shape**: Model-dependent, typically small compared to encoder caches.

### 4. Decoder State (LSTM)

**Purpose**: Maintain LSTM hidden/cell state across chunks.

**Shape**: `[2 * hidden_size]` for h and c each = `[1280]` floats

**Size**: 2 × 1280 × 4 bytes = **10 KB** per stream

The LSTM state is **only updated when emitting non-blank tokens**.

## Streaming Inference Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Streaming Session                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  INITIALIZATION:                                                     │
│    cache_attn[24] = zeros(70, 1024)   # K/V cache per layer         │
│    cache_conv[24] = zeros(1024, 8)    # Conv state per layer        │
│    cache_sub = zeros(...)             # Subsampling cache           │
│    decoder_h = zeros(1280)            # LSTM hidden state           │
│    decoder_c = zeros(1280)            # LSTM cell state             │
│    prev_token = blank                                                │
│                                                                      │
│  FOR EACH AUDIO CHUNK (80ms = 1280 samples @ 16kHz):                │
│                                                                      │
│    1. PREPROCESS                                                     │
│       mel = mel_spectrogram(chunk)    # ~8 frames × 128 mels        │
│                                                                      │
│    2. CONV SUBSAMPLING (with caching)                               │
│       enc_input, cache_sub = conv_subsample(mel, cache_sub)         │
│       # 8x downsampling: 8 mel frames → 1 encoder frame             │
│                                                                      │
│    3. FOR EACH CONFORMER LAYER (l = 0..23):                         │
│                                                                      │
│       3a. FFN1 (no caching needed - stateless)                      │
│           x = x + 0.5 * ffn1(layer_norm(x))                         │
│                                                                      │
│       3b. SELF-ATTENTION (with K/V caching)                         │
│           # Prepend cached K/V                                       │
│           k_full = concat(cache_attn[l], k_current)                 │
│           v_full = concat(cache_attn[l], v_current)                 │
│           # Attention over [cache + current]                        │
│           attn_out = rel_pos_mha(q, k_full, v_full, pos_emb)       │
│           # Update cache                                             │
│           cache_attn[l] = k_full[-70:]  # Keep last 70 frames       │
│           x = x + attn_out                                           │
│                                                                      │
│       3c. CONV MODULE (with state caching)                          │
│           # Prepend cached state for causal conv                    │
│           x_padded = concat(cache_conv[l], x)                       │
│           conv_out = depthwise_conv1d(x_padded, kernel_size=9)      │
│           # Update cache                                             │
│           cache_conv[l] = x_padded[-8:]  # Keep last 8 frames       │
│           x = x + conv_out                                           │
│                                                                      │
│       3d. FFN2 (no caching needed - stateless)                      │
│           x = x + 0.5 * ffn2(layer_norm(x))                         │
│                                                                      │
│       3e. FINAL LAYERNORM                                           │
│           x = layer_norm(x)                                          │
│                                                                      │
│    4. VALID OUTPUT SELECTION                                        │
│       # For right_context > 0, output may be unstable at edges      │
│       valid_out = encoder_out[:valid_out_len]                       │
│                                                                      │
│    5. RNN-T GREEDY DECODE (per encoder frame)                       │
│       FOR t in valid_frames:                                        │
│         WHILE True:                                                  │
│           emb = embedding[prev_token]                                │
│           dec_out, h_new, c_new = lstm(emb, decoder_h, decoder_c)   │
│           logits = joint(encoder_out[t], dec_out)                   │
│           token = argmax(logits)                                     │
│           IF token == blank:                                         │
│             BREAK  # Move to next time step                         │
│           ELSE:                                                      │
│             emit(token)                                              │
│             prev_token = token                                       │
│             decoder_h, decoder_c = h_new, c_new                     │
│                                                                      │
│    6. RETURN partial transcript                                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Attention Mask for Chunked Limited

For `att_context_size=[L, R]` with chunked_limited style:

```python
def create_attention_mask(seq_len, left_context, right_context, cache_len):
    """
    Create attention mask for cache-aware streaming.
    
    With cache: query positions attend to [cache_len + current positions]
    key_len = cache_len + seq_len
    
    mask[i, j] = True (masked) if:
        - j < cache_len - left_context  (too far in past)
        - j > cache_len + i + right_context  (too far in future)
    """
    total_len = cache_len + seq_len
    mask = torch.ones(seq_len, total_len, dtype=torch.bool)
    
    for i in range(seq_len):
        # Query at position i (within current chunk)
        query_pos = cache_len + i
        
        # Key positions that are visible
        left_bound = max(0, query_pos - left_context)
        right_bound = min(total_len, query_pos + right_context + 1)
        
        mask[i, left_bound:right_bound] = False
    
    return mask
```

For pure causal (`right_context=0`):
- Each query position only sees positions ≤ itself
- Simple lower-triangular mask over the extended [cache + current] sequence

## Positional Encoding with Cache

When using cached K/V, positional encodings need adjustment:

```python
# pos_emb covers positions [-(max_len-1), ..., 0, ..., +(max_len-1)]
# For rel_pos attention with cache:

def get_pos_emb_for_cached_attention(seq_len, cache_len, pos_emb_full):
    """
    Get positional embeddings for cache-aware attention.
    
    For query at position q (0-indexed in current chunk):
      - Query's absolute position = cache_len + q
      - Key at position k has absolute position = k (0 to cache_len + seq_len - 1)
      - Relative position = k - (cache_len + q) = k - cache_len - q
    
    For rel_shift operation:
      - Need embeddings from -(seq_len-1) to +(cache_len + seq_len - 1)
      - Total: cache_len + 2*seq_len - 1 positions
    """
    needed_len = cache_len + 2 * seq_len - 1
    center = max_len - 1  # Position of 0 in pos_emb_full
    
    start = center - (seq_len - 1)  # Most negative relative position
    end = start + needed_len
    
    return pos_emb_full[:, start:end]
```

## GGML Implementation Plan

### Phase 1: Cache Structures

```cpp
// Cache configuration
struct nemo_cache_config {
    int32_t att_left_context  = 70;      // Left attention context
    int32_t att_right_context = 0;       // Right attention context (0 for pure causal)
    int32_t conv_kernel_size  = 9;       // Depthwise conv kernel size
    int32_t conv_cache_size   = 8;       // kernel_size - 1
    int32_t d_model           = 1024;    // Model dimension
    int32_t n_layers          = 24;      // Number of conformer layers
};

// Per-layer cache
struct nemo_layer_cache {
    std::vector<float> k_cache;   // [cache_size * d_model]
    std::vector<float> v_cache;   // [cache_size * d_model]  
    std::vector<float> conv_cache; // [d_model * (kernel_size-1)]
    int32_t cache_len;            // Current valid cache length (0 to cache_size)
};

// Full encoder cache
struct nemo_encoder_cache {
    nemo_cache_config config;
    std::vector<nemo_layer_cache> layer_caches;  // [n_layers]
    
    // Subsampling cache (for streaming mel input)
    std::vector<float> sub_cache;
    
    void init(const nemo_cache_config& cfg);
    void reset();
};
```

### Phase 2: Cached Attention

```cpp
// Build cached relative position MHA
// x: [d_model, chunk_len, batch] - current chunk input
// k_cache, v_cache: [d_model, cache_len] - cached K/V from previous chunks
// Returns: output [d_model, chunk_len, batch] and updated k_cache, v_cache
struct ggml_tensor* build_cached_rel_pos_mha(
    struct ggml_context* ctx,
    struct ggml_tensor* x,           // [d_model, chunk_len, batch]
    struct ggml_tensor* k_cache,     // [d_model, cache_len]
    struct ggml_tensor* v_cache,     // [d_model, cache_len]  
    struct ggml_tensor* pos_emb,     // [d_model, pos_len]
    nemo_conformer_layer* layer,
    int n_heads,
    int d_head,
    int left_context,
    int right_context,
    struct ggml_tensor** k_cache_out, // Updated K cache
    struct ggml_tensor** v_cache_out  // Updated V cache
);
```

### Phase 3: Cached Convolution

```cpp
// Build cached causal depthwise conv1d
// x: [d_model, seq_len, batch] - current chunk (channels first)
// cache: [d_model, kernel_size-1] - cached state from previous chunk
// Returns: output [d_model, seq_len, batch] and updated cache
struct ggml_tensor* build_cached_causal_conv1d(
    struct ggml_context* ctx,
    struct ggml_tensor* x,           // [d_model, seq_len, batch]
    struct ggml_tensor* cache,       // [d_model, kernel_size-1]
    struct ggml_tensor* weight,      // [kernel_size, 1, d_model]
    int kernel_size,
    struct ggml_tensor** cache_out   // Updated cache
);
```

### Phase 4: Full Streaming Encoder

```cpp
// Streaming encoder step
// mel_chunk: [n_mels, chunk_frames, batch] - mel spectrogram for this chunk
// encoder_cache: Full encoder cache state
// Returns: [d_model, valid_out_len, batch] encoder output for this chunk
struct ggml_tensor* build_streaming_encoder_step(
    struct ggml_context* ctx,
    struct ggml_tensor* mel_chunk,
    nemo_model* model,
    nemo_encoder_cache* cache
);
```

### Phase 5: Streaming Context API

```cpp
// Streaming session context
struct nemo_stream_context {
    nemo_context* nctx;              // Base model context
    nemo_encoder_cache encoder_cache; // Encoder caches
    
    // Decoder state
    std::vector<float> decoder_h;    // [2 * hidden_size]
    std::vector<float> decoder_c;    // [2 * hidden_size]
    int prev_token;
    
    // Audio buffer for incomplete chunks
    std::vector<int16_t> audio_buffer;
    
    // Accumulated transcript
    std::string transcript;
};

// API
struct nemo_stream_context* nemo_stream_init(struct nemo_context* ctx);
std::string nemo_stream_process(struct nemo_stream_context* sctx, 
                                 const int16_t* audio, int n_samples);
std::string nemo_stream_finalize(struct nemo_stream_context* sctx);
void nemo_stream_free(struct nemo_stream_context* sctx);
```

## Testing Strategy

### Unit Tests (in order)

1. **Cache initialization**: Verify all caches initialize to zeros
2. **Cached conv1d**: Compare with non-cached version on same input
3. **Cached attention**: Compare with non-cached version on same input
4. **Single layer cached**: Process multiple chunks, verify consistency
5. **Full encoder cached**: Compare output with full audio vs. chunked
6. **Greedy decode with state**: Verify LSTM state preservation

### Integration Tests

1. **Short audio chunked**: 2-second audio, verify same output as batch
2. **Long audio chunked**: 30-second audio, verify reasonable latency
3. **Real-time simulation**: 80ms chunks with timing, verify RTF < 1.0

### Reference Generation

Generate reference outputs from NeMo Python:
```python
# Generate chunk-by-chunk reference outputs
for chunk_idx, chunk in enumerate(chunks):
    enc_out, cache_channel, cache_time = model.encoder.cache_aware_stream_step(
        mel_chunk, 
        cache_last_channel=cache_channel,
        cache_last_time=cache_time
    )
    # Save enc_out as reference
```

## Performance Considerations

### Memory Usage per Stream

| Component | Size | Formula |
|-----------|------|---------|
| Attention cache | 6.9 MB | 24 × 70 × 1024 × 4 |
| Conv cache | 768 KB | 24 × 1024 × 8 × 4 |
| Decoder state | 10 KB | 2 × 1280 × 4 |
| Audio buffer | ~10 KB | 1280 × 4 (80ms) |
| **Total** | **~7.7 MB** | Per concurrent stream |

### Compute Complexity

| Operation | Non-cached | Cached (per chunk) |
|-----------|------------|-------------------|
| Encoder | O(T²) | O(C × (L + C)) |
| Attention | O(T² × L × D) | O(C × (L + C) × D) |
| Conv | O(T × K × D) | O(C × K × D) |

Where:
- T = total sequence length
- C = chunk length (typically 1)
- L = left context (70)
- K = kernel size (9)
- D = d_model (1024)

**Speedup**: For 10-second audio (125 frames), cached is ~60× faster per chunk.

## Implementation Status

### Completed ✅

1. **Cache Structures** (`nemo-stream.h`):
   - `nemo_cache_config` - Configuration for streaming
   - `nemo_layer_attn_cache` - K/V cache per layer with sliding window update
   - `nemo_layer_conv_cache` - Conv state cache per layer
   - `nemo_encoder_cache` - Full encoder cache collection
   - `nemo_decoder_state` - LSTM h/c state
   - `nemo_stream_context` - Complete streaming session

2. **Cached Operations** (`nemo-stream.cpp`):
   - `build_cached_causal_conv1d()` - ✅ Tested, matches non-cached output
   - `build_cached_rel_pos_mha()` - Implemented (needs full testing)
   - `build_cached_conformer_layer()` - Implemented (needs full testing)

3. **Streaming API**:
   - `nemo_stream_init()` - ✅ Working
   - `nemo_stream_process()` - ✅ Working (accumulates audio)
   - `nemo_stream_get_transcript()` - ✅ Working (runs full encoder)
   - `nemo_stream_finalize()` - ✅ Working
   - `nemo_stream_free()` - ✅ Working

4. **Tests** (`test_streaming.cpp`):
   - Cache initialization - ✅ PASS
   - Attention cache update - ✅ PASS
   - Conv cache update - ✅ PASS
   - Cached conv1d equivalence - ✅ PASS (max_diff=0)
   - Decoder state persistence - ✅ PASS
   - Stream context lifecycle - ✅ PASS

5. **Example** (`transcribe_stream.cpp`):
   - ✅ Working demo showing progressive transcription
   - Currently uses O(n²) re-encoding fallback

### In Progress 🔄

1. **Cached Encoder Pipeline**:
   - Need to wire up `build_cached_conformer_layer` in `nemo_stream_process`
   - Handle subsampling caching for partial frames

### Not Started ⏳

1. **Streaming Subsampling**:
   - Conv2D subsampling needs state caching for streaming

2. **End-to-End Cached Test**:
   - Verify cached output matches full-sequence output

## Known Limitations

### 1. Cache Warmup Period (~8 seconds)

The current implementation uses zero-initialized attention caches. During the first ~8 seconds of audio, the attention operates over mostly zero K/V values, which causes the encoder output to be attenuated. This results in the decoder predicting only blank tokens during this warmup period.

**Workarounds**:
- Accept ~8 second latency before first transcription appears
- Use the non-cached encoder for the first chunk of audio to initialize caches
- Accumulate audio and process in larger batches initially

**Proper Fix** (TODO):
- Implement attention masking based on valid cache length
- Only attend to positions with valid cached data

### 2. Per-Layer Right Context Not Implemented

The model configuration specifies different right contexts per layer group:
- Layers 0-5: right_context = 13
- Layers 6-11: right_context = 6
- Layers 12-17: right_context = 1
- Layers 18-23: right_context = 0

The current implementation uses a uniform right_context for all layers, which may affect quality.

### 3. Position Embedding Mismatch

When cache contains zeros, the position embeddings may not align correctly with the actual data positions, leading to degraded attention patterns.

## Files

1. **`src-ggml/nemo-stream.h`**: Streaming API declarations
2. **`src-ggml/nemo-stream.cpp`**: Streaming implementation
3. **`tests-ggml/test_streaming.cpp`**: Streaming unit tests
4. **`examples/transcribe_stream.cpp`**: Streaming transcription example
5. **`Makefile.ggml`**: Streaming targets

## References

- [NeMo Cache-Aware Streaming Conformer Docs](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/models.html#cache-aware-streaming-conformer)
- [NeMo ConformerEncoder](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/modules/conformer_encoder.py)
- [NeMo MultiHeadAttention](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/submodules/multi_head_attention.py)
- [NeMo CausalConv1D](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/submodules/causal_convs.py)
- [Conformer Paper](https://arxiv.org/abs/2005.08100)
