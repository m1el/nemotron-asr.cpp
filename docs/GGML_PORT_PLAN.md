# GGML Porting Plan for NeMo ASR

## Overview

Port the working C++ NeMo ASR implementation to use ggml as the compute backend. This will enable:
- GPU acceleration (CUDA, Metal, Vulkan)
- Quantization support (INT8, INT4)
- Optimized SIMD operations
- Better memory management

## Porting Strategy

**Approach**: Layer-by-layer porting with verification at each step
- Keep the current implementation as reference
- Create new ggml-based files alongside existing ones
- Verify each component produces identical output before moving to next
- Use the same test inputs/outputs from debugging phase

## Directory Structure

```
nemotron-speech.cpp/
├── ggml/                    # ggml library (already cloned)
├── src/                     # Current implementation (keep as reference)
├── src-ggml/                # New ggml-based implementation
│   ├── nemo-ggml.h          # Main header with model structures
│   ├── nemo-ggml.cpp        # Model loading and graph building
│   ├── nemo-ggml-encoder.cpp # Encoder graph
│   └── nemo-ggml-decoder.cpp # Decoder graph
├── tests-ggml/              # Verification tests
└── weights/
    └── model.gguf           # Weights in GGUF format (new)
```

---

## Phase 1: Infrastructure Setup

### 1.1 Build System Integration
- [ ] Add ggml as a dependency in Makefile
- [ ] Create compilation flags for different backends (CPU, CUDA, Metal)
- [ ] Test basic ggml compilation

### 1.2 Weight Conversion to GGUF
- [ ] Create `scripts/convert_to_gguf.py`
- [ ] Convert current `model.bin` to GGUF format with proper tensor names
- [ ] Verify all 653 tensors are correctly converted

### 1.3 Basic Model Structure
- [ ] Define `nemo_model` struct with ggml tensors
- [ ] Define `nemo_state` struct for inference state
- [ ] Implement weight loading from GGUF

**Verification**: Load weights and compare tensor values with current implementation

---

## Phase 2: Basic Operations

### 2.1 Layer Normalization
Current: `layer_norm()` in `src/ops.cpp`
GGML: `ggml_norm()` + `ggml_mul()` + `ggml_add()`

```cpp
// Current
layer_norm(input, weight, bias, dim, eps, output);

// GGML
cur = ggml_norm(ctx, input, eps);
cur = ggml_mul(ctx, cur, weight);
cur = ggml_add(ctx, cur, bias);
```

- [ ] Implement `build_layer_norm()` helper
- [ ] Test with known input/output pair

### 2.2 Linear Projection (no bias)
Current: `linear_no_bias()` in `src/ops.cpp`
GGML: `ggml_mul_mat()`

```cpp
// Current: output = input @ weight.T
linear_no_bias(input, weight, out_features, in_features, output);

// GGML: ggml_mul_mat expects weight in [out, in] layout
cur = ggml_mul_mat(ctx, weight, input);
```

- [ ] Verify weight layout matches (may need transpose)
- [ ] Test linear projection

### 2.3 Activation Functions
- [ ] Swish: `ggml_silu()` (SiLU = Swish)
- [ ] ReLU: `ggml_relu()`
- [ ] GLU: Manual split + sigmoid + multiply

### 2.4 Convolution Operations
Current: `conv1d()`, `causal_conv1d()` in `src/ops.cpp`
GGML: `ggml_conv_1d()`, `ggml_conv_1d_ph()`

- [ ] Map conv parameters (stride, padding, dilation)
- [ ] Handle causal padding (may need manual padding)

**Verification**: Test each op individually with saved debug tensors

---

## Phase 3: ConvSubsampling

### 3.1 Port ConvSubsampling Module
Current: `src/conv_subsampling.cpp`
- Conv2D(1, 256, kernel=3, stride=2) + ReLU
- Conv2D(256, 256, kernel=3, stride=2) + ReLU
- Conv2D(256, 256, kernel=3, stride=2) + ReLU
- Linear(256*16, 1024)

```cpp
static struct ggml_tensor * build_conv_subsampling(
    struct ggml_context * ctx,
    struct ggml_tensor * mel,      // [batch, time, 128]
    struct nemo_model & model
) {
    // Reshape mel for conv2d: [batch, 1, time, 128]
    struct ggml_tensor * cur = ggml_reshape_4d(ctx, mel, 128, mel->ne[1], 1, mel->ne[2]);

    // Conv1: [batch, 1, T, 128] -> [batch, 256, T/2, 64]
    cur = ggml_conv_2d(ctx, model.conv1_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx, cur, model.conv1_b);
    cur = ggml_relu(ctx, cur);

    // Conv2, Conv3 similar...

    // Flatten and project
    cur = ggml_reshape_2d(ctx, cur, 256*16, T/8);
    cur = ggml_mul_mat(ctx, model.subsampling_proj_w, cur);

    return cur;  // [batch, T/8, 1024]
}
```

- [ ] Implement conv2d with correct padding for causal
- [ ] Handle dimension reshaping
- [ ] Test against `cpp_subsampling_debug.bin`

**Verification**: Compare output with saved `cpp_subsampling_debug.bin`

---

## Phase 4: Positional Encoding

### 4.1 Relative Positional Encoding
Current: `RelPositionalEncoding` in `src/conformer_modules.cpp`
- Precomputes sinusoidal embeddings for positions [-(T-1), T-1]
- Returns [2T-1, d_model] tensor

```cpp
static struct ggml_tensor * build_pos_emb(
    struct ggml_context * ctx,
    int seq_len,
    struct nemo_model & model
) {
    // Option 1: Precompute and store as model tensor
    // Option 2: Compute on-the-fly with ggml_sin/ggml_cos

    // For now, precompute for max_len and slice
    int pos_len = 2 * seq_len - 1;
    return ggml_view_2d(ctx, model.pos_emb_cache,
                        model.d_model, pos_len,
                        model.pos_emb_cache->nb[1],
                        (MAX_LEN - seq_len) * model.pos_emb_cache->nb[1]);
}
```

- [ ] Decide: precompute vs compute on-the-fly
- [ ] Implement position embedding extraction

**Verification**: Compare with `nemo_pos_emb_501.bin`

---

## Phase 5: Conformer Feed-Forward Module

### 5.1 FFN Module
Current: `ConformerFeedForward` in `src/conformer_modules.cpp`
- Linear(1024, 4096) -> Swish -> Linear(4096, 1024)

```cpp
static struct ggml_tensor * build_ffn(
    struct ggml_context * ctx,
    struct ggml_tensor * input,
    struct nemo_layer & layer
) {
    struct ggml_tensor * cur = input;

    // Linear1: [batch, time, 1024] -> [batch, time, 4096]
    cur = ggml_mul_mat(ctx, layer.ffn_linear1_w, cur);

    // Swish activation
    cur = ggml_silu(ctx, cur);

    // Linear2: [batch, time, 4096] -> [batch, time, 1024]
    cur = ggml_mul_mat(ctx, layer.ffn_linear2_w, cur);

    return cur;
}
```

- [ ] Implement FFN module
- [ ] Test with layer 0 weights

**Verification**: Compare FFN output with NeMo trace values

---

## Phase 6: Relative Position Multi-Head Attention (CRITICAL)

### 6.1 Self-Attention with Relative Position
Current: `RelPositionMultiHeadAttention` in `src/conformer_modules.cpp`
This is the most complex component.

```cpp
static struct ggml_tensor * build_rel_pos_mha(
    struct ggml_context * ctx,
    struct ggml_tensor * input,      // [batch, time, 1024]
    struct ggml_tensor * pos_emb,    // [2*time-1, 1024]
    struct nemo_layer & layer
) {
    int batch = input->ne[2];
    int time = input->ne[1];
    int d_model = input->ne[0];
    int n_head = 8;
    int d_head = d_model / n_head;  // 128

    // Q, K, V projections
    struct ggml_tensor * Q = ggml_mul_mat(ctx, layer.attn_q_w, input);
    struct ggml_tensor * K = ggml_mul_mat(ctx, layer.attn_k_w, input);
    struct ggml_tensor * V = ggml_mul_mat(ctx, layer.attn_v_w, input);

    // Position projection
    struct ggml_tensor * P = ggml_mul_mat(ctx, layer.attn_pos_w, pos_emb);

    // Reshape to [batch, n_head, time, d_head]
    Q = ggml_reshape_4d(ctx, Q, d_head, n_head, time, batch);
    Q = ggml_permute(ctx, Q, 0, 2, 1, 3);  // [d_head, time, n_head, batch]
    // ... similar for K, V

    // Add positional biases
    // q_u = Q + pos_bias_u
    // q_v = Q + pos_bias_v
    struct ggml_tensor * q_u = ggml_add(ctx, Q, layer.pos_bias_u);
    struct ggml_tensor * q_v = ggml_add(ctx, Q, layer.pos_bias_v);

    // Content attention: q_u @ K^T
    struct ggml_tensor * content_attn = ggml_mul_mat(ctx, K, q_u);

    // Position attention: q_v @ P^T then rel_shift
    struct ggml_tensor * pos_attn = ggml_mul_mat(ctx, P, q_v);
    pos_attn = build_rel_shift(ctx, pos_attn, time);

    // Combine and scale
    struct ggml_tensor * scores = ggml_add(ctx, content_attn, pos_attn);
    scores = ggml_scale(ctx, scores, 1.0f / sqrtf(d_head));

    // Softmax
    scores = ggml_soft_max(ctx, scores);

    // Apply to V
    struct ggml_tensor * out = ggml_mul_mat(ctx, V, scores);

    // Reshape back and output projection
    out = ggml_reshape_3d(ctx, out, d_model, time, batch);
    out = ggml_mul_mat(ctx, layer.attn_out_w, out);

    return out;
}
```

### 6.2 rel_shift Implementation
This is the tricky part - need to implement the pad-reshape-drop algorithm:

```cpp
static struct ggml_tensor * build_rel_shift(
    struct ggml_context * ctx,
    struct ggml_tensor * x,  // [batch, n_head, time, 2*time-1]
    int qlen
) {
    // Method 1: Use ggml_pad, ggml_reshape, ggml_view
    // Method 2: Custom operation (may need to add to ggml)
    // Method 3: Compute indices directly with ggml_get_rows

    // For now, use the direct index formula: out[i,j] = x[i, j + qlen - 1 - i]
    // This may need a custom ggml op or creative use of existing ops
}
```

- [ ] Research best way to implement rel_shift in ggml
- [ ] May need custom operation or creative tensor indexing
- [ ] Test extensively - this was the bug in original implementation

**Verification**: Compare attention output with NeMo trace (critical!)

---

## Phase 7: Conformer Convolution Module

### 7.1 ConvModule
Current: `ConformerConvolution` in `src/conformer_modules.cpp`
- PointwiseConv1D(1024, 2048) -> GLU -> DepthwiseConv1D -> LayerNorm -> Swish -> PointwiseConv1D(1024, 1024)

```cpp
static struct ggml_tensor * build_conv_module(
    struct ggml_context * ctx,
    struct ggml_tensor * input,      // [batch, time, 1024]
    struct nemo_layer & layer
) {
    // Transpose to [batch, 1024, time] for conv1d
    struct ggml_tensor * cur = ggml_permute(ctx, input, 1, 0, 2, 3);

    // Pointwise conv1: [batch, 1024, time] -> [batch, 2048, time]
    cur = ggml_conv_1d(ctx, layer.conv_pw1_w, cur, 1, 0, 1);

    // GLU: split and gate
    int half = cur->ne[0] / 2;
    struct ggml_tensor * a = ggml_view_3d(ctx, cur, half, cur->ne[1], cur->ne[2],
                                          cur->nb[1], cur->nb[2], 0);
    struct ggml_tensor * b = ggml_view_3d(ctx, cur, half, cur->ne[1], cur->ne[2],
                                          cur->nb[1], cur->nb[2], half * sizeof(float));
    cur = ggml_mul(ctx, a, ggml_sigmoid(ctx, b));

    // Depthwise causal conv (kernel=31)
    // Need to handle causal padding manually
    cur = build_causal_depthwise_conv(ctx, cur, layer.conv_dw_w, 31);

    // LayerNorm (over channel dimension)
    cur = ggml_norm(ctx, cur, 1e-5f);
    cur = ggml_mul(ctx, cur, layer.conv_ln_w);
    cur = ggml_add(ctx, cur, layer.conv_ln_b);

    // Swish
    cur = ggml_silu(ctx, cur);

    // Pointwise conv2
    cur = ggml_conv_1d(ctx, layer.conv_pw2_w, cur, 1, 0, 1);

    // Transpose back to [batch, time, 1024]
    cur = ggml_permute(ctx, cur, 1, 0, 2, 3);

    return cur;
}
```

- [ ] Implement GLU activation
- [ ] Handle causal depthwise convolution (padding on left)
- [ ] Test convolution module

**Verification**: Compare conv module output

---

## Phase 8: Full Conformer Layer

### 8.1 Assemble Conformer Layer
```cpp
static struct ggml_tensor * build_conformer_layer(
    struct ggml_context * ctx,
    struct ggml_tensor * input,
    struct ggml_tensor * pos_emb,
    struct nemo_layer & layer
) {
    struct ggml_tensor * residual;
    struct ggml_tensor * cur = input;

    // FFN1 (half residual)
    residual = cur;
    cur = build_layer_norm(ctx, cur, layer.norm_ff1_w, layer.norm_ff1_b);
    cur = build_ffn(ctx, cur, layer.ffn1);
    cur = ggml_add(ctx, residual, ggml_scale(ctx, cur, 0.5f));

    // Self-attention
    residual = cur;
    cur = build_layer_norm(ctx, cur, layer.norm_attn_w, layer.norm_attn_b);
    cur = build_rel_pos_mha(ctx, cur, pos_emb, layer);
    cur = ggml_add(ctx, residual, cur);

    // Conv module
    residual = cur;
    cur = build_layer_norm(ctx, cur, layer.norm_conv_w, layer.norm_conv_b);
    cur = build_conv_module(ctx, cur, layer);
    cur = ggml_add(ctx, residual, cur);

    // FFN2 (half residual)
    residual = cur;
    cur = build_layer_norm(ctx, cur, layer.norm_ff2_w, layer.norm_ff2_b);
    cur = build_ffn(ctx, cur, layer.ffn2);
    cur = ggml_add(ctx, residual, ggml_scale(ctx, cur, 0.5f));

    // Final layer norm
    cur = build_layer_norm(ctx, cur, layer.norm_final_w, layer.norm_final_b);

    return cur;
}
```

- [ ] Assemble full conformer layer
- [ ] Test layer 0 output against NeMo reference

**Verification**: Layer 0 output should match [-0.0436, -0.4306, -1.3677, -1.88, 1.5478]

---

## Phase 9: Full Encoder

### 9.1 Conformer Encoder
```cpp
static struct ggml_tensor * build_encoder(
    struct ggml_context * ctx,
    struct ggml_tensor * mel,
    struct nemo_model & model
) {
    // ConvSubsampling
    struct ggml_tensor * cur = build_conv_subsampling(ctx, mel, model);

    // Get positional embeddings
    int seq_len = cur->ne[1];
    struct ggml_tensor * pos_emb = build_pos_emb(ctx, seq_len, model);

    // 24 Conformer layers
    for (int i = 0; i < 24; i++) {
        cur = build_conformer_layer(ctx, cur, pos_emb, model.layers[i]);
    }

    // Final layer norm
    cur = build_layer_norm(ctx, cur, model.encoder_ln_w, model.encoder_ln_b);

    return cur;
}
```

- [ ] Implement full encoder
- [ ] Test with mel input

**Verification**: Encoder output should match `nemo_encoder_correct.bin`

---

## Phase 10: RNN-T Decoder

### 10.1 LSTM Cell
Current: `lstm_cell()` in `src/ops.cpp`
GGML doesn't have built-in LSTM, need to implement manually.

```cpp
static void build_lstm_cell(
    struct ggml_context * ctx,
    struct ggml_tensor * input,
    struct ggml_tensor * h_prev,
    struct ggml_tensor * c_prev,
    struct nemo_decoder & decoder,
    struct ggml_tensor ** h_out,
    struct ggml_tensor ** c_out
) {
    // gates = input @ W_ih + h_prev @ W_hh + bias
    struct ggml_tensor * gates_i = ggml_mul_mat(ctx, decoder.lstm_w_ih, input);
    struct ggml_tensor * gates_h = ggml_mul_mat(ctx, decoder.lstm_w_hh, h_prev);
    struct ggml_tensor * gates = ggml_add(ctx, gates_i, gates_h);
    gates = ggml_add(ctx, gates, decoder.lstm_b_ih);
    gates = ggml_add(ctx, gates, decoder.lstm_b_hh);

    // Split gates: [i, f, g, o] each of size hidden_dim
    int h = decoder.hidden_dim;
    struct ggml_tensor * i_gate = ggml_sigmoid(ctx, ggml_view_1d(ctx, gates, h, 0));
    struct ggml_tensor * f_gate = ggml_sigmoid(ctx, ggml_view_1d(ctx, gates, h, h*sizeof(float)));
    struct ggml_tensor * g_gate = ggml_tanh(ctx, ggml_view_1d(ctx, gates, h, 2*h*sizeof(float)));
    struct ggml_tensor * o_gate = ggml_sigmoid(ctx, ggml_view_1d(ctx, gates, h, 3*h*sizeof(float)));

    // c = f * c_prev + i * g
    *c_out = ggml_add(ctx, ggml_mul(ctx, f_gate, c_prev), ggml_mul(ctx, i_gate, g_gate));

    // h = o * tanh(c)
    *h_out = ggml_mul(ctx, o_gate, ggml_tanh(ctx, *c_out));
}
```

### 10.2 Decoder Forward
```cpp
static struct ggml_tensor * build_decoder_step(
    struct ggml_context * ctx,
    int token,
    struct ggml_tensor * h_prev,
    struct ggml_tensor * c_prev,
    struct nemo_decoder & decoder,
    struct ggml_tensor ** h_out,
    struct ggml_tensor ** c_out
) {
    // Embedding lookup
    struct ggml_tensor * emb = ggml_get_rows(ctx, decoder.embedding,
                                              ggml_new_i32(ctx, token));

    // LSTM
    build_lstm_cell(ctx, emb, h_prev, c_prev, decoder, h_out, c_out);

    // Output projection
    struct ggml_tensor * out = ggml_mul_mat(ctx, decoder.proj_w, *h_out);

    return out;
}
```

- [ ] Implement LSTM cell with ggml ops
- [ ] Implement decoder step
- [ ] Handle state persistence across steps

**Verification**: Compare decoder output with saved debug values

---

## Phase 11: Joint Network

### 11.1 Joint Forward
```cpp
static struct ggml_tensor * build_joint(
    struct ggml_context * ctx,
    struct ggml_tensor * encoder_out,  // [1, 1, 640]
    struct ggml_tensor * decoder_out,  // [1, 640]
    struct nemo_joint & joint
) {
    // Project encoder: [1, 1, 640] -> [1, 1, 640]
    struct ggml_tensor * enc = ggml_mul_mat(ctx, joint.enc_w, encoder_out);

    // Project decoder: [1, 640] -> [1, 640]
    struct ggml_tensor * dec = ggml_mul_mat(ctx, joint.dec_w, decoder_out);

    // Add and activate
    struct ggml_tensor * joint_out = ggml_add(ctx, enc, dec);
    joint_out = ggml_relu(ctx, joint_out);

    // Final projection to vocab
    joint_out = ggml_mul_mat(ctx, joint.out_w, joint_out);
    joint_out = ggml_add(ctx, joint_out, joint.out_b);

    return joint_out;  // [1, vocab_size]
}
```

- [ ] Implement joint network
- [ ] Test joint output

**Verification**: Compare joint logits

---

## Phase 12: Greedy Decoding

### 12.1 Decoding Loop
The decoding loop needs to handle dynamic graph building per step.

```cpp
std::vector<int> greedy_decode(
    struct nemo_context * ctx,
    struct ggml_tensor * encoder_out
) {
    std::vector<int> tokens;
    int blank_id = ctx->model.vocab_size - 1;

    // Initialize decoder state
    struct ggml_tensor * h = ggml_new_tensor_1d(ctx->ctx, GGML_TYPE_F32, 320);
    struct ggml_tensor * c = ggml_new_tensor_1d(ctx->ctx, GGML_TYPE_F32, 320);
    ggml_set_zero(h);
    ggml_set_zero(c);

    int prev_token = blank_id;
    int T = encoder_out->ne[1];

    for (int t = 0; t < T; t++) {
        // Get encoder frame
        struct ggml_tensor * enc_frame = ggml_view_2d(...);

        while (true) {
            // Build decoder step
            struct ggml_tensor * dec_out = build_decoder_step(...);

            // Build joint
            struct ggml_tensor * logits = build_joint(...);

            // Compute graph
            ggml_graph_compute(...);

            // Get argmax
            int token = argmax(logits);

            if (token == blank_id) break;

            tokens.push_back(token);
            prev_token = token;
        }
    }

    return tokens;
}
```

- [ ] Implement decoding loop with dynamic graph
- [ ] Handle state management between steps

**Verification**: Full pipeline should produce correct transcription

---

## Phase 13: Integration & Optimization

### 13.1 Main Integration
- [ ] Create unified `nemo_context` structure
- [ ] Implement `nemo_init()`, `nemo_free()`, `nemo_transcribe()`
- [ ] Add multi-threading support

### 13.2 Backend Support
- [ ] Test CPU backend
- [ ] Add CUDA backend support
- [ ] Add Metal backend support (macOS)

### 13.3 Performance Optimization
- [ ] Profile and identify bottlenecks
- [ ] Consider quantization (INT8/INT4)
- [ ] Optimize memory allocation patterns

---

## Verification Checkpoints

| Phase | Component | Test Data | Expected |
|-------|-----------|-----------|----------|
| 3 | ConvSubsampling | `test.mel.bin` | Match `cpp_subsampling_debug.bin` |
| 4 | PosEncoding | T=251 | Match `nemo_pos_emb_501.bin` |
| 5-7 | Layer 0 | subsample output | [-0.0436, -0.4306, -1.3677, -1.88, 1.5478] |
| 9 | Full Encoder | `test.mel.bin` | Match `nemo_encoder_correct.bin` |
| 12 | Full Pipeline | `test.mel.bin` | "So you might have heard..." |

---

## Estimated Effort

| Phase | Description | Complexity |
|-------|-------------|------------|
| 1 | Infrastructure | Low |
| 2 | Basic Ops | Low |
| 3 | ConvSubsampling | Medium |
| 4 | Positional Encoding | Low |
| 5 | FFN Module | Low |
| 6 | Self-Attention | **High** (rel_shift) |
| 7 | Conv Module | Medium |
| 8 | Conformer Layer | Medium |
| 9 | Full Encoder | Low (assembly) |
| 10 | LSTM Decoder | Medium |
| 11 | Joint Network | Low |
| 12 | Greedy Decode | Medium |
| 13 | Integration | Medium |

**Critical Path**: Phase 6 (Self-Attention with rel_shift) is the most complex and risk-prone component.

---

## Notes

1. **rel_shift**: May need a custom ggml operation or creative tensor manipulation
2. **LSTM**: Not native to ggml, manual implementation required
3. **Causal Conv**: Need explicit left-padding for causal behavior
4. **Memory**: Pre-allocate max-length tensors to avoid dynamic allocation
5. **Graph Building**: Consider pre-building encoder graph, dynamic decoder graph
