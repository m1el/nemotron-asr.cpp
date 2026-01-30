# NeMo ASR to ggml Port: Implementation Plan

## Goal
Port NVIDIA's NeMo ASR model (nemotron-speech-streaming-en-0.6b) to ggml framework in C++.

## Current Status
- **Preprocessor**: COMPLETE (`preprocessor.cpp`, 623 lines)
- **Encoder**: NOT STARTED (24 Conformer layers)
- **Decoder**: NOT STARTED (LSTM-based)
- **Joint Network**: NOT STARTED

## Architecture Summary
```
Audio → Preprocessor → Encoder → Joint ← Decoder
         (done)        (24 Conformer)   (LSTM)
                           ↓
                      Greedy Decode → BPE → Text
```

---

## Implementation Phases

### Phase 1: Weight Conversion Infrastructure
**Files to create**:
- `scripts/convert_weights.py` - Extract PyTorch weights to binary format
- `src/ggml_weights.cpp` / `include/ggml_weights.h` - Weight loading

**Tasks**:
1. Load `model_weights.ckpt` (2.4GB PyTorch checkpoint)
2. Save each tensor as `.bin` with header (dtype, dims, data)
3. Create single `weights/model.gguf` file

---

### Phase 2: Core ggml Building Blocks
**Files**: `src/ops.cpp`, `include/ops.h`

**Operations needed**:
- `linear()` - with/without bias
- `layer_norm()` - LayerNorm
- `conv1d()` / `conv2d()` - standard and depthwise
- `causal_conv1d()` / `causal_conv2d()` - left-padded for streaming
- `swish()` - x * sigmoid(x)
- `glu()` - Gated Linear Unit
- `lstm_cell()` - LSTM cell operation

---

### Phase 3: ConvSubsampling (8x subsampling)
**File**: `src/conv_subsampling.cpp`

**Architecture**:
```
Input: [B, T, 128]
  → CausalConv2D(1→256, k=3, s=2) + ReLU
  → DepthwiseConv2D(256, k=3, s=2) + PointwiseConv + ReLU
  → DepthwiseConv2D(256, k=3, s=2) + PointwiseConv + ReLU
  → Reshape + Linear(4352→1024)
Output: [B, T/8, 1024]
```

---

### Phase 4: Conformer Layer Components

#### 4.1 ConformerFeedForward
```cpp
Linear(1024→4096, no_bias) → Swish → Linear(4096→1024, no_bias)
```

#### 4.2 RelPositionalEncoding
- Precomputed position embeddings [1, 2*max_len-1, 1024]
- Slice based on input length for relative positions

#### 4.3 RelPositionMultiHeadAttention (MOST COMPLEX)
- 8 heads, d_k=128
- Q, K, V projections (no bias)
- Relative position bias (pos_bias_u, pos_bias_v)
- `rel_shift()` operation for relative attention
- Chunked local attention (different context sizes per layer)

#### 4.4 ConformerConvolution
```cpp
PointwiseConv(1024→2048) → GLU → DepthwiseConv(k=9, causal)
→ LayerNorm → Swish → PointwiseConv(1024→1024)
```

---

### Phase 5: Full Conformer Encoder
**File**: `src/conformer_encoder.cpp`

**Structure per layer**:
```
x → LayerNorm → FFN1 → residual(×0.5)
  → LayerNorm → MultiHeadAttn → residual
  → LayerNorm → ConvModule → residual
  → LayerNorm → FFN2 → residual(×0.5)
  → LayerNorm → output
```
Repeat 24 times with different weights.

---

### Phase 6: RNNTDecoder
**File**: `src/rnnt_decoder.cpp`

**Architecture**:
- Embedding(1025, 640) - vocab + blank token
- LSTM(640, 640, 2 layers)
- State management for streaming inference

---

### Phase 7: RNNTJoint
**File**: `src/rnnt_joint.cpp`

```cpp
f = Linear(encoder_out, 1024→640) // with bias
g = Linear(decoder_out, 640→640)  // with bias
joint = ReLU(f + g)
logits = Linear(joint, 640→1025)  // with bias
```

---

### Phase 8: Greedy Decoding
**File**: `src/greedy_decode.cpp`

For each encoder time step:
1. Get decoder output for last token
2. Compute joint logits
3. If argmax != blank: emit token, update decoder state
4. If argmax == blank: move to next time step

---

### Phase 9: BPE Tokenizer
**File**: `src/tokenizer.cpp`

- Load vocab from `tokenizer.model` or `vocab.txt`
- Decode token IDs to text (handle `▁` as space)

---

### Phase 10: Integration
**File**: `main.cpp`

```cpp
int main(int argc, char** argv) {
    load_model("weights/model.gguf");

    // Read audio file
    auto audio = load_wav(argv[1]);

    // Preprocess
    auto mel = preprocessor.forward(audio);

    // Encode
    auto enc_out = encoder.forward(mel);

    // Greedy decode
    auto tokens = greedy_decode(encoder, decoder, joint, enc_out);

    // Detokenize
    std::string text = tokenizer.decode(tokens);
    std::cout << text << std::endl;
}
```

---

## File Organization
```
nemotron-speech.cpp/
├── include/
│   ├── ops.h
│   ├── ggml_weights.h
│   ├── conv_subsampling.h
│   ├── conformer_encoder.h
│   ├── multi_head_attention.h
│   ├── rnnt_decoder.h
│   ├── rnnt_joint.h
│   ├── greedy_decode.h
│   └── tokenizer.h
├── src/
│   ├── ops.cpp
│   ├── ggml_weights.cpp
│   ├── conv_subsampling.cpp
│   ├── conformer_encoder.cpp
│   ├── multi_head_attention.cpp
│   ├── rnnt_decoder.cpp
│   ├── rnnt_joint.cpp
│   ├── greedy_decode.cpp
│   └── tokenizer.cpp
├── scripts/
│   └── convert_weights.py
├── weights/
│   └── model.gguf
├── main.cpp
├── preprocessor.cpp (EXISTING)
└── arch.md (EXISTING)
```

---

## Testing Strategy

Each phase includes verification against Python/NeMo:

```python
# test/compare_outputs.py
def test_component(component_name):
    py_out = run_pytorch(input)
    cpp_out = load_cpp_output(f"{component_name}_out.bin")
    assert np.allclose(py_out, cpp_out, atol=1e-5)
```

**Test order**:
1. Weight loading verification
2. Core ops unit tests
3. ConvSubsampling output
4. Single Conformer layer output
5. Full encoder output
6. Decoder output
7. Joint output
8. End-to-end transcription comparison

---

## Key Reference Files

| Component | NeMo Reference |
|-----------|---------------|
| Conformer | `NeMo/.../modules/conformer_encoder.py` |
| Attention | `NeMo/.../submodules/multi_head_attention.py` |
| Conv modules | `NeMo/.../submodules/conformer_modules.py` |
| Subsampling | `NeMo/.../submodules/subsampling.py` |
| Decoder/Joint | `NeMo/.../modules/rnnt.py` |

---

## Verification

After implementation, verify with:
```bash
# Build
cd nemotron-speech.cpp && make

# Run inference
./nemotron-speech test/HFTKzy5xRM-cut.wav

# Compare with Python
cd test && uv run main.py  # Add transcription comparison
```

Expected output should match NeMo model transcription within numerical tolerance.
