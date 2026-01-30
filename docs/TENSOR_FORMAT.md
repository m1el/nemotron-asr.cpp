# Tensor Format: NeMo vs GGML

This document describes the differences between tensor formats in the original NeMo model and the GGML/GGUF format used by this implementation, along with the reasons for these differences.

## Dimension Ordering

**GGUF reverses tensor dimensions** compared to PyTorch/NeMo:
- PyTorch: `(dim0, dim1, dim2)` (row-major, C-style)
- GGUF: `[dim2, dim1, dim0]` (reversed)

Example: A PyTorch tensor with shape `(1024, 512)` becomes `[512, 1024]` in GGUF, where `ne[0]=512` and `ne[1]=1024`.

## Quantization Requirements

GGML block quantization (Q8_0, Q4_0) requires **`ne[0] >= 32`** because:
- Q8_0: 34-byte blocks (2-byte float16 scale + 32 int8 values)
- Q4_0: 18-byte blocks (2-byte float16 scale + 16 packed bytes for 32 values)

Tensors with `ne[0] < 32` cannot be quantized and must remain F32.

## Conv Weight Reshaping

The Conformer conv module has three weight tensors that required special handling:

### Pointwise Convolutions (conv_pw1_w, conv_pw2_w)

| Format | pointwise_conv1 | pointwise_conv2 |
|--------|-----------------|-----------------|
| PyTorch (original) | `(2048, 1024, 1)` | `(1024, 1024, 1)` |
| GGUF (if stored as-is) | `[1, 1024, 2048]` | `[1, 1024, 1024]` |
| **ne[0]** | **1** ❌ | **1** ❌ |

**Problem**: `ne[0]=1` cannot be quantized (requires ≥32).

**Solution**: Squeeze the redundant kernel dimension (always 1 for 1x1 convolutions):

| Format | pointwise_conv1 | pointwise_conv2 |
|--------|-----------------|-----------------|
| Squeezed | `(2048, 1024)` | `(1024, 1024)` |
| GGUF | `[1024, 2048]` | `[1024, 1024]` |
| **ne[0]** | **1024** ✓ | **1024** ✓ |

**Benefit**: Removes runtime `ggml_reshape_2d` calls and enables quantization.

### Depthwise Convolution (conv_dw_w)

| Format | Shape |
|--------|-------|
| PyTorch (original) | `(1024, 1, 31)` |
| GGUF (if stored as-is) | `[31, 1, 1024]` |
| **ne[0]** | **31** ❌ |

**Problem**: `ne[0]=31` cannot be quantized.

**Solution**: Squeeze the groups dimension (always 1) and transpose:

| Format | Shape |
|--------|-------|
| Squeezed | `(1024, 31)` |
| Transposed | `(31, 1024)` |
| GGUF | `[1024, 31]` |
| **ne[0]** | **1024** ✓ |

**However**: The depthwise conv implementation uses `ggml_view_1d` to manually index kernel elements, which doesn't work with quantized tensors (block structure breaks view alignment). Therefore, **depthwise conv weights are kept as F32** despite having a quantization-compatible shape.

**Benefit**: Removes runtime `ggml_transpose` call. The ~31KB/layer F32 overhead is negligible.

## Summary Table

| Tensor | PyTorch Shape | GGUF Shape | ne[0] | Quantized | Reshape |
|--------|---------------|------------|-------|-----------|---------|
| pointwise_conv1 | `(2048, 1024, 1)` | `[1024, 2048]` | 1024 | ✓ Yes | squeeze(axis=2) |
| pointwise_conv2 | `(1024, 1024, 1)` | `[1024, 1024]` | 1024 | ✓ Yes | squeeze(axis=2) |
| depthwise_conv | `(1024, 1, 31)` | `[1024, 31]` | 1024 | ✗ No (F32) | squeeze(axis=1) + transpose |
| ffn1_linear1 | `(4096, 1024)` | `[1024, 4096]` | 1024 | ✓ Yes | none |
| ffn1_linear2 | `(1024, 4096)` | `[4096, 1024]` | 4096 | ✓ Yes | none |
| attn_q/k/v/out | `(1024, 1024)` | `[1024, 1024]` | 1024 | ✓ Yes | none |

## Inference Code Changes

The C++ inference code was updated to work with the reshaped tensors:

1. **Pointwise conv**: Use weight directly with `ggml_mul_mat` (no reshape needed)
2. **Depthwise conv**: Use weight directly (no transpose needed, stored pre-transposed)
3. **Kernel size inference**: Read from `ne[1]` instead of `ne[0]` for depthwise weight

## Model Compatibility

Models converted with the updated `convert_to_gguf.py` have 2D conv weights. The loader validates this and prints an error if old 3D weights are detected:

```
ERROR: conv weights are 3D (old format). Please reconvert the model with updated convert_to_gguf.py
```
