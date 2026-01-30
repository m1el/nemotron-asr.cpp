#!/usr/bin/env python3
"""
Convert NeMo ASR model weights to GGUF format with optional quantization.

Usage:
    # Full precision (F32)
    uv run scripts/convert_to_gguf.py model.nemo weights/model.gguf

    # Quantize encoder to Q8_0 (8-bit)
    uv run scripts/convert_to_gguf.py model.nemo weights/model-q8.gguf --quantize q8_0

    # Quantize encoder to Q4_0 (4-bit)
    uv run scripts/convert_to_gguf.py model.nemo weights/model-q4.gguf --quantize q4_0

    # Quantize only specific layers
    uv run scripts/convert_to_gguf.py model.nemo weights/model.gguf --quantize q8_0 --quantize-pattern "feed_forward"
"""

import argparse
import struct
import tarfile
from typing import Tuple
import torch
import yaml
import numpy as np
from pathlib import Path
import re

# GGUF constants
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

# GGUF metadata types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8

# GGML tensor types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9

# Block sizes for quantized types
QK4_0 = 32  # Block size for Q4_0
QK8_0 = 32  # Block size for Q8_0


def write_string(f, s: str | bytes):
    """Write a GGUF string (length + data, no null terminator)."""
    if isinstance(s, str):
        data = s.encode('utf-8')
    else:
        data = s
    f.write(struct.pack('<Q', len(data)))
    f.write(data)


def write_kv_string(f, key: str, value: str):
    """Write a string key-value pair."""
    write_string(f, key)
    f.write(struct.pack('<i', GGUF_TYPE_STRING))
    write_string(f, value)


def write_kv_uint32(f, key: str, value: int):
    """Write a uint32 key-value pair."""
    write_string(f, key)
    f.write(struct.pack('<i', GGUF_TYPE_UINT32))
    f.write(struct.pack('<I', value))


def write_kv_int32(f, key: str, value: int):
    """Write an int32 key-value pair."""
    write_string(f, key)
    f.write(struct.pack('<i', GGUF_TYPE_INT32))
    f.write(struct.pack('<i', value))


def write_kv_float32(f, key: str, value: float):
    """Write a float32 key-value pair."""
    write_string(f, key)
    f.write(struct.pack('<i', GGUF_TYPE_FLOAT32))
    f.write(struct.pack('<f', value))


def quantize_q8_0(data: np.ndarray) -> bytes:
    """
    Quantize float32 array to Q8_0 format (fully vectorized).

    Q8_0 block structure (34 bytes per 32 elements):
    - 1 x float16 scale (2 bytes)
    - 32 x int8 quantized values (32 bytes)
    """
    data = data.astype(np.float32).flatten()
    n_elements = len(data)

    # Pad to multiple of block size
    if n_elements % QK8_0 != 0:
        pad_size = QK8_0 - (n_elements % QK8_0)
        data = np.pad(data, (0, pad_size), mode='constant', constant_values=0)

    n_blocks = len(data) // QK8_0
    blocks = data.reshape(n_blocks, QK8_0)

    # Compute scales for all blocks at once
    amax = np.max(np.abs(blocks), axis=1)
    scales = np.where(amax != 0, amax / 127.0, 0.0).astype(np.float16)

    # Quantize all blocks at once
    scales_expanded = scales[:, np.newaxis].astype(np.float32)
    scales_safe = np.where(scales_expanded != 0, scales_expanded, 1.0)
    quantized = np.round(blocks / scales_safe).astype(np.int8)
    quantized = np.where(scales_expanded != 0, quantized, 0).astype(np.int8)

    # Build output using structured array (no Python loop)
    # Each block: 2 bytes scale + 32 bytes quantized = 34 bytes
    block_dtype = np.dtype([('scale', np.float16), ('quants', np.int8, QK8_0)])
    output_arr = np.empty(n_blocks, dtype=block_dtype)
    output_arr['scale'] = scales
    output_arr['quants'] = quantized

    return output_arr.tobytes()


def quantize_q4_0(data: np.ndarray) -> bytes:
    """
    Quantize float32 array to Q4_0 format (fully vectorized).

    Q4_0 block structure (18 bytes per 32 elements):
    - 1 x float16 scale (2 bytes)
    - 16 bytes of packed 4-bit values (32 values, 2 per byte)

    Values are quantized to [-8, 7] range (signed 4-bit).
    Packing: low nibble = first 16 values, high nibble = last 16 values
    """
    data = data.astype(np.float32).flatten()
    n_elements = len(data)

    # Pad to multiple of block size
    if n_elements % QK4_0 != 0:
        pad_size = QK4_0 - (n_elements % QK4_0)
        data = np.pad(data, (0, pad_size), mode='constant', constant_values=0)

    n_blocks = len(data) // QK4_0
    blocks = data.reshape(n_blocks, QK4_0)

    # Compute scales for all blocks at once
    amax = np.max(np.abs(blocks), axis=1)
    scales = np.where(amax != 0, amax / 7.0, 0.0).astype(np.float16)

    # Quantize all blocks to [-8, 7]
    scales_expanded = scales[:, np.newaxis].astype(np.float32)
    scales_safe = np.where(scales_expanded != 0, scales_expanded, 1.0)
    quantized = np.round(blocks / scales_safe).astype(np.int8)
    quantized = np.clip(quantized, -8, 7)
    quantized = np.where(scales_expanded != 0, quantized, 0)

    # Convert to unsigned [0, 15] and pack
    quantized_u = (quantized + 8).astype(np.uint8)
    # low nibble = first 16 values, high nibble = last 16 values
    low = quantized_u[:, :QK4_0//2] & 0x0F
    high = quantized_u[:, QK4_0//2:] & 0x0F
    packed = (low | (high << 4)).astype(np.uint8)  # [n_blocks, 16]

    # Build output using structured array (no Python loop)
    # Each block: 2 bytes scale + 16 bytes packed = 18 bytes
    block_dtype = np.dtype([('scale', np.float16), ('quants', np.uint8, QK4_0 // 2)])
    output_arr = np.empty(n_blocks, dtype=block_dtype)
    output_arr['scale'] = scales
    output_arr['quants'] = packed

    return output_arr.tobytes()


def get_quantized_size(n_elements: int, quant_type: int) -> int:
    """Calculate the size in bytes for quantized tensor."""
    if quant_type == GGML_TYPE_Q8_0:
        n_blocks = (n_elements + QK8_0 - 1) // QK8_0
        return n_blocks * (2 + QK8_0)  # float16 + 32 int8
    elif quant_type == GGML_TYPE_Q4_0:
        n_blocks = (n_elements + QK4_0 - 1) // QK4_0
        return n_blocks * (2 + QK4_0 // 2)  # float16 + 16 packed bytes
    elif quant_type == GGML_TYPE_F16:
        return n_elements * 2
    else:  # F32
        return n_elements * 4


def get_conv_reshape_type(name: str) -> str | None:
    """
    Determine how to reshape conv weights for direct use without runtime reshape.

    Returns:
        'pointwise' - squeeze kernel dim (out, in, 1) -> (out, in)
        'depthwise' - squeeze groups dim and transpose (out, 1, k) -> (k, out)
        None - no special handling
    """
    if re.search(r"\.conv\.(pointwise_conv1|pointwise_conv2)\.weight$", name):
        return 'pointwise'
    elif re.search(r"\.conv\.depthwise_conv\.weight$", name):
        return 'depthwise'
    return None


def should_skip_quantization(name: str) -> bool:
    """Check if tensor should be excluded from quantization."""
    # Depthwise conv uses manual loop with ggml_view_1d which doesn't work with quantized tensors
    # It's tiny anyway (~31KB per layer F32)
    if re.search(r"\.conv\.depthwise_conv\.weight$", name):
        return True
    return False


def should_quantize(name: str, patterns: list[str], exclude_patterns: list[str]) -> bool:
    """Determine if a tensor should be quantized based on patterns."""
    # Default: quantize encoder layer weights (not biases, not norms)
    # Note: feed_forward1/feed_forward2, not feed_forward
    if not patterns:
        patterns = [r"encoder\.layers\.\d+\.(feed_forward\d+|self_attn|conv)\.[^.]+\.weight$"]

    # Check exclusions first
    for pattern in exclude_patterns:
        if re.search(pattern, name):
            return False

    # Check inclusions
    for pattern in patterns:
        if re.search(pattern, name):
            return True

    return False


def load_nemo_model(path: str) -> Tuple[dict, bytes]:
    with tarfile.open(path) as tar:
        model_config = tar.extractfile("./model_config.yaml")
        model_config = yaml.safe_load(model_config)
        vocab = load_vocab(model_config['joint']['vocabulary'])
        weights = tar.extractfile("./model_weights.ckpt")
        torch_weights = torch.load(weights, weights_only=True, map_location='cpu')
        numpy_weights = {name: tensor.numpy() for name, tensor in torch_weights.items()}
    return numpy_weights, vocab


def load_vocab(vocab: list[str]) -> bytes:
    print(f"Loaded vocab with {len(vocab)} tokens")
    WORD_SIZE_BYTES = 8
    rv = bytearray(len(vocab) * WORD_SIZE_BYTES)
    for i, ch in enumerate(vocab):
        encoded = ch.encode("utf-8") + b"\0"
        assert len(encoded) <= WORD_SIZE_BYTES, f"token too long: {ch}"
        rv[i * WORD_SIZE_BYTES:i * WORD_SIZE_BYTES + len(encoded)] = encoded
    return rv


def convert_to_gguf(
    input_path: str,
    output_path: str,
    quant_type: str = None,
    quant_patterns: list[str] = None,
    exclude_patterns: list[str] = None,
):
    """Convert NEMO weights to GGUF format with optional quantization."""
    tensors, vocab = load_nemo_model(input_path)
    print(f"\nLoaded {len(tensors)} tensors")

    # Parse quantization type
    ggml_quant_type = GGML_TYPE_F32
    quant_name = "F32"
    if quant_type:
        quant_type = quant_type.lower()
        if quant_type in ("q8_0", "q8"):
            ggml_quant_type = GGML_TYPE_Q8_0
            quant_name = "Q8_0"
        elif quant_type in ("q4_0", "q4"):
            ggml_quant_type = GGML_TYPE_Q4_0
            quant_name = "Q4_0"
        elif quant_type in ("f16", "fp16"):
            ggml_quant_type = GGML_TYPE_F16
            quant_name = "F16"
        else:
            print(f"Warning: Unknown quantization type '{quant_type}', using F32")

    if quant_patterns is None:
        quant_patterns = []
    if exclude_patterns is None:
        exclude_patterns = []

    # Model hyperparameters
    hparams = {
        "nemo.n_mels": 128,
        "nemo.d_model": 1024,
        "nemo.n_heads": 8,
        "nemo.d_head": 128,
        "nemo.d_ff": 4096,
        "nemo.n_layers": 24,
        "nemo.kernel_size": 31,
        "nemo.vocab_size": 1025,
        "nemo.decoder_dim": 320,
        "nemo.joint_dim": 640,
    }

    # Prepare tensor info
    tensor_infos = []
    current_offset = 0

    stats = {
        "f32_tensors": 0,
        "f32_bytes": 0,
        "quantized_tensors": 0,
        "quantized_bytes_before": 0,
        "quantized_bytes_after": 0,
    }

    for name, data in tensors.items():
        # Reshape conv weights to 2D for direct use without runtime reshape
        # This enables quantization (ne[0] >= 32) and removes reshape ops from inference
        conv_type = get_conv_reshape_type(name)
        if conv_type == 'pointwise' and len(data.shape) == 3:
            # pointwise: (out_ch, in_ch, 1) -> (out_ch, in_ch)
            assert data.shape[2] == 1, f"Expected kernel=1 for pointwise conv {name}, got {data.shape[2]}"
            data = data.squeeze(axis=2)
            print(f"  -> squeezed pointwise to 2D: {data.shape}")
        elif conv_type == 'depthwise' and len(data.shape) == 3:
            # depthwise: (out_ch, 1, kernel) -> (kernel, out_ch)
            # Squeeze groups dim then transpose to make ne[0]=out_ch>=32 for quantization
            assert data.shape[1] == 1, f"Expected groups=1 for depthwise conv {name}, got {data.shape[1]}"
            data = data.squeeze(axis=1).T  # (out_ch, kernel) -> (kernel, out_ch)
            print(f"  -> squeezed+transposed depthwise to 2D: {data.shape}")

        # GGUF uses row-major with dimensions in reverse order
        shape_gguf = list(reversed(data.shape))
        while len(shape_gguf) < 4:
            shape_gguf.append(1)

        n_elements = int(np.prod(data.shape))

        # Decide whether to quantize this tensor
        do_quantize = (
            ggml_quant_type != GGML_TYPE_F32
            and should_quantize(name, quant_patterns, exclude_patterns)
            and not should_skip_quantization(name)  # Depthwise conv uses view ops, can't quantize
            and n_elements >= 256  # Don't quantize tiny tensors
            and len(data.shape) >= 2  # Only quantize matrices
        )

        print(f"processing tensor: name={name} shape={data.shape} dtype={data.dtype} quantize={do_quantize}")
        if do_quantize:
            tensor_type = ggml_quant_type
            if ggml_quant_type == GGML_TYPE_Q8_0:
                tensor_data = quantize_q8_0(data)
            elif ggml_quant_type == GGML_TYPE_Q4_0:
                tensor_data = quantize_q4_0(data)
            elif ggml_quant_type == GGML_TYPE_F16:
                tensor_data = data.astype(np.float16).tobytes()
            else:
                tensor_data = data.astype(np.float32).tobytes()
                tensor_type = GGML_TYPE_F32

            stats["quantized_tensors"] += 1
            stats["quantized_bytes_before"] += n_elements * 4
            stats["quantized_bytes_after"] += len(tensor_data)
        else:
            tensor_type = GGML_TYPE_F32
            tensor_data = data.astype(np.float32).tobytes()
            stats["f32_tensors"] += 1
            stats["f32_bytes"] += len(tensor_data)

        # Calculate aligned offset
        aligned_offset = (current_offset + GGUF_DEFAULT_ALIGNMENT - 1) // GGUF_DEFAULT_ALIGNMENT * GGUF_DEFAULT_ALIGNMENT

        tensor_infos.append({
            'name': name,
            'shape': shape_gguf[:4],
            'n_dims': len(data.shape),
            'type': tensor_type,
            'offset': aligned_offset,
            'data': tensor_data,
        })

        current_offset = aligned_offset + len(tensor_data)

    # Print quantization stats
    print(f"\nQuantization: {quant_name}")
    print(f"  F32 tensors: {stats['f32_tensors']} ({stats['f32_bytes'] / 1e6:.1f} MB)")
    if stats["quantized_tensors"] > 0:
        ratio = stats["quantized_bytes_before"] / stats["quantized_bytes_after"]
        print(f"  Quantized tensors: {stats['quantized_tensors']}")
        print(f"    Before: {stats['quantized_bytes_before'] / 1e6:.1f} MB")
        print(f"    After:  {stats['quantized_bytes_after'] / 1e6:.1f} MB")
        print(f"    Ratio:  {ratio:.2f}x compression")

    total_before = stats['f32_bytes'] + stats['quantized_bytes_before']
    total_after = stats['f32_bytes'] + stats['quantized_bytes_after']
    print(f"\nTotal size: {total_before / 1e6:.1f} MB -> {total_after / 1e6:.1f} MB ({total_after / total_before * 100:.1f}%)")

    # Write GGUF file
    print(f"\nWriting GGUF to {output_path}...")

    with open(output_path, 'wb') as f:
        # Write header
        f.write(GGUF_MAGIC)
        f.write(struct.pack('<I', GGUF_VERSION))
        f.write(struct.pack('<q', len(tensor_infos)))  # n_tensors
        f.write(struct.pack('<q', len(hparams) + 3))  # n_kv

        # Write KV pairs
        write_kv_string(f, "general.architecture", "nemo")
        write_kv_string(f, "general.name", "nemotron-speech-streaming-en-0.6b")
        write_kv_string(f, "tokenizer.vocab", vocab)

        for key, value in hparams.items():
            if isinstance(value, int):
                write_kv_uint32(f, key, value)
            elif isinstance(value, float):
                write_kv_float32(f, key, value)

        # Write tensor infos
        for info in tensor_infos:
            write_string(f, info['name'])
            f.write(struct.pack('<I', info['n_dims']))
            for dim in info['shape'][:info['n_dims']]:
                f.write(struct.pack('<q', dim))
            f.write(struct.pack('<i', info['type']))
            f.write(struct.pack('<Q', info['offset']))

        # Align to GGUF_DEFAULT_ALIGNMENT before tensor data
        current_pos = f.tell()
        aligned_pos = (current_pos + GGUF_DEFAULT_ALIGNMENT - 1) // GGUF_DEFAULT_ALIGNMENT * GGUF_DEFAULT_ALIGNMENT
        f.write(b'\x00' * (aligned_pos - current_pos))

        data_start = f.tell()

        # Write tensor data
        for info in tensor_infos:
            target_pos = data_start + info['offset']
            current_pos = f.tell()
            if target_pos > current_pos:
                f.write(b'\x00' * (target_pos - current_pos))
            f.write(info['data'])

        file_size = f.tell()

    print(f"Written {file_size / 1024 / 1024:.2f} MB")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert NeMo weights to GGUF format with optional quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full precision
  %(prog)s model.nemo model.gguf

  # 8-bit quantization (encoder weights)
  %(prog)s model.nemo model-q8.gguf --quantize q8_0

  # 4-bit quantization (encoder weights)
  %(prog)s model.nemo model-q4.gguf --quantize q4_0

  # Quantize only feed-forward layers
  %(prog)s model.nemo model.gguf --quantize q8_0 --pattern "feed_forward.*weight"

  # Exclude attention from quantization
  %(prog)s model.nemo model.gguf --quantize q8_0 --exclude "self_attn"

Quantization types:
  q8_0  - 8-bit (3.8x compression, minimal quality loss)
  q4_0  - 4-bit (7.1x compression, some quality loss)
  f16   - 16-bit float (2x compression, no quality loss)
"""
    )
    parser.add_argument("input", help="Input NeMo model file (.nemo)")
    parser.add_argument("output", help="Output GGUF file (.gguf)")
    parser.add_argument(
        "-q", "--quantize",
        choices=["q8_0", "q8", "q4_0", "q4", "f16"],
        help="Quantization type for encoder weights"
    )
    parser.add_argument(
        "-p", "--pattern",
        action="append",
        dest="patterns",
        default=[],
        help="Regex pattern for tensors to quantize (can be repeated)"
    )
    parser.add_argument(
        "-x", "--exclude",
        action="append",
        dest="exclude",
        default=[],
        help="Regex pattern for tensors to exclude from quantization"
    )
    args = parser.parse_args()

    convert_to_gguf(
        args.input,
        args.output,
        quant_type=args.quantize,
        quant_patterns=args.patterns,
        exclude_patterns=args.exclude,
    )


if __name__ == "__main__":
    main()
