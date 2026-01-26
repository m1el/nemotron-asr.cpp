#!/usr/bin/env python3
"""
Convert NeMo ASR model weights to GGUF format.

Usage:
    uv run scripts/convert_to_gguf.py \
        ../nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo \
        weights/model.gguf
"""

import argparse
import struct
import tarfile
from typing import Tuple
import torch
import yaml
import numpy as np
from pathlib import Path

# GGUF constants
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

# GGUF types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8

# GGML types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1


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


def load_nemo_model(path: str) -> Tuple[dict, bytes]:
    with tarfile.open(path) as tar:
        model_config = tar.extractfile("./model_config.yaml")
        model_config = yaml.safe_load(model_config)
        vocab = load_vocab(model_config['joint']['vocabulary'])
        weights = tar.extractfile("./model_weights.ckpt")
        torch_weights = torch.load(weights, weights_only=True, map_location='cpu')
        numpy_weights = { name: tensor.numpy() for name, tensor in torch_weights.items() }
    return numpy_weights, vocab


def load_vocab(vocab: list[str]) -> bytes:
    print(f"Loaded vocab with {len(vocab)} tokens")
    WORD_SIZE_BYTES = 8
    rv = bytearray(len(vocab) * WORD_SIZE_BYTES)
    for i, ch in enumerate(vocab):
        # null-terminate and pad to WORD_SIZE_BYTES
        encoded = ch.encode("utf-8") + b"\0"
        assert len(encoded) <= WORD_SIZE_BYTES, f"token too long: {ch}"
        rv[i * WORD_SIZE_BYTES:i * WORD_SIZE_BYTES + len(encoded)] = encoded
    return rv


def convert_to_gguf(input_path: str, output_path: str):
    """Convert NEMO weights to GGUF format."""
    tensors, vocab = load_nemo_model(input_path)
    print(f"\nLoaded {len(tensors)} tensors")

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

    for name, data in tensors.items():
        # GGUF uses row-major with dimensions in reverse order
        # For a [A, B, C] tensor, GGUF stores it as ne = [C, B, A]
        shape_gguf = list(reversed(data.shape))

        # Pad dimensions to 4
        while len(shape_gguf) < 4:
            shape_gguf.append(1)

        # Calculate size with alignment
        data_size = data.nbytes
        aligned_offset = (current_offset + GGUF_DEFAULT_ALIGNMENT - 1) // GGUF_DEFAULT_ALIGNMENT * GGUF_DEFAULT_ALIGNMENT

        tensor_infos.append({
            'name': name,
            'shape': shape_gguf[:4],  # Only first 4 dims
            'n_dims': len(data.shape),
            'type': GGML_TYPE_F32,
            'offset': aligned_offset,
            'data': data.astype(np.float32).tobytes(),
        })

        current_offset = aligned_offset + data_size

    # Write GGUF file
    print(f"\nWriting GGUF to {output_path}...")

    with open(output_path, 'wb') as f:
        # Write header
        f.write(GGUF_MAGIC)
        f.write(struct.pack('<I', GGUF_VERSION))
        f.write(struct.pack('<q', len(tensor_infos)))  # n_tensors
        f.write(struct.pack('<q', len(hparams) + 3))  # n_kv (hparams + general.architecture + general.name)

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
            # Seek to aligned offset
            target_pos = data_start + info['offset']
            current_pos = f.tell()
            if target_pos > current_pos:
                f.write(b'\x00' * (target_pos - current_pos))

            f.write(info['data'])

        file_size = f.tell()

    print(f"Written {file_size / 1024 / 1024:.2f} MB")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Convert NeMo weights to GGUF format")
    parser.add_argument("input", help="Input weights file (model.bin)")
    parser.add_argument("output", help="Output GGUF file (model.gguf)")
    args = parser.parse_args()

    convert_to_gguf(args.input, args.output)


if __name__ == "__main__":
    main()
