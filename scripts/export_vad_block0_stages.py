#!/usr/bin/env python3
"""
Run MarbleNet block 0 in PyTorch step-by-step, dumping intermediates after
each module of mconv plus the final ReLU. Used to bisect the C++ port mismatch.

Outputs (default tests/diarize/vad_ref/):
    block0_dw.f32     after mconv[0] (depthwise conv1d)
    block0_pw.f32     after mconv[1] (pointwise conv1d)
    block0_bn.f32     after mconv[2] (BatchNorm1d)
    block0_relu.f32   after mout (ReLU, == enc_block_0.f32)

Each shape is (1, C, T).
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import nemo.collections.asr as nemo_asr


def main():
    out = Path("tests/diarize/vad_ref")
    out.mkdir(parents=True, exist_ok=True)

    print("loading vad_multilingual_marblenet...")
    model = nemo_asr.models.EncDecClassificationModel.from_pretrained(
        model_name="vad_multilingual_marblenet")
    model.eval().to("cpu")

    audio, sr = sf.read("tests/diarize/an4_diarize_test.wav", dtype="float32")
    sig = torch.from_numpy(audio).unsqueeze(0)
    sig_len = torch.tensor([len(audio)], dtype=torch.int64)
    with torch.no_grad():
        mel, mel_len = model.preprocessor(input_signal=sig, length=sig_len)

    block = model.encoder.encoder[0]
    mconv = list(block.mconv)
    print(f"block 0 mconv has {len(mconv)} modules:")
    for i, m in enumerate(mconv):
        print(f"  [{i}] {type(m).__name__}")

    # mconv structure for separable, R=1: [MaskedConv1d (dw), MaskedConv1d (pw), BatchNorm1d]
    assert len(mconv) == 3, f"expected 3 modules in block 0 mconv, got {len(mconv)}"

    # Use the *valid* frame count from the preprocessor (520, not the padded 528).
    # MaskedConv1d zeroes out the input beyond lens before each conv — see
    # mask_input in jasper.py.
    lens = mel_len.clone()
    print(f"mel shape: {tuple(mel.shape)}, lens={int(lens[0])}")

    with torch.no_grad():
        x = mel  # (1, 80, T)

        # mconv[0]: depthwise. MaskedConv1d returns (out, lens).
        m = mconv[0]
        if hasattr(m, "use_mask") or m.__class__.__name__ == "MaskedConv1d":
            x_dw, _ = m(x, lens)
        else:
            x_dw = m(x)
        print(f"after dw: {tuple(x_dw.shape)}")
        x_dw.numpy().astype(np.float32).tofile(out / "block0_dw.f32")

        # mconv[1]: pointwise.
        m = mconv[1]
        if m.__class__.__name__ == "MaskedConv1d":
            x_pw, _ = m(x_dw, lens)
        else:
            x_pw = m(x_dw)
        print(f"after pw: {tuple(x_pw.shape)}")
        x_pw.numpy().astype(np.float32).tofile(out / "block0_pw.f32")

        # mconv[2]: BatchNorm1d.
        x_bn = mconv[2](x_pw)
        print(f"after bn: {tuple(x_bn.shape)}")
        x_bn.numpy().astype(np.float32).tofile(out / "block0_bn.f32")

        # block.mout = Sequential(ReLU, Dropout). Dropout is identity at eval.
        x_relu = block.mout(x_bn)
        print(f"after mout (relu): {tuple(x_relu.shape)}")
        x_relu.numpy().astype(np.float32).tofile(out / "block0_relu.f32")

    # Diff block0_relu against the previously-dumped enc_block_0.f32 to make sure
    # they match (sanity check for our manual stage decomposition).
    enc_ref = np.fromfile(out / "enc_block_0.f32", dtype=np.float32)
    relu_ref = np.fromfile(out / "block0_relu.f32", dtype=np.float32)
    diff = np.abs(enc_ref - relu_ref).max()
    print(f"sanity: |enc_block_0 - block0_relu|_inf = {diff:.2e} (should be 0)")

    shapes = {
        "block0_dw":   list(x_dw.shape),
        "block0_pw":   list(x_pw.shape),
        "block0_bn":   list(x_bn.shape),
        "block0_relu": list(x_relu.shape),
    }
    with open(out / "block0_shapes.json", "w") as f:
        json.dump(shapes, f, indent=2)

    print(f"\nstages saved to {out}/block0_*.f32")


if __name__ == "__main__":
    main()
