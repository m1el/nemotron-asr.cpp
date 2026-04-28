#!/usr/bin/env python3
"""
Run NeMo's MarbleNet on a known audio file and dump per-stage tensors as
binary fixtures for layer-by-layer verification of the C++ port.

Outputs (default tests/diarize/vad_ref/):
    input_audio.f32         raw audio, float32, 16 kHz mono
    mel.f32                 preprocessor output, shape (1, n_mels, T)  -- header below
    enc_block_0.f32 .. enc_block_5.f32   output of each Jasper block, (1, C, T)
    logits.f32              decoder output, shape (1, n_classes)
    shapes.json             JSON map: name -> [shape, dtype]

Each .f32 file is raw float32, no header — shape is recorded in shapes.json.

Usage:
    uv run scripts/export_vad_ref.py [--audio path] [--out-dir path]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import nemo.collections.asr as nemo_asr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="tests/diarize/an4_diarize_test.wav")
    ap.add_argument("--out-dir", default="tests/diarize/vad_ref")
    ap.add_argument("--model", default="vad_multilingual_marblenet")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading model: {args.model}")
    model = nemo_asr.models.EncDecClassificationModel.from_pretrained(model_name=args.model)
    model.eval()
    model = model.to("cpu")

    print(f"loading audio: {args.audio}")
    audio, sr = sf.read(args.audio, dtype="float32")
    assert sr == 16000, f"expected 16 kHz, got {sr}"
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    print(f"  {len(audio)} samples ({len(audio)/sr:.2f} s)")

    # Save input audio for the C++ side.
    audio.tofile(out / "input_audio.f32")

    # Run preprocessor.
    sig = torch.from_numpy(audio).unsqueeze(0)
    sig_len = torch.tensor([len(audio)], dtype=torch.int64)
    with torch.no_grad():
        mel, mel_len = model.preprocessor(input_signal=sig, length=sig_len)
    print(f"  preprocessor out: {tuple(mel.shape)}  len={int(mel_len[0])}")
    mel.numpy().astype(np.float32).tofile(out / "mel.f32")

    shapes = {
        "input_audio": [list(audio.shape), "float32"],
        "mel": [list(mel.shape), "float32"],
    }

    # Hook each Jasper block in the encoder.
    block_outputs = {}
    hooks = []
    blocks = list(model.encoder.encoder)
    print(f"  encoder has {len(blocks)} Jasper blocks")
    for i, blk in enumerate(blocks):
        def make_hook(idx):
            def hook(_mod, _inp, output):
                # JasperBlock.forward returns (xs, lens) where xs is a list with one tensor.
                if isinstance(output, tuple):
                    out = output[0]
                else:
                    out = output
                if isinstance(out, list):
                    out = out[0]
                block_outputs[idx] = out.detach().cpu()
            return hook
        hooks.append(blk.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        enc_out, enc_len = model.encoder(audio_signal=mel, length=mel_len)
        logits = model.decoder(encoder_output=enc_out)

    for h in hooks:
        h.remove()

    for i in sorted(block_outputs.keys()):
        t = block_outputs[i]
        path = out / f"enc_block_{i}.f32"
        t.numpy().astype(np.float32).tofile(path)
        shapes[f"enc_block_{i}"] = [list(t.shape), "float32"]
        print(f"  block {i}: {tuple(t.shape)}")

    print(f"  encoder out: {tuple(enc_out.shape)}  len={int(enc_len[0])}")
    enc_out.numpy().astype(np.float32).tofile(out / "encoder_out.f32")
    shapes["encoder_out"] = [list(enc_out.shape), "float32"]

    print(f"  logits (clip-level, after avg-pool): {tuple(logits.shape)}")
    logits.numpy().astype(np.float32).tofile(out / "logits.f32")
    shapes["logits"] = [list(logits.shape), "float32"]

    # Decoder applies softmax for classification (clip-level).
    probs = torch.softmax(logits, dim=-1) if logits.ndim == 2 else None
    if probs is not None:
        probs.numpy().astype(np.float32).tofile(out / "probs.f32")
        shapes["probs"] = [list(probs.shape), "float32"]
        print(f"  clip probs: {probs.numpy().tolist()}")

    # ---- Per-frame VAD logits (skip the AdaptiveAvgPool, apply Linear per frame).
    # NeMo's diarizer normally runs MarbleNet on 0.63s sliding windows with
    # 0.01s shift — equivalent to taking a 63-frame moving average of the per-frame
    # encoder logits. For our v1 we apply Linear directly per frame; this is
    # behaviorally similar for VAD and ~63x cheaper. See DIARIZATION_PLAN.md.
    dec_w = model.decoder.decoder_layers[0].weight.detach()  # (n_classes, C)
    dec_b = model.decoder.decoder_layers[0].bias.detach()    # (n_classes,)
    enc_out = enc_out.detach()
    # enc_out: (1, C, T)  ->  per-frame logits (1, T, n_classes)
    per_frame_logits = torch.einsum("bct,kc->btk", enc_out, dec_w) + dec_b
    per_frame_probs = torch.softmax(per_frame_logits, dim=-1)
    per_frame_logits.numpy().astype(np.float32).tofile(out / "per_frame_logits.f32")
    per_frame_probs.numpy().astype(np.float32).tofile(out / "per_frame_probs.f32")
    shapes["per_frame_logits"] = [list(per_frame_logits.shape), "float32"]
    shapes["per_frame_probs"] = [list(per_frame_probs.shape), "float32"]
    speech = per_frame_probs[..., 1].numpy()
    print(f"  per-frame probs: {per_frame_probs.shape}, "
          f"speech p_min={speech.min():.3f} p_max={speech.max():.3f} "
          f"p_mean={speech.mean():.3f}")

    # NeMo's actual sliding-window VAD probs (apparent ground truth).
    # window=0.63s (=63 frames @ 10ms), shift=0.01s (=1 frame).
    avg_pool = torch.nn.AvgPool1d(kernel_size=63, stride=1)
    enc_pooled = avg_pool(enc_out)  # (1, C, T-62)
    sw_logits = torch.einsum("bct,kc->btk", enc_pooled, dec_w) + dec_b
    sw_probs = torch.softmax(sw_logits, dim=-1)
    sw_logits.numpy().astype(np.float32).tofile(out / "sw_logits.f32")
    sw_probs.numpy().astype(np.float32).tofile(out / "sw_probs.f32")
    shapes["sw_logits"] = [list(sw_logits.shape), "float32"]
    shapes["sw_probs"] = [list(sw_probs.shape), "float32"]
    sw_speech = sw_probs[..., 1].numpy()
    print(f"  sliding-window probs: {sw_probs.shape}, "
          f"speech p_min={sw_speech.min():.3f} p_max={sw_speech.max():.3f} "
          f"p_mean={sw_speech.mean():.3f}")

    with open(out / "shapes.json", "w") as f:
        json.dump(shapes, f, indent=2)

    print(f"\nwrote fixtures to {out}/")


if __name__ == "__main__":
    main()
