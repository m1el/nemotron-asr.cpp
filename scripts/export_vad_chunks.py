#!/usr/bin/env python3
"""
Dump NeMo's per-chunk VAD inference: for each 0.63 s window with 0.01 s shift,
run MarbleNet on that window in isolation (the way the diarizer's _run_vad
calls test_dataloader → MarbleNet for each sliding sample) and record:
    chunk_logits.f32   (N_chunks, 2)
    chunk_probs.f32    (N_chunks, 2)
    chunk_speech.f32   (N_chunks,)  -- probs[:, 1]
plus a small JSON header describing window/shift.

This is the ground truth for streaming VAD inference; the previously-saved
sw_probs.f32 was an approximation computed via AvgPool over the full-clip
encoder output, which is similar but not identical to per-chunk inference.
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
    print(f"  {len(audio)} samples ({len(audio)/sr:.2f} s)")

    SR = 16000
    WIN_SEC = 0.63
    SHIFT_SEC = 0.01
    win_samples = int(WIN_SEC * SR)        # 10080
    shift_samples = int(SHIFT_SEC * SR)    # 160

    # Number of full windows we can fit:
    #   the i-th window is audio[i*shift : i*shift + win_samples]
    n_chunks = 1 + (len(audio) - win_samples) // shift_samples
    print(f"  chunks: {n_chunks} (window={win_samples} samples, shift={shift_samples})")

    # Batch them. For correctness this MUST match the dataloader's protocol:
    # each chunk is its own (B=1) forward pass.
    BATCH = 64
    all_logits = []
    with torch.no_grad():
        for start in range(0, n_chunks, BATCH):
            end = min(start + BATCH, n_chunks)
            batch = np.stack([audio[i*shift_samples : i*shift_samples + win_samples]
                              for i in range(start, end)])
            sig = torch.from_numpy(batch).float()              # (B, 10080)
            sig_len = torch.full((sig.shape[0],), win_samples, dtype=torch.int64)
            logits = model(input_signal=sig, input_signal_length=sig_len)
            all_logits.append(logits.detach().numpy().astype(np.float32))
            if start == 0:
                print(f"    first batch logits shape: {logits.shape}")

    chunk_logits = np.concatenate(all_logits, axis=0)
    chunk_probs = torch.softmax(torch.from_numpy(chunk_logits), dim=-1).numpy()
    chunk_speech = chunk_probs[:, 1]

    chunk_logits.tofile(out / "chunk_logits.f32")
    chunk_probs.tofile(out / "chunk_probs.f32")
    chunk_speech.tofile(out / "chunk_speech.f32")

    print(f"  chunk_logits: {chunk_logits.shape}  range [{chunk_logits.min():.3f}, {chunk_logits.max():.3f}]")
    print(f"  chunk_probs:  {chunk_probs.shape}")
    print(f"  speech min={chunk_speech.min():.4f}  mean={chunk_speech.mean():.4f}  max={chunk_speech.max():.4f}")

    meta = {
        "n_chunks": int(n_chunks),
        "window_samples": win_samples,
        "shift_samples": shift_samples,
        "sample_rate": SR,
        "audio_samples": int(len(audio)),
        "shapes": {
            "chunk_logits": list(chunk_logits.shape),
            "chunk_probs":  list(chunk_probs.shape),
            "chunk_speech": list(chunk_speech.shape),
        },
    }
    with open(out / "chunk_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote chunk fixtures to {out}/chunk_*")


if __name__ == "__main__":
    main()
