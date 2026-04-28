#!/usr/bin/env python3
"""
Run NeMo's TitaNet-L on a 1.5 s slice of the tutorial audio and dump per-stage
tensors as binary fixtures so the C++ port can verify each layer.

Outputs (default tests/diarize/spk_ref/):
    input_audio.f32         24000 audio samples (1.5 s) used for the test.
    mel.f32                 (1, 80, T) preprocessor output (per_feature norm).
    enc_block_{0..4}.f32    output of each Jasper block (with SE).
    encoder_out.f32         encoder final, (1, 3072, T).
    pool_out.f32            attentive pooling output (1, 6144, 1).
    embedding.f32           final 192-d speaker embedding.
    shapes.json             tensor shapes by name.
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import nemo.collections.asr as nemo_asr


def main():
    out = Path("tests/diarize/spk_ref")
    out.mkdir(parents=True, exist_ok=True)

    print("loading titanet_large...")
    model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large")
    model.eval().to("cpu")

    audio, sr = sf.read("tests/diarize/an4_diarize_test.wav", dtype="float32")
    assert sr == 16000

    # Take a 1.5 s sub-segment from the middle of the audio.
    start = sr * 1  # 1.0 s
    end = start + int(1.5 * sr)
    seg = audio[start:end]
    print(f"  segment: {len(seg)} samples ({len(seg)/sr:.2f} s)")
    seg.astype(np.float32).tofile(out / "input_audio.f32")

    sig = torch.from_numpy(seg).unsqueeze(0)
    sig_len = torch.tensor([len(seg)], dtype=torch.int64)

    with torch.no_grad():
        mel, mel_len = model.preprocessor(input_signal=sig, length=sig_len)
    print(f"  preprocessor out: {tuple(mel.shape)}  len={int(mel_len[0])}")
    mel.numpy().astype(np.float32).tofile(out / "mel.f32")

    shapes = {
        "input_audio": [list(seg.shape), "float32"],
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
                if isinstance(output, tuple):
                    output = output[0]
                if isinstance(output, list):
                    output = output[0]
                block_outputs[idx] = output.detach().cpu()
            return hook
        hooks.append(blk.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        enc_out, enc_len = model.encoder(audio_signal=mel, length=mel_len)
    for h in hooks:
        h.remove()

    for i in sorted(block_outputs.keys()):
        t = block_outputs[i]
        t.numpy().astype(np.float32).tofile(out / f"enc_block_{i}.f32")
        shapes[f"enc_block_{i}"] = [list(t.shape), "float32"]
        print(f"  block {i}: {tuple(t.shape)}")

    print(f"  encoder out: {tuple(enc_out.shape)}  len={int(enc_len[0])}")
    enc_out.detach().numpy().astype(np.float32).tofile(out / "encoder_out.f32")
    shapes["encoder_out"] = [list(enc_out.shape), "float32"]

    # Decoder: attentive pooling + BN + Conv1d(192).
    # Hook the pool output too.
    pool_out_holder = {}
    def pool_hook(_mod, _inp, output):
        pool_out_holder["pool"] = output.detach().cpu()
    h = model.decoder._pooling.register_forward_hook(pool_hook)

    with torch.no_grad():
        logits, emb = model.decoder(encoder_output=enc_out, length=enc_len)
    h.remove()

    pool = pool_out_holder["pool"]
    print(f"  pool out: {tuple(pool.shape)}")
    pool.numpy().astype(np.float32).tofile(out / "pool_out.f32")
    shapes["pool_out"] = [list(pool.shape), "float32"]

    print(f"  embedding: {tuple(emb.shape)}")
    emb.detach().numpy().astype(np.float32).tofile(out / "embedding.f32")
    shapes["embedding"] = [list(emb.shape), "float32"]

    print(f"  embedding norm: {emb.norm().item():.4f}")
    print(f"  embedding range: [{emb.min().item():.4f}, {emb.max().item():.4f}]")

    with open(out / "shapes.json", "w") as f:
        json.dump(shapes, f, indent=2)
    print(f"\nwrote fixtures to {out}/")


if __name__ == "__main__":
    main()
