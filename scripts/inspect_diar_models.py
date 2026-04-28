#!/usr/bin/env python3
"""
Download MarbleNet (VAD) and TitaNet-L (speaker embedding) checkpoints from
NeMo's pretrained model registry, save them as local .nemo archives, and dump
their state_dict tensor names + shapes for converter design.

Usage:  uv run scripts/inspect_diar_models.py
"""

import json
import os
import sys
from pathlib import Path

import torch

import nemo.collections.asr as nemo_asr


# Where to cache the downloaded .nemo archives.
CACHE_DIR = Path(__file__).parent.parent / "weights" / "nemo_src"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch(model_name: str, klass) -> Path:
    """Download a pretrained model and save it as <CACHE_DIR>/<name>.nemo."""
    out_path = CACHE_DIR / f"{model_name}.nemo"
    if out_path.exists():
        print(f"  [cache hit] {out_path}")
        return out_path
    print(f"  downloading {model_name}...")
    model = klass.from_pretrained(model_name=model_name)
    model.save_to(str(out_path))
    print(f"  saved {out_path}")
    return out_path


def dump_state_dict(nemo_path: Path):
    """Dump tensor names + shapes from a .nemo archive."""
    import tarfile, tempfile, yaml
    with tarfile.open(nemo_path) as tar, tempfile.TemporaryDirectory() as td:
        names = tar.getnames()
        ckpt_name = next(n for n in names if n.endswith("model_weights.ckpt"))
        cfg_name = next((n for n in names if n.endswith("model_config.yaml")), None)
        tar.extract(ckpt_name, td)
        sd = torch.load(os.path.join(td, ckpt_name), map_location="cpu", weights_only=True)
        cfg = None
        if cfg_name:
            tar.extract(cfg_name, td)
            with open(os.path.join(td, cfg_name)) as f:
                cfg = yaml.safe_load(f)
    return sd, cfg


def main():
    print("== MarbleNet (vad_multilingual_marblenet) ==")
    vad_path = fetch("vad_multilingual_marblenet", nemo_asr.models.EncDecClassificationModel)

    print("\n== TitaNet-L (titanet_large) ==")
    spk_path = fetch("titanet_large", nemo_asr.models.EncDecSpeakerLabelModel)

    print("\n--- VAD state_dict ---")
    vad_sd, vad_cfg = dump_state_dict(vad_path)
    for k, v in vad_sd.items():
        print(f"  {k:80s}  {tuple(v.shape)}  {v.dtype}")
    print(f"  total tensors: {len(vad_sd)}")
    print(f"  total params:  {sum(v.numel() for v in vad_sd.values()):,}")

    print("\n--- VAD preprocessor cfg ---")
    if vad_cfg and "preprocessor" in vad_cfg:
        print(json.dumps(vad_cfg["preprocessor"], indent=2, default=str))

    print("\n--- VAD encoder cfg ---")
    if vad_cfg and "encoder" in vad_cfg:
        print(json.dumps(vad_cfg["encoder"], indent=2, default=str))

    print("\n\n--- SPK state_dict ---")
    spk_sd, spk_cfg = dump_state_dict(spk_path)
    for k, v in spk_sd.items():
        print(f"  {k:80s}  {tuple(v.shape)}  {v.dtype}")
    print(f"  total tensors: {len(spk_sd)}")
    print(f"  total params:  {sum(v.numel() for v in spk_sd.values()):,}")

    print("\n--- SPK preprocessor cfg ---")
    if spk_cfg and "preprocessor" in spk_cfg:
        print(json.dumps(spk_cfg["preprocessor"], indent=2, default=str))

    print("\n--- SPK encoder cfg ---")
    if spk_cfg and "encoder" in spk_cfg:
        print(json.dumps(spk_cfg["encoder"], indent=2, default=str))

    print("\n--- SPK decoder cfg ---")
    if spk_cfg and "decoder" in spk_cfg:
        print(json.dumps(spk_cfg["decoder"], indent=2, default=str))


if __name__ == "__main__":
    main()
