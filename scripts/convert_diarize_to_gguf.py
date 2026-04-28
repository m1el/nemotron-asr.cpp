#!/usr/bin/env python3
"""
Convert MarbleNet (VAD) and TitaNet-L (speaker embedding) NeMo checkpoints into
a single diarize.gguf containing both subnets, namespaced by tensor-name prefix:

    vad.<orig PyTorch name>      MarbleNet (vad_multilingual_marblenet)
    spk.<orig PyTorch name>      TitaNet-L (titanet_large)

The existing ASR convert_to_gguf.py and the GGUF files it produces are not touched.

Usage:
    uv run scripts/convert_diarize_to_gguf.py \
        --vad weights/nemo_src/vad_multilingual_marblenet.nemo \
        --spk weights/nemo_src/titanet_large.nemo \
        weights/diarize-v0.1.f32.gguf

If --vad / --spk paths are omitted, the script will download via NeMo's pretrained
registry into weights/nemo_src/.

This v0.1 keeps everything in F32 — quantization is added later, after the
inference path matches NeMo numerically.
"""

import argparse
import os
import struct
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

# Reuse small writer helpers from the ASR converter.
sys.path.insert(0, str(Path(__file__).parent))
from convert_to_gguf import (  # noqa: E402
    GGUF_MAGIC,
    GGUF_VERSION,
    GGUF_DEFAULT_ALIGNMENT,
    GGML_TYPE_F32,
    write_string,
    write_kv_string,
    write_kv_uint32,
    write_kv_float32,
)


CACHE_DIR = Path(__file__).parent.parent / "weights" / "nemo_src"


# ---------------------------------------------------------------------------
# .nemo loading
# ---------------------------------------------------------------------------

def load_nemo(path: Path) -> tuple[dict, dict]:
    """Return (state_dict_as_numpy, config_dict) from a .nemo tarball."""
    with tarfile.open(path) as tar, tempfile.TemporaryDirectory() as td:
        names = tar.getnames()
        ckpt_name = next(n for n in names if n.endswith("model_weights.ckpt"))
        cfg_name = next(n for n in names if n.endswith("model_config.yaml"))

        tar.extract(ckpt_name, td)
        tar.extract(cfg_name, td)

        sd = torch.load(os.path.join(td, ckpt_name), map_location="cpu", weights_only=True)
        with open(os.path.join(td, cfg_name)) as f:
            cfg = yaml.safe_load(f)

    np_sd = {}
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            np_sd[k] = v.detach().cpu().numpy()
        else:
            np_sd[k] = v
    return np_sd, cfg


def fetch_pretrained(model_name: str, klass_path: str) -> Path:
    """Download a NeMo pretrained model to CACHE_DIR/<name>.nemo if missing."""
    out = CACHE_DIR / f"{model_name}.nemo"
    if out.exists():
        return out
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {model_name} via NeMo...")
    import importlib
    mod_path, klass_name = klass_path.rsplit(".", 1)
    klass = getattr(importlib.import_module(mod_path), klass_name)
    model = klass.from_pretrained(model_name=model_name)
    model.save_to(str(out))
    print(f"  saved {out}")
    return out


# ---------------------------------------------------------------------------
# Tensor selection / reshape
# ---------------------------------------------------------------------------

# Tensors we drop entirely:
#   *.num_batches_tracked        BN bookkeeping, irrelevant at inference
#   spk.decoder.final.*          16681-class speaker classifier; we only want emb
#   *.preprocessor.featurizer.*  preprocessor weights (window, fb) — we recompute
#                                them in C++ from sample_rate/n_mels/n_fft for
#                                deterministic, dependency-free behavior.
DROP_SUFFIXES = (".num_batches_tracked",)
DROP_SUBSTRINGS_PER_PREFIX = {
    "spk": ("decoder.final.",),
}
DROP_PREFIX_PER_NAMESPACE = {
    "vad": ("preprocessor.",),
    "spk": ("preprocessor.",),
}


def should_drop(orig_name: str, namespace: str) -> Optional[str]:
    """Return a reason string if tensor should be dropped, else None."""
    if orig_name.endswith(DROP_SUFFIXES):
        return "BN num_batches_tracked"
    for sub in DROP_SUBSTRINGS_PER_PREFIX.get(namespace, ()):
        if sub in orig_name:
            return f"in DROP_SUBSTRINGS for {namespace}"
    for pre in DROP_PREFIX_PER_NAMESPACE.get(namespace, ()):
        if orig_name.startswith(pre):
            return f"prefix '{pre}' dropped for {namespace}"
    return None


def reshape_for_ggml(orig_name: str, data: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Apply the same conv reshape conventions used by the ASR converter:
      - pointwise conv  (out, in, 1)   -> (out, in)        2D matmul-friendly
      - depthwise conv  (ch, 1, k)     -> (k, ch)          stride-1 friendly

    Pointwise convs in MarbleNet/TitaNet are named:
      encoder.encoder.<i>.mconv.{1,6,11}.conv.weight
      encoder.encoder.<i>.res.0.0.conv.weight
      decoder._pooling.attention_layer.0.conv_layer.weight
      decoder._pooling.attention_layer.2.weight
      decoder.emb_layers.0.1.weight
    Depthwise convs are named:
      encoder.encoder.<i>.mconv.{0,5,10}.conv.weight  (groups=ch, in=1)
    SE conv1d-as-linear "fc" weights are stored as 2D Linear weights
    (out, in) — already 2D, leave as-is.

    Returns (reshaped_data, note) where note is a string like "pointwise->2D".
    """
    if data.ndim == 3:
        out, mid, k = data.shape
        if k == 1 and mid >= 1:
            # pointwise: (out, in, 1) -> (out, in)
            return data.squeeze(axis=2), "pointwise->2D"
        if mid == 1 and k > 1:
            # depthwise: (ch, 1, k) -> (k, ch)
            return data.squeeze(axis=1).T.copy(), "depthwise->(k,ch)"
    return data, ""


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def to_gguf(prefix: str, sd: dict) -> list[tuple[str, np.ndarray, int]]:
    """
    Apply prefix, drop irrelevant tensors, reshape conv weights.
    Returns list of (gguf_name, data, n_dims).
    """
    out = []
    for orig, arr in sd.items():
        reason = should_drop(orig, prefix)
        if reason is not None:
            print(f"  drop  {prefix}.{orig:70s}  ({reason})")
            continue
        if not isinstance(arr, np.ndarray):
            print(f"  skip  {prefix}.{orig:70s}  (non-array: {type(arr).__name__})")
            continue
        new_arr, note = reshape_for_ggml(orig, arr)
        n_dims = max(1, new_arr.ndim)
        new_name = f"{prefix}.{orig}"
        print(f"  keep  {new_name:78s}  shape={tuple(new_arr.shape)}  {note}")
        out.append((new_name, new_arr.astype(np.float32, copy=False), n_dims))
    return out


def gather_hparams(vad_cfg: dict, spk_cfg: dict) -> dict:
    """Hyperparameters needed by the C++ side at load time."""
    h: dict = {}

    # General header.
    h["general.architecture"] = "nemo-diarize"
    h["general.name"] = "nemo-diarize-v0.1"

    # MarbleNet preprocessor (audio -> 80-mel log-spectrogram, no normalization).
    p = vad_cfg["preprocessor"]
    h["vad.sample_rate"] = int(p["sample_rate"])
    h["vad.n_mels"] = int(p["features"])
    h["vad.n_fft"] = int(p["n_fft"])
    h["vad.window_size"] = float(p["window_size"])
    h["vad.window_stride"] = float(p["window_stride"])
    h["vad.dither"] = float(p.get("dither", 0.0))
    h["vad.normalize"] = str(p.get("normalize", "None"))
    h["vad.window"] = str(p.get("window", "hann"))
    h["vad.n_classes"] = 2  # background, speech

    # TitaNet preprocessor (80-mel, per-feature normalize).
    p = spk_cfg["preprocessor"]
    h["spk.sample_rate"] = int(p["sample_rate"])
    h["spk.n_mels"] = int(p["features"])
    h["spk.n_fft"] = int(p["n_fft"])
    h["spk.window_size"] = float(p["window_size"])
    h["spk.window_stride"] = float(p["window_stride"])
    h["spk.dither"] = float(p.get("dither", 0.0))
    h["spk.normalize"] = str(p.get("normalize", "per_feature"))
    h["spk.window"] = str(p.get("window", "hann"))
    h["spk.emb_dim"] = 192
    h["spk.attn_channels"] = 128

    return h


def write_gguf(out_path: Path, hparams: dict, tensors: list[tuple[str, np.ndarray, int]]):
    print(f"\nWriting {out_path} ...")

    # Compute tensor offsets up front (data section is aligned).
    tensor_infos = []
    cur_off = 0
    for name, data, n_dims in tensors:
        # GGUF stores shape in reverse order, padded to 4 entries.
        shape_gguf = list(reversed(data.shape))
        while len(shape_gguf) < 4:
            shape_gguf.append(1)
        aligned = (cur_off + GGUF_DEFAULT_ALIGNMENT - 1) // GGUF_DEFAULT_ALIGNMENT * GGUF_DEFAULT_ALIGNMENT
        blob = data.astype(np.float32).tobytes()
        tensor_infos.append({
            "name": name,
            "shape": shape_gguf[:4],
            "n_dims": n_dims,
            "type": GGML_TYPE_F32,
            "offset": aligned,
            "data": blob,
        })
        cur_off = aligned + len(blob)

    with open(out_path, "wb") as f:
        f.write(GGUF_MAGIC)
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<q", len(tensor_infos)))
        f.write(struct.pack("<q", len(hparams)))

        for k, v in hparams.items():
            if isinstance(v, bool):
                # No bool helper in convert_to_gguf — use uint32 0/1.
                write_kv_uint32(f, k, int(v))
            elif isinstance(v, int):
                write_kv_uint32(f, k, v)
            elif isinstance(v, float):
                write_kv_float32(f, k, v)
            elif isinstance(v, str):
                write_kv_string(f, k, v)
            else:
                raise TypeError(f"unsupported hparam type for {k}: {type(v).__name__}")

        for info in tensor_infos:
            write_string(f, info["name"])
            f.write(struct.pack("<I", info["n_dims"]))
            for dim in info["shape"][:info["n_dims"]]:
                f.write(struct.pack("<q", dim))
            f.write(struct.pack("<i", info["type"]))
            f.write(struct.pack("<Q", info["offset"]))

        # Pad to alignment before tensor data.
        cur = f.tell()
        aligned = (cur + GGUF_DEFAULT_ALIGNMENT - 1) // GGUF_DEFAULT_ALIGNMENT * GGUF_DEFAULT_ALIGNMENT
        f.write(b"\x00" * (aligned - cur))

        data_start = f.tell()
        for info in tensor_infos:
            target = data_start + info["offset"]
            cur = f.tell()
            if target > cur:
                f.write(b"\x00" * (target - cur))
            f.write(info["data"])

        size = f.tell()

    total_params = sum(int(np.prod(d.shape)) for _, d, _ in tensors)
    print(f"Wrote {len(tensor_infos)} tensors ({total_params:,} params, {size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output", help="Output GGUF file path")
    ap.add_argument("--vad", type=Path, default=None, help="Path to MarbleNet .nemo (will download if absent)")
    ap.add_argument("--spk", type=Path, default=None, help="Path to TitaNet .nemo (will download if absent)")
    args = ap.parse_args()

    vad_path = args.vad or fetch_pretrained(
        "vad_multilingual_marblenet",
        "nemo.collections.asr.models.EncDecClassificationModel",
    )
    spk_path = args.spk or fetch_pretrained(
        "titanet_large",
        "nemo.collections.asr.models.EncDecSpeakerLabelModel",
    )

    print(f"\n== Loading VAD from {vad_path} ==")
    vad_sd, vad_cfg = load_nemo(vad_path)
    print(f"VAD: {len(vad_sd)} raw tensors")

    print(f"\n== Loading SPK from {spk_path} ==")
    spk_sd, spk_cfg = load_nemo(spk_path)
    print(f"SPK: {len(spk_sd)} raw tensors")

    print("\n== Mapping VAD tensors ==")
    vad_tensors = to_gguf("vad", vad_sd)

    print("\n== Mapping SPK tensors ==")
    spk_tensors = to_gguf("spk", spk_sd)

    hparams = gather_hparams(vad_cfg, spk_cfg)

    print("\n== Hparams ==")
    for k, v in hparams.items():
        print(f"  {k} = {v!r}")

    write_gguf(Path(args.output), hparams, vad_tensors + spk_tensors)
    print("Done.")


if __name__ == "__main__":
    main()
