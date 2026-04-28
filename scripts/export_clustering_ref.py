#!/usr/bin/env python3
"""
Generate test embeddings and run NeMo's NME-SC + SpectralClustering on them.
Saves the inputs and outputs as fixtures for the C++ port verifier:

    embeddings.f32        (N, 192)   normalized speaker embeddings
    affinity.f32          (N, N)     cosine affinity (after min-max scaling)
    labels.i32            (N,)       NeMo's clustering labels
    cluster_meta.json     algorithm parameters and N

For a stable, controllable test we synthesize embeddings around a few cluster
centroids derived from the real TitaNet output on the AN4 audio. This gives
a workload very similar to real diarization without depending on the audio
pipeline being wired up.
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.offline_clustering import (
    NMESC,
    SpectralClustering,
    getCosAffinityMatrix,
    getAffinityGraphMat,
)


def synthesize_embeddings_from_anchors(anchors: np.ndarray, n_per: int, sigma: float, seed: int):
    """For each anchor (n_anchors, D), produce n_per noisy copies."""
    rng = np.random.default_rng(seed)
    K, D = anchors.shape
    out = []
    labels = []
    for k in range(K):
        for _ in range(n_per):
            v = anchors[k] + sigma * rng.standard_normal(D).astype(np.float32)
            out.append(v)
            labels.append(k)
    embs = np.stack(out)
    # L2 normalize like real titanet outputs are typically used.
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    return embs.astype(np.float32), np.array(labels, dtype=np.int32)


def main():
    out = Path("tests/diarize/cluster_ref")
    out.mkdir(parents=True, exist_ok=True)

    # Anchor embeddings: pull two distinct ones from the AN4 audio (one near
    # the start, one near the end — likely different speakers).
    print("loading titanet_large for anchor embeddings...")
    spk = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large")
    spk.eval().to("cpu")
    audio, sr = sf.read("tests/diarize/an4_diarize_test.wav", dtype="float32")
    assert sr == 16000

    def emb_of(start_s: float, dur_s: float = 1.5) -> np.ndarray:
        s = int(start_s * sr)
        e = s + int(dur_s * sr)
        sig = torch.from_numpy(audio[s:e]).unsqueeze(0)
        ln = torch.tensor([e - s], dtype=torch.int64)
        with torch.no_grad():
            mel, ml = spk.preprocessor(input_signal=sig, length=ln)
            enc, el = spk.encoder(audio_signal=mel, length=ml)
            _, e_emb = spk.decoder(encoder_output=enc, length=el)
        v = e_emb.squeeze(0).numpy().astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-8)
        return v

    anchor_a = emb_of(1.0)  # first speaker region
    anchor_b = emb_of(3.5)  # second speaker region
    anchor_c = emb_of(2.2)  # mid (could be same as one of the above)
    anchors = np.stack([anchor_a, anchor_b])  # use 2 anchors for the AN4 setup

    print(f"  anchor cos: a·b={float(anchor_a @ anchor_b):.3f}, "
          f"a·c={float(anchor_a @ anchor_c):.3f}, "
          f"b·c={float(anchor_b @ anchor_c):.3f}")

    embs, true_labels = synthesize_embeddings_from_anchors(
        anchors, n_per=30, sigma=0.10, seed=0)
    N, D = embs.shape
    print(f"  embeddings: {N} × {D}")
    embs.tofile(out / "embeddings.f32")

    # Cosine affinity (NeMo's getCosAffinityMatrix: cos_sim with eps=3.5e-4 and
    # diagonal forced to 1, then min-max scaled).
    embs_t = torch.from_numpy(embs).float()
    aff = getCosAffinityMatrix(embs_t)
    aff_np = aff.numpy().astype(np.float32)
    aff_np.tofile(out / "affinity.f32")
    print(f"  affinity: {aff_np.shape} range [{aff_np.min():.4f}, {aff_np.max():.4f}]")

    # Run NMESC to get p_hat and est_num_of_spk.
    nmesc = NMESC(
        aff,
        max_num_speakers=8,
        max_rp_threshold=0.25,
        sparse_search=True,
        sparse_search_volume=30,
        nme_mat_size=512,
        use_subsampling_for_nme=True,
        fixed_thres=-1.0,
        maj_vote_spk_count=False,
        parallelism=False,
        cuda=False,
    )
    est_num_spk, p_hat = nmesc.forward()
    print(f"  NMESC -> est_num_spk={int(est_num_spk)} p_hat={int(p_hat)}")

    # Spectral clustering on the p_hat affinity.
    pruned = getAffinityGraphMat(aff, int(p_hat))
    sc = SpectralClustering(n_clusters=int(est_num_spk), n_random_trials=1, cuda=False)
    labels = sc.forward(pruned)
    labels_np = labels.numpy().astype(np.int32)
    labels_np.tofile(out / "labels.i32")
    print(f"  labels: {labels_np[:30]} ...")
    print(f"  per-cluster counts: {np.bincount(labels_np)}")

    meta = {
        "N": int(N),
        "D": int(D),
        "anchors_dot_ab": float(anchor_a @ anchor_b),
        "true_labels_first30": true_labels[:30].tolist(),
        "est_num_spk": int(est_num_spk),
        "p_hat": int(p_hat),
        "max_num_speakers": 8,
        "max_rp_threshold": 0.25,
        "sparse_search_volume": 30,
        "nme_mat_size": 512,
    }
    with open(out / "cluster_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote fixtures to {out}/")


if __name__ == "__main__":
    main()
