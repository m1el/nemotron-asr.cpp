# Diarization Port Plan

Goal: reproduce the result of `NeMo/tutorials/speaker_tasks/ASR_with_SpeakerDiarization.ipynb`
inside this project, sharing the existing GGML/GGUF infrastructure with the ASR side.

**Weight distribution:** the existing ASR `.gguf` is untouched. Diarization weights ship
in a separate `diarize.gguf` containing both subnets (VAD + speaker) under `vad.*` and
`spk.*` tensor namespaces. The CLI loads both files independently. Bundling everything
into one file is a possible future change but is **out of scope here.**

## Pipeline (mirrors the NeMo tutorial)

```
audio (16 kHz mono)
   │
   ├─► [existing] preprocessor + RNN-T encoder/decoder ─► words + word timestamps
   │
   └─► MarbleNet VAD ─► speech segments
                          │
                          ▼
                   sub-segment splitter (1.5 s windows, 0.75 s shift)
                          │
                          ▼
                   TitaNet-L encoder ─► 192-d L2-normalized embedding per sub-segment
                          │
                          ▼
                   NME-SC clustering ─► speaker label per sub-segment
                          │
                          ▼
                   contiguous-segment merge ─► RTTM (start, dur, speaker)
                          │
                          ▼
   word ↔ speaker alignment via timestamp overlap
                          │
                          ▼
   speaker-tagged transcript + RTTM file
```

## Scope decisions (proposed — please confirm/override)

1. **Use MarbleNet VAD, not ASR-based VAD.** It is small (~88K params, ~0.35 MB fp32),
   matches the tutorial's default config, and avoids coupling diarization timing to the
   RNN-T blank-frame heuristics.
2. **Single scale, not multi-scale.** NeMo defaults to a 5-scale weighted clustering
   (1.5 / 1.25 / 1.0 / 0.75 / 0.5 s with cosine-mean weights). MVP uses single 1.5 s scale
   with 50% overlap. Quality should still be acceptable for ≤4 speakers in the tutorial
   sample; revisit if results are poor.
3. **TitaNet-Large** (`titanet_large`, ~23 M params, 192-d output) — same as the tutorial.
4. **Offline only.** No streaming diarization in this pass. Streaming end-to-end
   diarization (Sortformer, etc.) is out of scope.
5. **Separate `diarize.gguf`** alongside the existing ASR file. ASR converter, ASR loader,
   and shipped ASR GGUFs are not modified.
6. **No language-model realignment, no beam-search CTC decoder, no cpWER scoring** in v1
   (tutorial's "Optional Features" section). Add later if needed.

## diarize.gguf layout

Single new file (separate from ASR), two subnets namespaced by tensor-name prefix:

```
vad.encoder.<jasper-block-name>...   (MarbleNet)
vad.decoder.<linear-name>
spk.encoder.<jasper-block-name>...   (TitaNet)
spk.decoder.attention_pool.*
spk.decoder.emb.*
```

KV pairs:

```
general.architecture = "nemo-diarize"
vad.sample_rate, vad.n_mfcc, vad.n_fft, vad.window_size, vad.window_stride,
vad.hop_length, vad.labels (= "background,speech")
spk.sample_rate, spk.n_mels, spk.n_fft, spk.emb_dim, spk.enc_feat_out
```

The CLI takes the ASR gguf and the diarize gguf as separate arguments.

## Subnet 1: MarbleNet VAD (`vad_multilingual_marblenet`)

Reference config: lifted directly from the published `.nemo` checkpoint (the
`marblenet_3x2x64.yaml` example file in the NeMo repo describes the *English-only*
variant which differs in front-end). 84 tensors, ~114K params total in the
multilingual variant.

- **Front-end:** the multilingual checkpoint actually uses
  `AudioToMelSpectrogramPreprocessor` with **80 log-mel** (not the 64-MFCC of the
  English `marblenet_3x2x64`), 25 ms window, 10 ms stride, hann, n_fft=512,
  `normalize: None` (raw log-mel, no per-feature normalization). This is the same
  shape as TitaNet's preprocessor, just without normalization.
- **Encoder (`ConvASREncoder`, "Jasper" blocks):**
  - Block 0: separable, k=11, 64→128, no residual
  - Blocks 1–3: separable, k=13/15/17, 128→64, repeat=2, residual
  - Block 4: separable, k=29, dilation=2, 64→128, no residual
  - Block 5: 1×1, 128→128, not separable
  Each "separable" sub-conv = depthwise + pointwise + BN + ReLU(+ dropout in train).
- **Decoder:** `ConvASRDecoderClassification` — global avg-pool over time, linear 128→2.
- **Inference:** softmax → speech-probability frame stream → threshold/onset/offset to
  segments. NeMo defaults: onset 0.5, offset 0.3, min_duration_on 0.1 s, min_duration_off
  0.2 s, pad_onset 0.05 s, pad_offset −0.05 s.

**Verification:** dump frame-level logits from PyTorch on `an4_diarize_test.wav`,
compare to ggml output element-wise (rtol≈1e-4 fp32, looser for quantized).

## Subnet 2: TitaNet-L speaker encoder

Reference config: `NeMo/examples/speaker_tasks/recognition/conf/titanet-large.yaml`.

- **Front-end:** `AudioToMelSpectrogramPreprocessor` — 80-mel, per-feature normalize,
  25 ms / 10 ms, n_fft=512. Re-uses most of the existing ASR mel code, but the ASR
  model uses 128 mels with different normalization, so we need a separate config path.
- **Encoder (Jasper + SE):** all 5 blocks have SE in this checkpoint (including the
  no-residual prologue and epilogue).
  - Block 0: separable, k=3, 80→1024, residual=False, SE
  - Blocks 1–3: separable, k=7/11/15, 1024→1024, repeat=3, residual=True, SE
  - Block 4: separable, k=1, 1024→3072, no residual, SE (squeeze ratio 8: 3072→384)
  - SE block: global avg over time → linear (C→C/8) → ReLU → linear (C/8→C) →
    sigmoid → multiply per-channel.
- **Decoder (`SpeakerDecoder`, attention pooling):** input is (B, 3072, T).
  - Compute mean/std along T (length-masked).
  - Build context: `concat([x, mean, std], dim=channels)` → (B, **9216**, T).
  - `TDNNModule(9216→128, k=1)` = Conv1d + ReLU + BN(128).  *(yes, ReLU before BN.)*
  - `Tanh`.
  - Conv1d(128→3072, k=1).
  - Mask, softmax over T → α (B, 3072, T).
  - Weighted mean and weighted std of original x using α as weights → 2 × (B, 3072).
  - Concat → (B, 6144, 1).
  - BN(6144) → Conv1d(6144→192, k=1) → squeeze T=1 → 192-d embedding.
  - L2 normalize → 192-d embedding. (We ignore the 16681-class classifier `decoder.final`.)
- **Inference unit:** 1.5 s window. TitaNet was trained on 3 s segments but is robust to
  shorter; NeMo's diarizer defaults sit at 1.5 s as well.

**Verification:** embed a fixed clip in PyTorch and ggml; cosine similarity should be
≥0.9999 in fp32.

## Subnet 3: NME-SC clustering (no weights)

Pure C++/ggml-on-CPU math, ported from
`NeMo/nemo/collections/asr/parts/utils/offline_clustering.py`.

1. Affinity `A_ij = (1 + cos(e_i, e_j)) / 2`, zero diagonal.
2. For each candidate `p ∈ {p_min..p_max}` (typically ~5..30, capped at N−1):
   - keep top-`p` neighbours per row, symmetrize → `A_p`
   - eigen-decompose `L = I − D^{-1/2} A_p D^{-1/2}`
   - score = ratio of (p-th eigengap) to (max eigengap among p..k_max)
3. Pick `p*` minimising NME score.
4. Re-eigendecompose `L_{p*}`, count speakers via max eigengap (or honour `oracle_num_speakers`).
5. K-means (Lloyd, cosine-init or k-means++) on top-K eigenvectors of `L_{p*}` → labels.

Implementation notes:
- Affinity is N×N where N = #sub-segments (typically <2000 in 10-min audio), so dense
  EVD via LAPACK or a tiny in-tree symmetric eigensolver is fine. We are CPU-only here;
  ggml does not provide eig.
- Use Eigen's `SelfAdjointEigenSolver` (vendored at `vendor/eigen`, header-only, no
  runtime dep). Decision recorded after benching: a hand-rolled Jacobi solver was
  100–500× slower at our problem sizes (30 s @ N=512 vs. 60 ms in Eigen), making it
  unusable inside the NME-SC inner loop. See `scratch/bench_jacobi.cpp` /
  `scratch/bench_eigen.cpp`.
- All numbers fp64 to be safe; clusters are only a few thousand items.

**Verification:** golden affinity + label set generated by NeMo on the tutorial audio,
loaded as a fixture; require exact label match (modulo permutation).

## Glue: VAD → sub-segments → embeddings → labels → words

- Build sub-segments: walk each VAD speech region, emit 1.5 s windows shifted by 0.75 s.
  Pad the last window if remaining region < 1.5 s but ≥ ~0.5 s (NeMo's `min_subsegment`).
- Run TitaNet over each sub-segment in a batch loop. Memory: 192 × N × 4 B ≈ 800 KB at
  N=1000. Trivial.
- After clustering, merge adjacent same-speaker sub-segments to get RTTM rows.
- For each ASR word with `(t_start, t_end)`, pick the speaker label whose timeline has
  max overlap with the word interval. NeMo also supports anchor-offset tweaks
  (`word_ts_anchor_offset`); MVP just uses the midpoint.

## Output

- `<input>.rttm`: standard RTTM (`SPEAKER <id> 1 <start> <dur> <NA> <NA> spk_K <NA> <NA>`)
- `<input>.diar.txt`: lines `[t0 - t1] spk_K: words ...`
- (optional) `<input>.diar.json`: per-word records like
  `{ "word": "...", "start": ..., "end": ..., "speaker": "spk_0" }`

## Tests / fixtures

Add under `tests/diarize/`:
- `an4_diarize_test.wav` (fetch from
  `https://nemo-public.s3.us-east-2.amazonaws.com/an4_diarize_test.wav`, ~90 KB).
- `an4_marblenet_logits.bin` — NeMo VAD frame logits.
- `an4_titanet_emb.bin` — NeMo embeddings for the same sub-segment grid we'll use.
- `an4_labels.json` — NeMo's diarized labels and final speaker-tagged transcript.

A single `tests/run_diarize_tests.sh` (or extension to existing test driver) runs the
binary and diffs against fixtures with tolerance.

## File layout (additions only)

```
nemotron-asr.cpp/
├── docs/
│   └── DIARIZATION_PLAN.md         (this file)
├── scripts/
│   ├── convert_diarize_to_gguf.py  (new — produces diarize.gguf from two .nemo files)
│   ├── export_marblenet_ref.py     (new — dump VAD logits for tests)
│   └── export_titanet_ref.py       (new — dump speaker embeddings for tests)
├── src/
│   ├── preprocessor.cpp / .h       (extended: MFCC variant for MarbleNet)
│   ├── vad.cpp / vad.h             (new — MarbleNet)
│   ├── speaker.cpp / speaker.h     (new — TitaNet)
│   ├── diarize_cluster.cpp / .h    (new — NME-SC + k-means)
│   ├── diarize.cpp / diarize.h     (new — pipeline glue, RTTM/JSON output)
│   └── transcribe_diarize.cpp      (new — CLI binary)
└── tests/
    └── diarize/                    (new — fixtures + driver)
```

## Phasing / order of work

1. Write `convert_diarize_to_gguf.py` (0.5 day; large code reuse from existing converter).
2. MarbleNet end-to-end with golden fixture (2–3 days; small model, mostly preprocessor).
3. TitaNet end-to-end with golden fixture (4–6 days; bigger, SE + attention pooling new).
4. NME-SC clustering with golden fixture (2 days).
5. Glue + RTTM/transcript output (1–2 days).
6. CLI + README (0.5 day).

Quantization can be added later once the pipeline matches NeMo's labels.

## Resolved decisions

- **Eigensolver:** Eigen 3.4.0 vendored at `vendor/eigen` (header-only). Hand-rolled
  Jacobi rejected on bench grounds (above).
- **Single-scale clustering** (1.5 s window, 0.75 s shift) for v1.
- **Multilingual MarbleNet** (`vad_multilingual_marblenet`) — matches the tutorial.

## Future work

- **Multi-scale clustering** (5 scales, cosine-mean weighted affinity, optional learned
  MSDD weights). Affects: sub-segment generation, affinity matrix construction, and
  per-segment scale-anchoring lookup. Eigensolver, k-means, glue layer are unchanged.
  Estimated +2–3 days on top of v1.
- Optional: VAD-only / speaker-embedding-only binaries for users who want just one piece.
- Optional: language-model realignment, beam-search CTC, cpWER metric (tutorial extras).
- Optional: streaming diarization (Sortformer / streaming end-to-end).
