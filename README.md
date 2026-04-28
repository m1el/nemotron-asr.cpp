# Nemotron ASR GGML port

This is a rewrite of Automatic Speech Recognition from NVidia's NeMo framework.

The goal of this project is to have a fast-loading, low-dependency speech recognition software. This program only uses [ggml-org/ggml](https://github.com/ggml-org/ggml) library to work with neural networks.

## Demo

[![asciicast](https://asciinema.org/a/J1MAnH3Z93HIBMBA.svg)](https://asciinema.org/a/J1MAnH3Z93HIBMBA)

## Usage

The program expects raw single-channel s16le 16kHz samples as an input. Standard input denoted as `-`.
```bash
# read from standard input
ffmpeg  -hide_banner -loglevel error -i your-file.mp3 -ar 16000 -ac 1 -f s16le -  \
    | ./nemotron-asr.cpp weights/nemotron-speech-streaming-0.6B-v0.1.Q8_0.gguf - 70 13

# read from a file
ffmpeg  -hide_banner -loglevel error -i your-file.mp3 -ar 16000 -ac 1 -f s16le raw-audio.pcm
./nemotron-asr.cpp weights/nemotron-speech-streaming-0.6B-v0.1.Q8_0.gguf raw-audio.pcm 70 13
```

## Model Weights

Full and quantized versions of the models can be downloaded from Hugging Face Hub: https://huggingface.co/m1el/nemotron-speech-streaming-0.6B-gguf

Notice: the weights are [Licensed by NVIDIA Corporation under the NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)

Or converted from https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b using [convert_to_gguf.py](scripts/convert_to_gguf.py)

## Speaker diarization (optional)

If you also want speaker labels in the output, build a separate
`diarize.gguf` from MarbleNet (VAD) + TitaNet-L (speaker embeddings):

```bash
uv run scripts/convert_diarize_to_gguf.py weights/diarize.gguf
```

(This downloads `vad_multilingual_marblenet` and `titanet_large` from NeMo's
pretrained registry on first run.)

Then run with the `--diarize` flag:

```bash
ffmpeg -hide_banner -loglevel error -i your-file.mp3 -ar 16000 -ac 1 -f s16le - \
  | ./nemotron-asr.cpp weights/nemotron-speech-streaming-0.6B-v0.1.Q8_0.gguf - 80 0 \
        --diarize weights/diarize.gguf \
        --num-speakers 2 \
        --rttm out.rttm \
        --speaker-text out.spk.txt
```

Flags:

- `--diarize <gguf>`     enable diarization (~89 MB extra GGUF)
- `--num-speakers K`     force K speakers; otherwise NME-SC estimates
- `--sub-shift sec`      sub-segment shift, default 0.75 s
- `--rttm <path>`        write RTTM-format output for evaluation
- `--speaker-text <path>` write the speaker-tagged transcript at EOF
                          (defaults to stdout when `--diarize` is set)
- `--json <path>`        per-word JSON lines emitted as the audio streams

The diarization side runs alongside ASR: VAD on each 0.63 s window as the
audio arrives, embedding each 1.5 s sub-segment immediately, audio dropped
behind the cursor. Clustering runs once at end-of-input. See
[docs/DIARIZATION_PLAN.md](docs/DIARIZATION_PLAN.md) for design notes.

## Model weight differences from NeMo

The original tensors in the `nvidia/nemotron-speech-streaming-en-0.6b` require transposition to be used in matrix multiplication in ggml.
Additionally, those changes also help with quantization, which has requirements on tensor shape. For details, see [TENSOR_SHAPES.md](docs/TENSOR_SHAPES.md)

## Development

ggml and Eigen are vendored as git submodules. After cloning:

```bash
git submodule update --init --recursive
```

(Or pass `--recurse-submodules` to the original `git clone`.)

Build ggml:

```bash
cmake -S ggml -B ggml/build
cmake --build ggml/build -j8
```

Then build this binary:

```bash
make nemotron-asr.cpp
```

Eigen is header-only (`vendor/eigen`) — used by the diarization code, no separate build step.

## Comparison to [whisper.cpp](https://github.com/ggml-org/whisper.cpp)?

1) Whisper operates on 30s audio chunks, so it is not feasible to use in interactive applications.
NeMotron's ASR model has configurable latency (from 80ms to 1.12s), which can traded for quality and speed. (bigger lookahead gives better quality and bigger chunks require less work)
2) Whisper has troubles on long audio streams, where NeMotron's ASR is a streaming model, works for infinite streams. It does get stuck in lowercase mode though.
3) Quiality seems to be better

## License

MIT
