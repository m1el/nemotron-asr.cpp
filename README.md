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

## Model weight differences from NeMo

The original tensors in the `nvidia/nemotron-speech-streaming-en-0.6b` require transposition to be used in matrix multiplication in ggml.
Additionally, those changes also help with quantization, which has requirements on tensor shape. For details, see [TENSOR_SHAPES.md](docs/TENSOR_SHAPES.md)

## Development

First, build ggml with all your favorite settings.
```bash
git subdmodule update --init
mkdir ggml/build
cd ggml/build
cmake ..
make -j8
cd ../..
```

Once you have built ggml, you can build this binary
```bash
make nemotron-asr.cpp
```

All of the ggml backends that you built with should be available to `nemotron-asr.cpp` through the `--backend` flag.

## Comparison to [whisper.cpp](https://github.com/ggml-org/whisper.cpp)?

1) Whisper operates on 30s audio chunks, so it is not feasible to use in interactive applications.
NeMotron's ASR model has configurable latency (from 80ms to 1.12s), which can traded for quality and speed. (bigger lookahead gives better quality and bigger chunks require less work)
2) Whisper has troubles on long audio streams, where NeMotron's ASR is a streaming model, works for infinite streams. It does get stuck in lowercase mode though.
3) Quality seems to be better

## License

MIT
