# Makefile for ggml-based NeMo ASR implementation

GGML_DIR = /var/data/nvidia-speech/ggml
GGML_BUILD = $(GGML_DIR)/build

CXX = g++
CXXFLAGS = -g -std=c++17 -Wall -Wextra -O2
CXXFLAGS += -I $(GGML_DIR)/include
CXXFLAGS += -I include

# Check if CUDA backend is available
CUDA_LIB = $(GGML_BUILD)/src/ggml-cuda/libggml-cuda.so
CUDA_AVAILABLE = $(shell test -f $(CUDA_LIB) && echo 1 || echo 0)

LDFLAGS = -L $(GGML_BUILD)/src
LDFLAGS += -lggml -lggml-base -lggml-cpu
LDFLAGS += -Wl,-rpath,$(GGML_BUILD)/src
LDFLAGS += -lm -lpthread

# Add CUDA support if available
ifeq ($(CUDA_AVAILABLE),1)
    CXXFLAGS += -DGGML_USE_CUDA
    LDFLAGS += -L $(GGML_BUILD)/src/ggml-cuda -lggml-cuda
    LDFLAGS += -Wl,-rpath,$(GGML_BUILD)/src/ggml-cuda
    LDFLAGS += -L /usr/local/cuda/lib64 -lcudart -lcublas
    LDFLAGS += -Wl,-rpath,/usr/local/cuda/lib64
endif

# Source files
GGML_SRCS = src/nemo-ggml.cpp src/preprocessor.cpp
GGML_STREAM_SRCS = src/nemo-stream.cpp

# Original implementation (for comparison tests)
ORIG_SRCS = src/reference/ggml_weights.cpp src/reference/ops.cpp src/reference/conv_subsampling.cpp src/reference/conformer_modules.cpp src/reference/conformer_encoder.cpp src/reference/rnnt_decoder.cpp src/reference/rnnt_joint.cpp src/reference/greedy_decode.cpp src/reference/tokenizer.cpp

.PHONY: all clean clean_bin test transcribe streaming

all: test_ggml_weights test_ggml_compute transcribe streaming

streaming: test_streaming transcribe_stream

# Test weight loading
test_ggml_weights: tests/test_weights.cpp $(GGML_SRCS) $(ORIG_SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Test computation
test_ggml_compute: tests/test_compute.cpp $(GGML_SRCS) $(ORIG_SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Precompute encoder reference output (run once, saves ~2 min per test run)
precompute_encoder_ref: scripts/precompute_encoder_ref.cpp $(ORIG_SRCS)
	$(CXX) $(CXXFLAGS) $^ -I include -o $@

# Transcribe example
transcribe: src/transcribe.cpp $(GGML_SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Streaming test
test_streaming: tests/test_streaming.cpp $(GGML_SRCS) $(GGML_STREAM_SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Python reference comparison test
test_python_ref: tests/test_python_reference.cpp $(GGML_SRCS) $(GGML_STREAM_SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Preprocessor test
test_preprocessor: tests/test_preprocessor.cpp $(GGML_SRCS) $(GGML_STREAM_SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Streaming transcribe example
transcribe_stream: src/transcribe_stream.cpp $(GGML_SRCS) $(GGML_STREAM_SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

clean:
	rm -f test_ggml_weights test_ggml_compute precompute_encoder_ref transcribe test_streaming transcribe_stream test_python_ref test_preprocessor

clean_bin:
	rm my_bin/*

test: test_ggml_weights test_ggml_compute
	./test_ggml_weights
	./test_ggml_compute

test_stream: test_streaming
	./test_streaming

test_ref: test_python_ref
	./test_python_ref
