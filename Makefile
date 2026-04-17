# Makefile for ggml-based NeMo ASR implementation

GGML_DIR ?= ggml
GGML_BUILD ?= $(GGML_DIR)/build

CXX ?= g++
CXXFLAGS ?= -g -std=c++17 -Wall -Wextra -O2
CXXFLAGS += -I $(GGML_DIR)/include
CXXFLAGS += -I include

# Detect static libraries in GGML build directory
GGML_STATIC_LIBS := $(wildcard $(GGML_BUILD)/src/*.a)

# Configure linking based on whether static libraries are available
ifneq ($(GGML_STATIC_LIBS),)
    # Static linking - use .a files directly
    LDFLAGS += $(GGML_STATIC_LIBS)
else
    # Dynamic linking - use -l flags and rpath
    LDFLAGS += -L $(GGML_BUILD)/src
    LDFLAGS += -lggml -lggml-base
    LDFLAGS += -Wl,-rpath,$(GGML_BUILD)/src
    LDFLAGS += -Wl,-rpath,$(GGML_BUILD)/bin
endif

# Source files
GGML_SRCS = src/nemo-ggml.cpp src/preprocessor.cpp
GGML_STREAM_SRCS = src/nemo-stream.cpp
C_WRAPPER_SRCS = src/nemotron_asr_c.cpp

# Original implementation (for comparison tests)
ORIG_SRCS = src/reference/ggml_weights.cpp src/reference/ops.cpp src/reference/conv_subsampling.cpp src/reference/conformer_modules.cpp src/reference/conformer_encoder.cpp src/reference/rnnt_decoder.cpp src/reference/rnnt_joint.cpp src/reference/greedy_decode.cpp src/reference/tokenizer.cpp

.PHONY: all clean clean_bin test transcribe streaming lib

all: nemotron-asr.cpp
# test_ggml_weights test_ggml_compute transcribe streaming

streaming: test_streaming nemotron-asr.cpp

# C wrapper library for FFI (static and shared)
lib: libnemotron_asr.a libnemotron_asr.so

libnemotron_asr.a: $(GGML_SRCS:.cpp=.o) $(GGML_STREAM_SRCS:.cpp=.o) $(C_WRAPPER_SRCS:.cpp=.o)
	ar rcs $@ $^

libnemotron_asr.so: $(GGML_SRCS) $(GGML_STREAM_SRCS) $(C_WRAPPER_SRCS)
	$(CXX) $(CXXFLAGS) -fPIC -shared $^ $(LDFLAGS) -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

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
nemotron-asr.cpp: src/transcribe_stream.cpp $(GGML_SRCS) $(GGML_STREAM_SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

clean:
	rm -f test_ggml_weights test_ggml_compute precompute_encoder_ref transcribe test_streaming transcribe_stream test_python_ref test_preprocessor
	rm -f libnemotron_asr.a libnemotron_asr.so
	rm -f src/*.o

clean_bin:
	rm my_bin/*

test: test_ggml_weights test_ggml_compute
	./test_ggml_weights
	./test_ggml_compute

test_stream: test_streaming
	./test_streaming

test_ref: test_python_ref
	./test_python_ref
