// Sanity check: load diarize.gguf and print all tensors.
//
// Usage:  ./test_diarize_load weights/diarize-v0.1.f32.gguf
#include <cstdio>

#include "diarize.h"

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <diarize.gguf>\n", argv[0]);
        return 1;
    }
    diarize_model m;
    if (!diarize_model_load(argv[1], m)) return 2;
    diarize_model_print_tensors(m);
    diarize_model_free(m);
    return 0;
}
