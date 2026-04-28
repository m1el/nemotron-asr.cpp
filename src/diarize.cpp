#include "diarize.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

#include "ggml.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

// ---------------------------------------------------------------------------
// KV helpers
// ---------------------------------------------------------------------------

static int  kv_u32(gguf_context * c, const char * k, int def) {
    int64_t i = gguf_find_key(c, k);
    return (i >= 0) ? (int)gguf_get_val_u32(c, i) : def;
}

static float kv_f32(gguf_context * c, const char * k, float def) {
    int64_t i = gguf_find_key(c, k);
    return (i >= 0) ? gguf_get_val_f32(c, i) : def;
}

static std::string kv_str(gguf_context * c, const char * k, const char * def) {
    int64_t i = gguf_find_key(c, k);
    return (i >= 0) ? std::string(gguf_get_val_str(c, i)) : std::string(def);
}

static void load_audio_hparams(gguf_context * c, const char * prefix,
                               diarize_audio_hparams & h) {
    auto K = [&](const char * suffix) { return std::string(prefix) + "." + suffix; };
    h.sample_rate   = kv_u32(c, K("sample_rate").c_str(), 16000);
    h.n_mels        = kv_u32(c, K("n_mels").c_str(), 80);
    h.n_fft         = kv_u32(c, K("n_fft").c_str(), 512);
    h.window_size   = kv_f32(c, K("window_size").c_str(), 0.025f);
    h.window_stride = kv_f32(c, K("window_stride").c_str(), 0.01f);
    h.dither        = kv_f32(c, K("dither").c_str(), 0.0f);
    h.normalize     = kv_str(c, K("normalize").c_str(), "None");
    h.window        = kv_str(c, K("window").c_str(), "hann");
}

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

static ggml_backend_t init_backend_for(diarize_backend choice) {
    auto try_cuda = []() -> ggml_backend_t {
#ifdef GGML_USE_CUDA
        if (ggml_backend_cuda_get_device_count() > 0) {
            return ggml_backend_cuda_init(0);
        }
#endif
        return nullptr;
    };
    auto try_metal = []() -> ggml_backend_t {
#ifdef GGML_USE_METAL
        return ggml_backend_metal_init();
#endif
        return nullptr;
    };
    switch (choice) {
        case diarize_backend::CPU:   return ggml_backend_cpu_init();
        case diarize_backend::CUDA:  return try_cuda();
        case diarize_backend::METAL: return try_metal();
        case diarize_backend::AUTO:
            if (auto * b = try_cuda())  return b;
            if (auto * b = try_metal()) return b;
            return ggml_backend_cpu_init();
    }
    return ggml_backend_cpu_init();
}

bool diarize_model_load(const std::string & path, diarize_model & model,
                        diarize_backend backend) {
    model.backend = init_backend_for(backend);
    if (!model.backend) {
        fprintf(stderr, "diarize_model_load: failed to init backend (choice=%d)\n",
                (int)backend);
        return false;
    }

    ggml_context * ctx_meta = nullptr;
    gguf_init_params params = { .no_alloc = true, .ctx = &ctx_meta };
    gguf_context * gguf_ctx = gguf_init_from_file(path.c_str(), params);
    if (!gguf_ctx) {
        fprintf(stderr, "diarize_model_load: failed to open '%s'\n", path.c_str());
        return false;
    }

    model.hparams.arch = kv_str(gguf_ctx, "general.architecture", "");
    model.hparams.name = kv_str(gguf_ctx, "general.name", "");
    if (model.hparams.arch != "nemo-diarize") {
        fprintf(stderr,
                "diarize_model_load: expected arch 'nemo-diarize', got '%s'\n",
                model.hparams.arch.c_str());
        gguf_free(gguf_ctx);
        ggml_free(ctx_meta);
        return false;
    }

    load_audio_hparams(gguf_ctx, "vad", model.hparams.vad.audio);
    load_audio_hparams(gguf_ctx, "spk", model.hparams.spk.audio);
    model.hparams.vad.n_classes     = kv_u32(gguf_ctx, "vad.n_classes",   2);
    model.hparams.spk.emb_dim       = kv_u32(gguf_ctx, "spk.emb_dim",     192);
    model.hparams.spk.attn_channels = kv_u32(gguf_ctx, "spk.attn_channels", 128);

    int64_t n_tensors = gguf_get_n_tensors(gguf_ctx);

    // Allocate ggml context big enough for tensor metadata.
    ggml_init_params ctx_params = {
        .mem_size   = ggml_tensor_overhead() * (n_tensors + 16),
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    model.ctx_w = ggml_init(ctx_params);
    if (!model.ctx_w) {
        fprintf(stderr, "diarize_model_load: ggml_init failed\n");
        gguf_free(gguf_ctx);
        ggml_free(ctx_meta);
        return false;
    }

    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        ggml_tensor * meta = ggml_get_tensor(ctx_meta, name);
        if (!meta) {
            fprintf(stderr, "diarize_model_load: meta tensor '%s' missing\n", name);
            continue;
        }
        ggml_tensor * t = ggml_dup_tensor(model.ctx_w, meta);
        ggml_set_name(t, name);
        model.tensors[name] = t;
    }

    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx_w, model.backend);
    if (!model.buffer) {
        fprintf(stderr, "diarize_model_load: backend buffer alloc failed\n");
        gguf_free(gguf_ctx);
        ggml_free(ctx_meta);
        return false;
    }

    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "diarize_model_load: reopen '%s' failed\n", path.c_str());
        gguf_free(gguf_ctx);
        ggml_free(ctx_meta);
        return false;
    }
    size_t data_off = gguf_get_data_offset(gguf_ctx);
    std::vector<char> buf;
    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        size_t off = gguf_get_tensor_offset(gguf_ctx, i);
        size_t sz  = gguf_get_tensor_size(gguf_ctx, i);
        auto it = model.tensors.find(name);
        if (it == model.tensors.end()) continue;
        buf.resize(sz);
        fseek(f, data_off + off, SEEK_SET);
        if (fread(buf.data(), 1, sz, f) != sz) {
            fprintf(stderr, "diarize_model_load: short read on '%s'\n", name);
            fclose(f);
            gguf_free(gguf_ctx);
            ggml_free(ctx_meta);
            return false;
        }
        ggml_backend_tensor_set(it->second, buf.data(), 0, sz);
    }
    fclose(f);
    gguf_free(gguf_ctx);
    ggml_free(ctx_meta);
    return true;
}

void diarize_model_free(diarize_model & m) {
    if (m.buffer)  { ggml_backend_buffer_free(m.buffer);  m.buffer  = nullptr; }
    if (m.ctx_w)   { ggml_free(m.ctx_w);                  m.ctx_w   = nullptr; }
    if (m.backend) { ggml_backend_free(m.backend);        m.backend = nullptr; }
    m.tensors.clear();
}

const ggml_tensor * diarize_model_get_tensor(const diarize_model & m, const std::string & name) {
    auto it = m.tensors.find(name);
    return (it == m.tensors.end()) ? nullptr : it->second;
}

void diarize_model_print_tensors(const diarize_model & m) {
    printf("diarize_model: %s (arch=%s)\n", m.hparams.name.c_str(), m.hparams.arch.c_str());
    printf("  vad: sr=%d n_mels=%d n_fft=%d normalize=%s\n",
           m.hparams.vad.audio.sample_rate, m.hparams.vad.audio.n_mels,
           m.hparams.vad.audio.n_fft, m.hparams.vad.audio.normalize.c_str());
    printf("  spk: sr=%d n_mels=%d n_fft=%d normalize=%s emb_dim=%d\n",
           m.hparams.spk.audio.sample_rate, m.hparams.spk.audio.n_mels,
           m.hparams.spk.audio.n_fft, m.hparams.spk.audio.normalize.c_str(),
           m.hparams.spk.emb_dim);
    printf("  %zu tensors:\n", m.tensors.size());
    size_t total_bytes = 0;
    for (const auto & [name, t] : m.tensors) {
        size_t bytes = ggml_nbytes(t);
        total_bytes += bytes;
        printf("    %-72s  ne=[%lld,%lld,%lld,%lld]  type=%-4s  %7zu B\n",
               name.c_str(),
               (long long)t->ne[0], (long long)t->ne[1],
               (long long)t->ne[2], (long long)t->ne[3],
               ggml_type_name(t->type), bytes);
    }
    printf("  total: %.1f MB\n", total_bytes / 1e6);
}
