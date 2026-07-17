// Streaming ASR server for nemotron-asr.cpp.
//
// Loads the model weights once and serves many concurrent streams over a TCP or Unix
// domain socket, using the length-prefixed binary protocol in server-protocol.h.
//
// Threading model (see the plan): per-connection reader threads only move bytes and
// enqueue events; a single worker thread owns ALL ggml/backend state (create, compute,
// destroy) and is the only caller into the streaming API. This serializes the single
// ggml backend and keeps CUDA on one thread. Backpressure is a global queued-bytes
// budget the readers block on.
//
// Usage: nemotron-server <model.gguf> [--tcp host:port | --unix /path/sock]
//                        [--cpu|--cuda] [--right-context N]

#include "nemo-ggml.h"
#include "nemo-stream.h"
#include "server-protocol.h"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

using namespace nemo_proto;

// ----------------------------------------------------------------------------
// Config
// ----------------------------------------------------------------------------
static constexpr size_t MEM_BUDGET_BYTES = 64 * 1024 * 1024;  // max queued PCM in flight
static constexpr size_t MAX_SEGMENT_SAMPLES = 8000;           // ~0.5s: reader splits PUSH
                                                              // into segments for fairness
static int   g_default_right_context = 0;

// ----------------------------------------------------------------------------
// Connection: shared between the owning reader thread and the worker. The socket
// closes when the last shared_ptr drops. All sends are whole-frame under write_mtx.
// ----------------------------------------------------------------------------
struct Connection {
    int fd;
    std::mutex write_mtx;
    std::atomic<bool> closed{false};

    explicit Connection(int fd_) : fd(fd_) {}
    ~Connection() { if (fd >= 0) ::close(fd); }
};

static bool send_all(int fd, const uint8_t* data, size_t n) {
    size_t off = 0;
    while (off < n) {
        ssize_t k = ::send(fd, data + off, n - off, MSG_NOSIGNAL);
        if (k <= 0) return false;
        off += (size_t)k;
    }
    return true;
}

// Send one framed message. Thread-safe; drops silently if the connection is closed.
static void send_frame(const std::shared_ptr<Connection>& conn, uint8_t opcode,
                       uint32_t stream_id, const void* payload, uint32_t len) {
    if (!conn || conn->closed.load()) return;
    uint8_t hdr[HEADER_SIZE];
    write_header(hdr, opcode, stream_id, len);
    std::lock_guard<std::mutex> lk(conn->write_mtx);
    if (conn->closed.load()) return;
    if (!send_all(conn->fd, hdr, HEADER_SIZE) ||
        (len > 0 && !send_all(conn->fd, (const uint8_t*)payload, len))) {
        conn->closed.store(true);
    }
}

static void send_text_frame(const std::shared_ptr<Connection>& conn, uint8_t opcode,
                            uint32_t stream_id, const std::string& s) {
    send_frame(conn, opcode, stream_id, s.data(), (uint32_t)s.size());
}

// ----------------------------------------------------------------------------
// Events: readers produce, the single worker consumes (global FIFO preserves each
// session's order; different sessions interleave fairly).
// ----------------------------------------------------------------------------
enum class EvType { CREATE, DATA, SET_LANG, END, CLOSE };

struct Event {
    EvType type;
    uint32_t stream_id;                  // server-assigned, monotonic == session token
    std::shared_ptr<Connection> conn;
    std::vector<int16_t> pcm;            // DATA
    std::string text;                    // SET_LANG (lang code)
    int right_context = 0;               // CREATE
    std::string lang;                    // CREATE (optional initial language)
};

static std::mutex               g_queue_mtx;
static std::condition_variable  g_queue_cv;
static std::deque<Event>        g_queue;
static bool                     g_stop = false;

// Global backpressure: total queued PCM bytes across all sessions.
static std::mutex               g_mem_mtx;
static std::condition_variable  g_mem_cv;
static size_t                   g_queued_bytes = 0;

static void mem_reserve(size_t bytes) {
    std::unique_lock<std::mutex> lk(g_mem_mtx);
    g_mem_cv.wait(lk, [&] { return g_stop || g_queued_bytes + bytes <= MEM_BUDGET_BYTES
                                   || g_queued_bytes == 0; });
    g_queued_bytes += bytes;
}
static void mem_release(size_t bytes) {
    std::lock_guard<std::mutex> lk(g_mem_mtx);
    g_queued_bytes -= bytes;
    g_mem_cv.notify_all();
}

static void enqueue(Event ev) {
    std::lock_guard<std::mutex> lk(g_queue_mtx);
    g_queue.push_back(std::move(ev));
    g_queue_cv.notify_one();
}

// ----------------------------------------------------------------------------
// Session: worker-exclusive. Never touched by reader threads.
// ----------------------------------------------------------------------------
struct Session {
    nemo_stream_context* sctx = nullptr;
    std::shared_ptr<Connection> conn;
    uint32_t stream_id = 0;
};

// ----------------------------------------------------------------------------
// Minimal JSON helpers (avoid a dependency for tiny control payloads)
// ----------------------------------------------------------------------------
static std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else o += c;
    }
    return o;
}

// Extract a string field: "key":"value"
static bool json_get_str(const std::string& j, const char* key, std::string& out) {
    std::string pat = std::string("\"") + key + "\"";
    size_t p = j.find(pat);
    if (p == std::string::npos) return false;
    p = j.find(':', p + pat.size());
    if (p == std::string::npos) return false;
    p = j.find('"', p);
    if (p == std::string::npos) return false;
    size_t q = j.find('"', p + 1);
    if (q == std::string::npos) return false;
    out = j.substr(p + 1, q - p - 1);
    return true;
}

// Extract an integer field: "key":N
static bool json_get_int(const std::string& j, const char* key, int& out) {
    std::string pat = std::string("\"") + key + "\"";
    size_t p = j.find(pat);
    if (p == std::string::npos) return false;
    p = j.find(':', p + pat.size());
    if (p == std::string::npos) return false;
    p++;
    while (p < j.size() && (j[p] == ' ' || j[p] == '\t')) p++;
    bool neg = (p < j.size() && j[p] == '-');
    if (neg) p++;
    if (p >= j.size() || j[p] < '0' || j[p] > '9') return false;
    int v = 0;
    while (p < j.size() && j[p] >= '0' && j[p] <= '9') { v = v * 10 + (j[p] - '0'); p++; }
    out = neg ? -v : v;
    return true;
}

// ----------------------------------------------------------------------------
// Worker: the sole owner of ggml/backend state and the streaming API.
// ----------------------------------------------------------------------------
static void worker_loop(nemo_context* model) {
    std::unordered_map<uint32_t, std::unique_ptr<Session>> sessions;

    auto free_session = [&](uint32_t id) {
        auto it = sessions.find(id);
        if (it == sessions.end()) return;
        if (it->second->sctx) nemo_stream_free(it->second->sctx);
        sessions.erase(it);
    };

    for (;;) {
        Event ev;
        {
            std::unique_lock<std::mutex> lk(g_queue_mtx);
            g_queue_cv.wait(lk, [] { return g_stop || !g_queue.empty(); });
            if (g_stop && g_queue.empty()) break;
            ev = std::move(g_queue.front());
            g_queue.pop_front();
        }

        switch (ev.type) {
        case EvType::CREATE: {
            nemo_cache_config cfg = nemo_cache_config::default_config();
            cfg.att_right_context = ev.right_context;
            auto s = std::make_unique<Session>();
            s->conn = ev.conn;
            s->stream_id = ev.stream_id;
            s->sctx = nemo_stream_init(model, &cfg);
            if (!s->sctx) {
                send_text_frame(ev.conn, OP_ERROR, ev.stream_id, "failed to init stream");
                break;
            }
            if (!ev.lang.empty() && ev.lang != "auto") {
                nemo_stream_set_language(s->sctx, ev.lang.c_str());
            }
            sessions[ev.stream_id] = std::move(s);
            break;
        }
        case EvType::DATA: {
            mem_release(ev.pcm.size() * sizeof(int16_t));
            auto it = sessions.find(ev.stream_id);
            if (it == sessions.end()) break;  // stale (session already closed)
            std::string text = nemo_stream_process_incremental(
                it->second->sctx, ev.pcm.data(), (int)ev.pcm.size());
            if (!text.empty())
                send_text_frame(it->second->conn, OP_TEXT, ev.stream_id, text);
            break;
        }
        case EvType::SET_LANG: {
            auto it = sessions.find(ev.stream_id);
            if (it == sessions.end()) break;
            if (nemo_stream_set_language(it->second->sctx, ev.text.c_str())) {
                int idx = it->second->sctx->prompt_index;
                char buf[128];
                int n = snprintf(buf, sizeof(buf), "{\"id\":%u,\"lang\":\"%s\",\"index\":%d}",
                                 ev.stream_id, json_escape(ev.text).c_str(), idx);
                send_frame(it->second->conn, OP_LANG_SET, ev.stream_id, buf, (uint32_t)n);
            } else {
                send_text_frame(it->second->conn, OP_ERROR, ev.stream_id,
                                "unknown or unsupported language: " + ev.text);
            }
            break;
        }
        case EvType::END: {
            auto it = sessions.find(ev.stream_id);
            if (it == sessions.end()) break;
            std::string final_text = nemo_stream_finalize(it->second->sctx);
            send_text_frame(it->second->conn, OP_ENDED, ev.stream_id, final_text);
            free_session(ev.stream_id);
            break;
        }
        case EvType::CLOSE:
            free_session(ev.stream_id);
            break;
        }
    }

    for (auto& kv : sessions)
        if (kv.second->sctx) nemo_stream_free(kv.second->sctx);
}

// ----------------------------------------------------------------------------
// Reader: one per connection. Moves bytes only; never touches ggml state.
// ----------------------------------------------------------------------------
static std::atomic<uint32_t> g_next_stream_id{1};

static bool recv_full(int fd, uint8_t* buf, size_t n) {
    size_t off = 0;
    while (off < n) {
        ssize_t k = ::recv(fd, buf + off, n - off, 0);
        if (k <= 0) return false;
        off += (size_t)k;
    }
    return true;
}

static void reader_loop(int fd) {
    auto conn = std::make_shared<Connection>(fd);
    std::vector<uint32_t> my_streams;  // for cleanup on disconnect

    for (;;) {
        uint8_t hdr[HEADER_SIZE];
        if (!recv_full(fd, hdr, HEADER_SIZE)) break;
        uint8_t opcode; uint32_t stream_id, len;
        read_header(hdr, opcode, stream_id, len);

        std::vector<uint8_t> payload(len);
        if (len > 0 && !recv_full(fd, payload.data(), len)) break;

        switch (opcode) {
        case OP_STREAM_START: {
            uint32_t id = g_next_stream_id.fetch_add(1);
            std::string cfg(payload.begin(), payload.end());
            std::string lang; int rc = g_default_right_context;
            json_get_str(cfg, "lang", lang);
            json_get_int(cfg, "right_context", rc);

            char buf[64];
            int n = snprintf(buf, sizeof(buf), "{\"id\":%u}", id);
            send_frame(conn, OP_STARTED, id, buf, (uint32_t)n);

            Event ev; ev.type = EvType::CREATE; ev.stream_id = id; ev.conn = conn;
            ev.right_context = rc; ev.lang = lang;
            enqueue(std::move(ev));
            my_streams.push_back(id);
            break;
        }
        case OP_PUSH: {
            const int16_t* samples = (const int16_t*)payload.data();
            size_t total = len / sizeof(int16_t);
            // Split large pushes into bounded segments so the worker interleaves streams.
            for (size_t off = 0; off < total; off += MAX_SEGMENT_SAMPLES) {
                size_t seg = std::min(MAX_SEGMENT_SAMPLES, total - off);
                mem_reserve(seg * sizeof(int16_t));
                Event ev; ev.type = EvType::DATA; ev.stream_id = stream_id; ev.conn = conn;
                ev.pcm.assign(samples + off, samples + off + seg);
                enqueue(std::move(ev));
            }
            char buf[64];
            int n = snprintf(buf, sizeof(buf), "{\"queued_samples\":%zu}", total);
            send_frame(conn, OP_ACK, stream_id, buf, (uint32_t)n);
            break;
        }
        case OP_SET_LANG: {
            Event ev; ev.type = EvType::SET_LANG; ev.stream_id = stream_id; ev.conn = conn;
            ev.text.assign(payload.begin(), payload.end());
            enqueue(std::move(ev));
            break;
        }
        case OP_STREAM_END: {
            Event ev; ev.type = EvType::END; ev.stream_id = stream_id; ev.conn = conn;
            enqueue(std::move(ev));
            break;
        }
        default:
            send_text_frame(conn, OP_ERROR, stream_id, "unknown opcode");
            break;
        }
    }

    // Disconnect: tell the worker to reclaim every session this connection opened.
    conn->closed.store(true);
    for (uint32_t id : my_streams) {
        Event ev; ev.type = EvType::CLOSE; ev.stream_id = id; ev.conn = conn;
        enqueue(std::move(ev));
    }
}

// ----------------------------------------------------------------------------
// Sockets
// ----------------------------------------------------------------------------
static int make_tcp_server(const char* host, int port) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    addr.sin_addr.s_addr = (host && *host) ? inet_addr(host) : INADDR_ANY;
    if (::bind(fd, (sockaddr*)&addr, sizeof(addr)) < 0) { ::close(fd); return -1; }
    if (::listen(fd, 64) < 0) { ::close(fd); return -1; }
    return fd;
}

static int make_unix_server(const char* path) {
    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    ::unlink(path);  // remove stale socket
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
    if (::bind(fd, (sockaddr*)&addr, sizeof(addr)) < 0) { ::close(fd); return -1; }
    if (::listen(fd, 64) < 0) { ::close(fd); return -1; }
    return fd;
}

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s <model.gguf> [--tcp host:port | --unix /path/sock] "
        "[--cpu|--cuda] [--right-context N]\n"
        "  --tcp host:port    listen on TCP (host empty = all interfaces), e.g. --tcp :8300\n"
        "  --unix /path/sock  listen on a Unix domain socket\n"
        "  --right-context N  default latency mode (0,1,6,13) if client omits it\n",
        prog);
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 1; }
    const char* model_path = argv[1];
    nemo_backend_type backend = NEMO_BACKEND_AUTO;

    std::string tcp_arg, unix_arg;
    for (int i = 2; i < argc; i++) {
        if      (!strcmp(argv[i], "--cpu"))  backend = NEMO_BACKEND_CPU;
        else if (!strcmp(argv[i], "--cuda")) backend = NEMO_BACKEND_CUDA;
        else if (!strcmp(argv[i], "--tcp")  && i + 1 < argc) tcp_arg = argv[++i];
        else if (!strcmp(argv[i], "--unix") && i + 1 < argc) unix_arg = argv[++i];
        else if (!strcmp(argv[i], "--right-context") && i + 1 < argc) g_default_right_context = atoi(argv[++i]);
        else { usage(argv[0]); return 1; }
    }
    if (tcp_arg.empty() && unix_arg.empty()) tcp_arg = ":8300";  // default

    fprintf(stderr, "Loading model %s ...\n", model_path);
    nemo_context* model = nemo_init_with_backend(model_path, backend);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }
    fprintf(stderr, "Model loaded (backend: %s)\n", nemo_get_backend_name(model));

    int listen_fd;
    if (!unix_arg.empty()) {
        listen_fd = make_unix_server(unix_arg.c_str());
        fprintf(stderr, "Listening on unix:%s\n", unix_arg.c_str());
    } else {
        std::string host = tcp_arg.substr(0, tcp_arg.find(':'));
        int port = atoi(tcp_arg.substr(tcp_arg.find(':') + 1).c_str());
        listen_fd = make_tcp_server(host.c_str(), port);
        fprintf(stderr, "Listening on tcp:%s (port %d)\n", host.empty() ? "*" : host.c_str(), port);
    }
    if (listen_fd < 0) { fprintf(stderr, "Failed to bind socket\n"); nemo_free(model); return 1; }

    std::thread worker(worker_loop, model);

    for (;;) {
        int cfd = ::accept(listen_fd, nullptr, nullptr);
        if (cfd < 0) continue;
        int one = 1;
        setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));  // no-op on AF_UNIX
        std::thread(reader_loop, cfd).detach();
    }

    // Not reached in v1 (no graceful shutdown signal); worker joins on g_stop.
    { std::lock_guard<std::mutex> lk(g_queue_mtx); g_stop = true; g_queue_cv.notify_all(); }
    worker.join();
    nemo_free(model);
    return 0;
}
