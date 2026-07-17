#!/usr/bin/env python3
"""Reference client for the nemotron streaming ASR server.

Streams a raw PCM16 mono 16kHz file to the server and prints the transcript.

Usage:
    uv run scripts/stream_client.py audio.pcm [--tcp host:port | --unix /path/sock]
                                    [--lang ru-RU] [--right-context 13]
                                    [--chunk-ms 200] [--realtime]
"""
import argparse
import socket
import struct
import sys
import threading
import time

HEADER = struct.Struct("<BII")  # opcode, stream_id, payload_len

OP_STREAM_START = 0x01
OP_PUSH = 0x02
OP_STREAM_END = 0x03
OP_SET_LANG = 0x04
OP_STARTED = 0x81
OP_ACK = 0x82
OP_TEXT = 0x83
OP_ENDED = 0x84
OP_LANG_SET = 0x85
OP_ERROR = 0x8F

NAMES = {
    OP_STARTED: "STARTED", OP_ACK: "ACK", OP_TEXT: "TEXT",
    OP_ENDED: "ENDED", OP_LANG_SET: "LANG_SET", OP_ERROR: "ERROR",
}


def send_frame(sock, opcode, stream_id, payload=b""):
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    sock.sendall(HEADER.pack(opcode, stream_id, len(payload)) + payload)


def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def reader(sock, state):
    """Print server frames until ENDED/ERROR."""
    while True:
        hdr = recv_exact(sock, HEADER.size)
        if hdr is None:
            break
        opcode, stream_id, length = HEADER.unpack(hdr)
        payload = recv_exact(sock, length) if length else b""
        if payload is None:
            break
        if opcode == OP_STARTED:
            state["id"] = int(payload.decode().split(":")[1].strip("{}"))
            state["started"].set()
        elif opcode == OP_TEXT:
            sys.stdout.write(payload.decode("utf-8"))
            sys.stdout.flush()
        elif opcode == OP_ENDED:
            tail = payload.decode("utf-8")
            if tail:
                sys.stdout.write(tail)
            sys.stdout.write("\n")
            sys.stdout.flush()
            state["done"].set()
            break
        elif opcode == OP_LANG_SET:
            sys.stderr.write(f"\n[lang set: {payload.decode()}]\n")
        elif opcode == OP_ERROR:
            sys.stderr.write(f"\n[server error: {payload.decode()}]\n")
            state["done"].set()
            break
        elif opcode == OP_ACK:
            pass  # backpressure signal; ignored by this simple client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pcm", help="raw int16le mono 16kHz PCM file")
    ap.add_argument("--tcp", default="127.0.0.1:8300")
    ap.add_argument("--unix", default=None)
    ap.add_argument("--lang", default=None)
    ap.add_argument("--right-context", type=int, default=13)
    ap.add_argument("--chunk-ms", type=int, default=200)
    ap.add_argument("--realtime", action="store_true",
                    help="pace pushes at wall-clock speed (simulate a live mic)")
    args = ap.parse_args()

    if args.unix:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(args.unix)
    else:
        host, port = args.tcp.rsplit(":", 1)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, int(port)))

    state = {"id": None, "started": threading.Event(), "done": threading.Event()}
    threading.Thread(target=reader, args=(sock, state), daemon=True).start()

    cfg = "{" + f'"right_context":{args.right_context}'
    if args.lang:
        cfg += f',"lang":"{args.lang}"'
    cfg += "}"
    send_frame(sock, OP_STREAM_START, 0, cfg)
    if not state["started"].wait(timeout=5):
        sys.stderr.write("no STARTED from server\n")
        return 1
    sid = state["id"]

    with open(args.pcm, "rb") as f:
        data = f.read()
    chunk_bytes = int(16000 * 2 * args.chunk_ms / 1000)
    for off in range(0, len(data), chunk_bytes):
        send_frame(sock, OP_PUSH, sid, data[off:off + chunk_bytes])
        if args.realtime:
            time.sleep(args.chunk_ms / 1000.0)

    send_frame(sock, OP_STREAM_END, sid)
    state["done"].wait(timeout=60)
    sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
