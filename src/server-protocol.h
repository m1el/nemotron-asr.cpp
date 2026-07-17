#ifndef NEMO_SERVER_PROTOCOL_H
#define NEMO_SERVER_PROTOCOL_H

// Wire protocol for the nemotron streaming ASR server.
//
// Framing: every message is  [u8 opcode][u32 stream_id][u32 payload_len][payload]
// with the two u32 fields little-endian. One byte-stream connection (TCP or Unix
// socket) may multiplex multiple stream_ids.
//
// Flow:
//   C -> STREAM_START {json cfg}        S -> STARTED {json id}
//   C -> PUSH id [pcm16le @16kHz mono]  S -> ACK {json queued_samples}
//   C -> SET_LANG id "ru-RU"            S -> LANG_SET {json id,lang,index}
//                                       S -> TEXT id "..."   (async, 0+ per stream)
//   C -> STREAM_END id                  S -> ENDED id "final text"
//   (any error)                         S -> ERROR id "message"
//
// Audio on the wire is raw signed 16-bit little-endian PCM, mono, 16 kHz.

#include <cstdint>

namespace nemo_proto {

// 9-byte fixed frame header, little-endian u32 fields. Sent as raw bytes.
static constexpr int HEADER_SIZE = 9;   // 1 (opcode) + 4 (stream_id) + 4 (payload_len)

enum Opcode : uint8_t {
    // Client -> Server
    OP_STREAM_START = 0x01,  // payload: JSON {"lang":"ru-RU","right_context":13} (optional)
    OP_PUSH         = 0x02,  // payload: int16le PCM @16kHz mono
    OP_STREAM_END   = 0x03,  // payload: none
    OP_SET_LANG     = 0x04,  // payload: UTF-8 language code

    // Server -> Client
    OP_STARTED      = 0x81,  // payload: JSON {"id":N}
    OP_ACK          = 0x82,  // payload: JSON {"queued_samples":N}
    OP_TEXT         = 0x83,  // payload: UTF-8 incremental transcript
    OP_ENDED        = 0x84,  // payload: UTF-8 final flushed text
    OP_LANG_SET     = 0x85,  // payload: JSON {"id":N,"lang":"ru-RU","index":M}
    OP_ERROR        = 0x8F,  // payload: UTF-8 message
};

// Encode/decode the 9-byte header. buf must have room for HEADER_SIZE bytes.
inline void write_header(uint8_t* buf, uint8_t opcode, uint32_t stream_id, uint32_t len) {
    buf[0] = opcode;
    buf[1] = (uint8_t)(stream_id      ); buf[2] = (uint8_t)(stream_id >>  8);
    buf[3] = (uint8_t)(stream_id >> 16); buf[4] = (uint8_t)(stream_id >> 24);
    buf[5] = (uint8_t)(len            ); buf[6] = (uint8_t)(len       >>  8);
    buf[7] = (uint8_t)(len       >> 16); buf[8] = (uint8_t)(len       >> 24);
}

inline void read_header(const uint8_t* buf, uint8_t& opcode, uint32_t& stream_id, uint32_t& len) {
    opcode = buf[0];
    stream_id = (uint32_t)buf[1] | ((uint32_t)buf[2] << 8) | ((uint32_t)buf[3] << 16) | ((uint32_t)buf[4] << 24);
    len       = (uint32_t)buf[5] | ((uint32_t)buf[6] << 8) | ((uint32_t)buf[7] << 16) | ((uint32_t)buf[8] << 24);
}

} // namespace nemo_proto

#endif // NEMO_SERVER_PROTOCOL_H
