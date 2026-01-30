#pragma once

#include "conformer_encoder.h"
#include "rnnt_decoder.h"
#include "rnnt_joint.h"
#include "ops.h"

#include <vector>

namespace nemo {

// Greedy RNNT Decoding
// For each encoder frame:
//   1. Get decoder output for last token
//   2. Compute joint logits
//   3. If argmax != blank: emit token, update decoder state
//   4. If argmax == blank: move to next encoder frame
class GreedyDecoder {
public:
    static constexpr int BLANK_TOKEN = 1024;
    static constexpr size_t MAX_SYMBOLS_PER_STEP = 10;  // Prevent infinite loops

    RNNTDecoder* decoder = nullptr;
    RNNTJoint* joint = nullptr;

    void init(RNNTDecoder* dec, RNNTJoint* jnt) {
        decoder = dec;
        joint = jnt;
    }

    // Greedy decode from encoder output
    // enc_out: [batch=1, time, 1024] encoder output
    // Returns: vector of token IDs (excluding blanks)
    std::vector<int> decode(const TensorF& enc_out);

private:
    TensorF dec_out_;
    TensorF enc_frame_;
    TensorF logits_;
};

// Full ASR pipeline: audio -> mel -> encoder -> greedy decode -> tokens
class ASRPipeline {
public:
    ConformerEncoder encoder;
    RNNTDecoder decoder;
    RNNTJoint joint;
    GreedyDecoder greedy;

    void load_weights(const ModelWeights& weights);

    // Transcribe mel features to token IDs
    // mel: [batch=1, time, 128] mel spectrogram
    // Returns: vector of token IDs
    std::vector<int> transcribe(const TensorF& mel);

private:
    TensorF enc_out_;
};

}  // namespace nemo
