#pragma once

#include <string>
#include <vector>

namespace nemo {

// SentencePiece Tokenizer for decoding token IDs to text
class Tokenizer {
public:
    static constexpr int BLANK_TOKEN = 1024;
    static constexpr size_t VOCAB_SIZE = 1025;

    // Load vocabulary from file
    // Format: one token per line, line number = token ID
    bool load(const std::string& vocab_path);

    // Decode single token ID to string
    std::string decode_token(int token_id) const;

    // Decode sequence of token IDs to text
    // Handles SentencePiece ▁ word start markers
    std::string decode(const std::vector<int>& tokens) const;

    // Get vocabulary size
    size_t size() const { return vocab_.size(); }

    // Check if token starts a new word (starts with ▁)
    bool is_word_start(int token_id) const;

private:
    std::vector<std::string> vocab_;
};

}  // namespace nemo
