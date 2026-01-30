#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

namespace nemo {

// Tensor data type
enum class DType : uint32_t {
    F32 = 0,
    F16 = 1,
};

// Tensor metadata and data
struct Tensor {
    std::string name;
    std::vector<size_t> shape;
    DType dtype;
    std::vector<float> data;  // Always stored as f32 internally

    // Get total number of elements
    size_t numel() const {
        size_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }

    // Get number of dimensions
    size_t ndim() const { return shape.size(); }

    // Access helpers for common shapes
    float& at(size_t i) { return data[i]; }
    const float& at(size_t i) const { return data[i]; }

    float& at(size_t i, size_t j) {
        return data[i * shape[1] + j];
    }
    const float& at(size_t i, size_t j) const {
        return data[i * shape[1] + j];
    }

    float& at(size_t i, size_t j, size_t k) {
        return data[(i * shape[1] + j) * shape[2] + k];
    }
    const float& at(size_t i, size_t j, size_t k) const {
        return data[(i * shape[1] + j) * shape[2] + k];
    }

    float& at(size_t i, size_t j, size_t k, size_t l) {
        return data[((i * shape[1] + j) * shape[2] + k) * shape[3] + l];
    }
    const float& at(size_t i, size_t j, size_t k, size_t l) const {
        return data[((i * shape[1] + j) * shape[2] + k) * shape[3] + l];
    }

    // Print tensor info
    void print_info() const {
        printf("%s: [", name.c_str());
        for (size_t i = 0; i < shape.size(); i++) {
            printf("%zu%s", shape[i], i < shape.size() - 1 ? ", " : "");
        }
        printf("] (%s)\n", dtype == DType::F32 ? "f32" : "f16");
    }
};

// Model weights container
class ModelWeights {
public:
    ModelWeights() = default;

    // Load weights from binary file
    bool load(const std::string& path);

    // Get tensor by name (returns nullptr if not found)
    const Tensor* get(const std::string& name) const;

    // Get tensor by name (throws if not found)
    const Tensor& require(const std::string& name) const;

    // Check if tensor exists
    bool has(const std::string& name) const;

    // Get all tensor names
    std::vector<std::string> names() const;

    // Print all tensors info
    void print_info() const;

    // Get number of tensors
    size_t size() const { return tensors_.size(); }

    // Get total parameter count
    size_t total_params() const;

private:
    std::unordered_map<std::string, Tensor> tensors_;
};

}  // namespace nemo
