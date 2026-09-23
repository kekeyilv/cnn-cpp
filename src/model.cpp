#include "model.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>

#include "layer.h"

namespace {

template <typename T>
T read_value(std::istream& input) {
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) {
        throw std::runtime_error("unexpected end of model file");
    }
    return value;
}

void require_shape(const Tensor& tensor,
                   const std::vector<std::size_t>& expected) {
    if (tensor.shape != expected) {
        throw std::runtime_error("unexpected tensor shape in model file");
    }
}

}  // namespace

Tensor read_tensor(std::istream& input) {
    const std::uint32_t ndim = read_value<std::uint32_t>(input);
    std::vector<std::size_t> shape(ndim);
    std::size_t element_count = 1;

    for (std::size_t index = 0; index < ndim; ++index) {
        shape[index] = read_value<std::uint32_t>(input);
        element_count *= shape[index];
    }

    std::vector<float> data(element_count);
    input.read(reinterpret_cast<char*>(data.data()),
               static_cast<std::streamsize>(data.size() * sizeof(float)));
    if (!input) {
        throw std::runtime_error("unexpected end of tensor data");
    }

    return Tensor(std::move(data), std::move(shape));
}

void FashionMnistModel::load(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open model file: " + path);
    }

    std::array<char, 4> magic{};
    input.read(magic.data(), magic.size());
    if (!input || magic != std::array<char, 4>{'F', 'M', 'N', '1'}) {
        throw std::runtime_error("invalid model file magic");
    }

    Tensor conv1_weight = read_tensor(input);
    Tensor conv1_bias = read_tensor(input);
    Tensor conv2_weight = read_tensor(input);
    Tensor conv2_bias = read_tensor(input);
    Tensor fc1_weight = read_tensor(input);
    Tensor fc1_bias = read_tensor(input);
    Tensor fc2_weight = read_tensor(input);
    Tensor fc2_bias = read_tensor(input);

    require_shape(conv1_weight, {8, 1, 3, 3});
    require_shape(conv1_bias, {8});
    require_shape(conv2_weight, {16, 8, 3, 3});
    require_shape(conv2_bias, {16});
    require_shape(fc1_weight, {64, 16 * 7 * 7});
    require_shape(fc1_bias, {64});
    require_shape(fc2_weight, {10, 64});
    require_shape(fc2_bias, {10});

    conv1_weight_ = std::move(conv1_weight);
    conv1_bias_ = std::move(conv1_bias);
    conv2_weight_ = std::move(conv2_weight);
    conv2_bias_ = std::move(conv2_bias);
    fc1_weight_ = std::move(fc1_weight);
    fc1_bias_ = std::move(fc1_bias);
    fc2_weight_ = std::move(fc2_weight);
    fc2_bias_ = std::move(fc2_bias);
    loaded_ = true;
}

std::vector<float> FashionMnistModel::forward(const Tensor& input) const {
    if (!loaded_) {
        throw std::runtime_error("model is not loaded");
    }
    if (input.shape != std::vector<std::size_t>{1, 28, 28}) {
        throw std::runtime_error("expected input shape 1 x 28 x 28");
    }

    Tensor value = conv2d(input, conv1_weight_, conv1_bias_);
    value = relu(value);
    value = max_pool2d(value);
    value = conv2d(value, conv2_weight_, conv2_bias_);
    value = relu(value);
    value = max_pool2d(value);
    value = flatten(value);
    value = linear(value, fc1_weight_, fc1_bias_);
    value = relu(value);
    value = linear(value, fc2_weight_, fc2_bias_);
    return value.data;
}

int FashionMnistModel::predict(const Tensor& input) const {
    const std::vector<float> logits = forward(input);
    return static_cast<int>(std::distance(
        logits.begin(), std::max_element(logits.begin(), logits.end())));
}
