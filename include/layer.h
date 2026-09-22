#pragma once

#include "tensor.h"

#include <cstddef>

Tensor conv2d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    std::size_t stride = 1,
    std::size_t padding = 1);

Tensor relu(const Tensor& input);

Tensor max_pool2d(
    const Tensor& input,
    std::size_t kernel_size = 2,
    std::size_t stride = 2);

Tensor flatten(const Tensor& input);

Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias);
