#include "layer.h"

#include <algorithm>
#include <cstddef>
#include <limits>

Tensor conv2d(const Tensor& input, const Tensor& weight, const Tensor& bias,
              const std::size_t stride, const std::size_t padding) {
    const std::size_t in_channels = input.shape[0];
    const std::size_t in_height = input.shape[1];
    const std::size_t in_width = input.shape[2];
    const std::size_t out_channels = weight.shape[0];
    const std::size_t kernel_height = weight.shape[2];
    const std::size_t kernel_width = weight.shape[3];
    const std::size_t out_height =
        (in_height + 2 * padding - kernel_height) / stride + 1;
    const std::size_t out_width =
        (in_width + 2 * padding - kernel_width) / stride + 1;

    Tensor output({out_channels, out_height, out_width});

    for (std::size_t out_channel = 0; out_channel < out_channels;
         ++out_channel) {
        for (std::size_t out_y = 0; out_y < out_height; ++out_y) {
            for (std::size_t out_x = 0; out_x < out_width; ++out_x) {
                float sum = bias.data[out_channel];

                for (std::size_t in_channel = 0; in_channel < in_channels;
                     ++in_channel) {
                    for (std::size_t kernel_y = 0; kernel_y < kernel_height;
                         ++kernel_y) {
                        const auto in_y = static_cast<std::ptrdiff_t>(
                                              out_y * stride + kernel_y) -
                                          static_cast<std::ptrdiff_t>(padding);
                        if (in_y < 0 ||
                            in_y >= static_cast<std::ptrdiff_t>(in_height)) {
                            continue;
                        }

                        for (std::size_t kernel_x = 0; kernel_x < kernel_width;
                             ++kernel_x) {
                            const auto in_x =
                                static_cast<std::ptrdiff_t>(out_x * stride +
                                                            kernel_x) -
                                static_cast<std::ptrdiff_t>(padding);
                            if (in_x < 0 ||
                                in_x >= static_cast<std::ptrdiff_t>(in_width)) {
                                continue;
                            }

                            const std::size_t input_index =
                                (in_channel * in_height +
                                 static_cast<std::size_t>(in_y)) *
                                    in_width +
                                static_cast<std::size_t>(in_x);
                            const std::size_t weight_index =
                                ((out_channel * in_channels + in_channel) *
                                     kernel_height +
                                 kernel_y) *
                                    kernel_width +
                                kernel_x;
                            sum += input.data[input_index] *
                                   weight.data[weight_index];
                        }
                    }
                }

                output.data[(out_channel * out_height + out_y) * out_width +
                            out_x] = sum;
            }
        }
    }

    return output;
}

Tensor relu(const Tensor& input) {
    Tensor output = input;
    for (float& value : output.data) {
        value = std::max(0.0f, value);
    }
    return output;
}

Tensor max_pool2d(const Tensor& input, const std::size_t kernel_size,
                  const std::size_t stride) {
    const std::size_t channels = input.shape[0];
    const std::size_t in_height = input.shape[1];
    const std::size_t in_width = input.shape[2];
    const std::size_t out_height = (in_height - kernel_size) / stride + 1;
    const std::size_t out_width = (in_width - kernel_size) / stride + 1;

    Tensor output({channels, out_height, out_width});

    for (std::size_t channel = 0; channel < channels; ++channel) {
        for (std::size_t out_y = 0; out_y < out_height; ++out_y) {
            for (std::size_t out_x = 0; out_x < out_width; ++out_x) {
                float maximum = -std::numeric_limits<float>::infinity();

                for (std::size_t kernel_y = 0; kernel_y < kernel_size;
                     ++kernel_y) {
                    for (std::size_t kernel_x = 0; kernel_x < kernel_size;
                         ++kernel_x) {
                        const std::size_t in_y = out_y * stride + kernel_y;
                        const std::size_t in_x = out_x * stride + kernel_x;
                        const std::size_t input_index =
                            (channel * in_height + in_y) * in_width + in_x;
                        maximum = std::max(maximum, input.data[input_index]);
                    }
                }

                output
                    .data[(channel * out_height + out_y) * out_width + out_x] =
                    maximum;
            }
        }
    }

    return output;
}

Tensor flatten(const Tensor& input) {
    return Tensor(input.data, {input.size()});
}

Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    const std::size_t out_features = weight.shape[0];
    const std::size_t in_features = weight.shape[1];
    Tensor output({out_features});

    for (std::size_t out_feature = 0; out_feature < out_features;
         ++out_feature) {
        float sum = bias.data[out_feature];
        for (std::size_t in_feature = 0; in_feature < in_features;
             ++in_feature) {
            sum += weight.data[out_feature * in_features + in_feature] *
                   input.data[in_feature];
        }
        output.data[out_feature] = sum;
    }

    return output;
}
