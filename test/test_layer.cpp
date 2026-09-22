#include "layer.h"

#include <cassert>
#include <cmath>
#include <vector>

namespace {

constexpr float epsilon = 1e-6f;

void expect_values(const Tensor& tensor, const std::vector<float>& expected) {
    assert(tensor.size() == expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
        assert(std::fabs(tensor.data[index] - expected[index]) < epsilon);
    }
}

void test_relu() {
    const Tensor input({-2.0f, 0.0f, 3.5f}, {3});
    expect_values(relu(input), {0.0f, 0.0f, 3.5f});
}

void test_conv2d() {
    const Tensor input(
        {1.0f, 2.0f, 3.0f,
         4.0f, 5.0f, 6.0f,
         7.0f, 8.0f, 9.0f},
        {1, 3, 3});
    const Tensor weight({1.0f, 1.0f, 1.0f, 1.0f}, {1, 1, 2, 2});
    const Tensor bias({0.5f}, {1});

    const Tensor output = conv2d(input, weight, bias, 1, 0);
    assert((output.shape == std::vector<std::size_t>{1, 2, 2}));
    expect_values(output, {12.5f, 16.5f, 24.5f, 28.5f});
}

void test_max_pool2d() {
    const Tensor input(
        {1.0f, 2.0f, 3.0f, 4.0f,
         5.0f, 6.0f, 7.0f, 8.0f,
         9.0f, 10.0f, 11.0f, 12.0f,
         13.0f, 14.0f, 15.0f, 16.0f},
        {1, 4, 4});

    const Tensor output = max_pool2d(input);
    assert((output.shape == std::vector<std::size_t>{1, 2, 2}));
    expect_values(output, {6.0f, 8.0f, 14.0f, 16.0f});
}

void test_flatten() {
    const Tensor input({1.0f, 2.0f, 3.0f, 4.0f}, {1, 2, 2});
    const Tensor output = flatten(input);
    assert((output.shape == std::vector<std::size_t>{4}));
    expect_values(output, input.data);
}

void test_linear() {
    const Tensor input({1.0f, 2.0f}, {2});
    const Tensor weight({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    const Tensor bias({0.5f, -0.5f}, {2});

    const Tensor output = linear(input, weight, bias);
    expect_values(output, {5.5f, 10.5f});
}

}

int main() {
    test_relu();
    test_conv2d();
    test_max_pool2d();
    test_flatten();
    test_linear();
    return 0;
}
