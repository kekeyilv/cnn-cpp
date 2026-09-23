#include "model.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr float epsilon = 1e-6f;

template <typename T>
void write_value(std::ostream& output, const T& value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_tensor(std::ostream& output, const Tensor& tensor) {
    const auto ndim = static_cast<std::uint32_t>(tensor.shape.size());
    write_value(output, ndim);
    for (const std::size_t dimension : tensor.shape) {
        const auto value = static_cast<std::uint32_t>(dimension);
        write_value(output, value);
    }
    output.write(
        reinterpret_cast<const char*>(tensor.data.data()),
        static_cast<std::streamsize>(tensor.data.size() * sizeof(float)));
}

void test_read_tensor() {
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    write_tensor(stream, Tensor({1.0f, -2.0f, 3.5f, 4.0f}, {2, 2}));
    stream.seekg(0);

    const Tensor tensor = read_tensor(stream);
    assert((tensor.shape == std::vector<std::size_t>{2, 2}));
    assert(tensor.data.size() == 4);
    assert(std::fabs(tensor.data[0] - 1.0f) < epsilon);
    assert(std::fabs(tensor.data[1] + 2.0f) < epsilon);
    assert(std::fabs(tensor.data[2] - 3.5f) < epsilon);
    assert(std::fabs(tensor.data[3] - 4.0f) < epsilon);
}

void write_test_model(const std::string& path) {
    std::ofstream output(path, std::ios::binary);
    output.write("FMN1", 4);
    write_tensor(output, Tensor(std::vector<float>(8 * 1 * 3 * 3), {8, 1, 3, 3}));
    write_tensor(output, Tensor(std::vector<float>(8), {8}));
    write_tensor(output, Tensor(std::vector<float>(16 * 8 * 3 * 3), {16, 8, 3, 3}));
    write_tensor(output, Tensor(std::vector<float>(16), {16}));
    write_tensor(output, Tensor(std::vector<float>(64 * 16 * 7 * 7), {64, 16 * 7 * 7}));
    write_tensor(output, Tensor(std::vector<float>(64), {64}));
    write_tensor(output, Tensor(std::vector<float>(10 * 64), {10, 64}));
    write_tensor(
        output,
        Tensor({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f}, {10}));
}

void test_model_prediction() {
    const std::string path = "test_model.bin";
    write_test_model(path);

    FashionMnistModel model;
    model.load(path);

    const Tensor input(std::vector<float>(28 * 28), {1, 28, 28});
    const std::vector<float> logits = model.forward(input);
    assert(logits.size() == 10);
    for (std::size_t index = 0; index < logits.size(); ++index) {
        assert(std::fabs(logits[index] - static_cast<float>(index)) < epsilon);
    }
    assert(model.predict(input) == 9);

    std::remove(path.c_str());
}

}

int main() {
    test_read_tensor();
    test_model_prediction();
    return 0;
}
