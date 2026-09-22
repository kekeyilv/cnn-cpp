#pragma once

#include <cstddef>
#include <utility>
#include <vector>

struct Tensor {
    std::vector<float> data;
    std::vector<std::size_t> shape;

    Tensor() = default;

    explicit Tensor(std::vector<std::size_t> dimensions)
        : data(element_count(dimensions)), shape(std::move(dimensions)) {}

    Tensor(std::vector<float> values, std::vector<std::size_t> dimensions)
        : data(std::move(values)), shape(std::move(dimensions)) {}

    std::size_t size() const {
        return data.size();
    }

private:
    static std::size_t element_count(const std::vector<std::size_t>& dimensions) {
        std::size_t count = 1;
        for (const std::size_t dimension : dimensions) {
            count *= dimension;
        }
        return count;
    }
};
