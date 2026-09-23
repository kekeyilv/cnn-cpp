#pragma once

#include "tensor.h"

#include <istream>
#include <string>
#include <vector>

Tensor read_tensor(std::istream& input);

class FashionMnistModel {
public:
    void load(const std::string& path);
    std::vector<float> forward(const Tensor& input) const;
    int predict(const Tensor& input) const;

private:
    Tensor conv1_weight_;
    Tensor conv1_bias_;
    Tensor conv2_weight_;
    Tensor conv2_bias_;
    Tensor fc1_weight_;
    Tensor fc1_bias_;
    Tensor fc2_weight_;
    Tensor fc2_bias_;
    bool loaded_ = false;
};
