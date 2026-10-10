#include <algorithm>
#include <array>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "model.h"

namespace {

struct Options {
    std::string model_path;
    std::string input_path;
    bool show_help = false;
    bool show_logits = false;
};

void print_usage(std::ostream& output) {
    output << "Usage: fashion_mnist --model <model.bin> --input <image.txt> [--logits]\n";
}

Options parse_options(const int argc, char* argv[]) {
    Options options;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help") {
            options.show_help = true;
        } else if (argument == "--logits") {
            options.show_logits = true;
        } else if (argument == "--model") {
            if (++index >= argc) {
                throw std::runtime_error("--model requires a path");
            }
            options.model_path = argv[index];
        } else if (argument == "--input") {
            if (++index >= argc) {
                throw std::runtime_error("--input requires a path");
            }
            options.input_path = argv[index];
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }

    if (!options.show_help &&
        (options.model_path.empty() || options.input_path.empty())) {
        throw std::runtime_error("both --model and --input are required");
    }
    return options;
}

std::vector<int> read_pixels(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open input file: " + path);
    }

    std::vector<int> pixels;
    int value = 0;
    while (input >> value) {
        pixels.push_back(value);
    }

    if (pixels.size() != 28 * 28) {
        throw std::runtime_error("input file must contain exactly 784 pixels");
    }
    if (!input.eof()) {
        throw std::runtime_error("input file contains an invalid pixel");
    }

    return pixels;
}

Tensor preprocess_pixels(const std::vector<int>& pixels) {
    if (pixels.size() != 28 * 28) {
        throw std::runtime_error("expected exactly 784 pixels");
    }

    std::vector<float> values;
    values.reserve(pixels.size());
    for (const int pixel : pixels) {
        if (pixel < 0 || pixel > 255) {
            throw std::runtime_error("pixel values must be between 0 and 255");
        }
        const float value = static_cast<float>(pixel) / 255.0f;
        values.push_back((value - 0.5f) / 0.5f);
    }

    return Tensor(std::move(values), {1, 28, 28});
}

Tensor load_fashion_mnist_input(const std::string& path) {
    return preprocess_pixels(read_pixels(path));
}

}

int main(const int argc, char* argv[]) {
    static const std::array<const char*, 10> class_names = {
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal",      "Shirt",   "Sneaker",  "Bag",   "Ankle boot",
    };

    try {
        const Options options = parse_options(argc, argv);
        if (options.show_help) {
            print_usage(std::cout);
            return 0;
        }

        const Tensor input = load_fashion_mnist_input(options.input_path);
        FashionMnistModel model;
        model.load(options.model_path);

        int prediction = 0;
        if (options.show_logits) {
            const std::vector<float> logits = model.forward(input);
            std::cout << "Logits:" << std::setprecision(9);
            for (const float logit : logits) {
                std::cout << ' ' << logit;
            }
            std::cout << '\n';
            prediction = static_cast<int>(std::distance(
                logits.begin(), std::max_element(logits.begin(), logits.end())));
        } else {
            prediction = model.predict(input);
        }

        std::cout << "Prediction: " << prediction << " ("
                  << class_names[prediction] << ")\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        print_usage(std::cerr);
        return 1;
    }
}
