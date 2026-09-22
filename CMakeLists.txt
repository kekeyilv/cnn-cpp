cmake_minimum_required(VERSION 3.16)

project(fashion_mnist_cpp LANGUAGES CXX)

add_executable(fashion_mnist src/main.cpp)

target_compile_features(fashion_mnist PRIVATE cxx_std_17)
target_include_directories(fashion_mnist PRIVATE include)
