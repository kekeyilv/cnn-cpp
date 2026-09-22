cmake_minimum_required(VERSION 3.16)

project(fashion_mnist_cpp LANGUAGES CXX)

add_library(cnn_layers src/layer.cpp)
target_compile_features(cnn_layers PUBLIC cxx_std_17)
target_include_directories(cnn_layers PUBLIC include)

add_executable(fashion_mnist src/main.cpp)
target_compile_features(fashion_mnist PRIVATE cxx_std_17)
target_link_libraries(fashion_mnist PRIVATE cnn_layers)

enable_testing()

add_executable(test_layer test/test_layer.cpp)
target_link_libraries(test_layer PRIVATE cnn_layers)

add_test(NAME layer_tests COMMAND test_layer)
