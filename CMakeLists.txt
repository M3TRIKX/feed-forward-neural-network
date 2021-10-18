cmake_minimum_required(VERSION 3.19)
project(FeedForwardNeuralNet)

# Include Google tests
add_subdirectory(include/Google_tests)

set(CMAKE_CXX_STANDARD 20)

add_executable(FeedForwardNeuralNet src/main.cpp src/activation_functions/sigmoid.h src/csv/csv_reader.cpp src/csv/csv_reader.h src/data_structures/matrix.cpp src/data_structures/matrix.h src/activation_functions/template.h src/activation_functions/fast_sigmoid.h)
