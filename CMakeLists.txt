cmake_minimum_required(VERSION 3.19)
project(FeedForwardNeuralNet)

# Include Google tests
add_subdirectory(include/Google_tests)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Wall -Werror")
#set(CMAKE_CXX_FLAGS -Ofast)

add_executable(FeedForwardNeuralNet src/main.cpp src/activation_functions/sigmoid.h src/csv/csv_reader.h src/data_structures/matrix.h src/activation_functions/template.h src/activation_functions/fast_sigmoid.h src/activation_functions/relu.h src/csv/csv_writer.h src/statistics/accuracy.h src/statistics/crossentropy.h src/statistics/argmax.h src/statistics/stats_printer.h src/statistics/weights_info.h)
