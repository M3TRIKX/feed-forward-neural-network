cmake_minimum_required(VERSION 3.19)
project(FeedForwardNeuralNet)

# Include Google tests
add_subdirectory(include/Google_tests)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Wall -Werror -pthread")
#set(CMAKE_CXX_FLAGS -Ofast)

set(THREADS_PREFER_PTHREAD_FLAG ON)

add_executable(FeedForwardNeuralNet src/main.cpp src/activation_functions/sigmoid.cpp src/activation_functions/sigmoid.h src/csv/csv_reader.cpp src/csv/csv_reader.h src/data_structures/matrix.cpp src/data_structures/matrix.h)
