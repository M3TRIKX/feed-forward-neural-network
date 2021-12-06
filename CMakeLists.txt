cmake_minimum_required(VERSION 3.10)
project(FeedForwardNeuralNet)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "-pipe -Ofast -fno-math-errno -fexcess-precision=fast -fno-rounding-math -fno-signaling-nans -funroll-loops -mavx2 -mfma -march=native -openmp:experimental")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    message("OPENMP FOUND")
else()
    message("OPENMP NOT FOUND")
endif()

add_executable(FeedForwardNeuralNet src/main.cpp src/activation_functions/sigmoid.hpp src/csv/csv_reader.hpp src/data_structures/matrix.hpp src/activation_functions/template.hpp src/activation_functions/fast_sigmoid.hpp src/activation_functions/relu.hpp src/csv/csv_writer.hpp src/statistics/accuracy.hpp src/statistics/crossentropy.hpp src/statistics/stats.hpp src/statistics/weights_info.hpp src/network/config.cpp src/network/config.hpp src/network/network.cpp src/network/network.hpp src/activation_functions/functions_enum.hpp src/activation_functions/softmax.hpp src/data_manager/data_manager.cpp src/data_manager/data_manager.hpp src/optimizers/sgd.hpp src/optimizers/adam.hpp src/optimizers/optimizer_template.hpp src/schedulers/lr_sheduler.cpp src/utils/util_functions.cpp src/utils/config_tester.hpp src/utils/util_functions.hpp src/utils/config_tester.cpp)
