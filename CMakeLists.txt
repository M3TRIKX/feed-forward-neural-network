cmake_minimum_required(VERSION 3.10)
project(FeedForwardNeuralNet)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "-pipe -O3 -fno-math-errno -fexcess-precision=fast -fno-rounding-math -fno-signaling-nans -funroll-loops -mavx2 -mfma -march=native -openmp:experimental")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    message("OPENMP FOUND")
else()
    message("OPENMP NOT FOUND")
endif()

add_executable(FeedForwardNeuralNet src/main.cpp src/activation_functions/sigmoid.h src/csv/csv_reader.h src/data_structures/matrix.h src/activation_functions/template.h src/activation_functions/fast_sigmoid.h src/activation_functions/relu.h src/csv/csv_writer.h src/statistics/accuracy.h src/statistics/crossentropy.h src/statistics/stats_printer.h src/statistics/weights_info.h src/network/config.cpp src/network/config.h src/network/network.cpp src/network/network.h src/activation_functions/functions_enum.h src/activation_functions/softmax.h src/data_manager/data_manager.cpp src/data_manager/data_manager.h src/optimizers/sgd.h src/optimizers/adam.h src/optimizers/optimizer_template.h src/schedulers/lr_sheduler.cpp src/utils/util_functions.cpp src/utils/config_tester.h src/utils/util_functions.h src/utils/config_tester.cpp)
