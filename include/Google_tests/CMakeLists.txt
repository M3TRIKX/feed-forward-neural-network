project(Google_tests)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(lib)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

add_executable(Google_Tests_run ../../tests/basics_tests.cpp ../../tests/matrix_tests.cpp)
target_link_libraries(Google_Tests_run gtest gtest_main)
