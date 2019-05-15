cmake_minimum_required(VERSION 3.9)
project(MBLib VERSION 0.1
        DESCRIPTION "A deep learning library"
        LANGUAGES CXX)

# sets cmake as subdirectory for helper modules
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(TEST_TARGET testmain)

add_executable(${TEST_TARGET} ../main.cpp)
target_compile_features(${TEST_TARGET} PUBLIC cxx_std_17)
set_target_properties(${TEST_TARGET} PROPERTIES CXX_EXTENSIONS OFF)

