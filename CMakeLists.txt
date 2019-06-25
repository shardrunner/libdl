cmake_minimum_required(VERSION 3.9)
project(MBLib VERSION 0.1
        DESCRIPTION "A deep learning library"
        LANGUAGES CXX)

#set(PYBIND11_CPP_STANDARD -std=c++1z)
set(TEST_TARGET testmain)
include_directories(include)

find_package(MNIST PATHS extern/mnist)
if(NOT MNIST_FOUND)
    message(FATAL_ERROR "MNIST loader could not be found. It is available under https://github.com/wichtounet/mnist")
endif(NOT MNIST_FOUND)

include_directories(${MNIST_INCLUDE_DIR})

find_package(Doxygen)

set(DOXYGEN_GENERATE_HTML YES)

# Added submodules (from https://cliutils.gitlab.io/modern-cmake)
add_subdirectory(extern/Catch2)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
#add_subdirectory(extern/pybind11)
add_subdirectory(extern/spdlog)

find_package(spdlog)

# Update submodules (from https://cliutils.gitlab.io/modern-cmake)
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    # Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

if((NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/Catch2/CMakeLists.txt") OR (NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/Eigen3/CMakeLists.txt") OR (NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/pybind11/CMakeLists.txt") OR (NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/spdlog/CMakeLists.txt"))
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

# Interprocedural optimization (from https://cliutils.gitlab.io/modern-cmake)
#include(CheckIPOSupported)
#check_ipo_supported(RESULT result)
#if(result)
#    set_target_properties(foo PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
#endif()

set(HDRS
#        include/NN.h
        include/ActivationFunction/ActivationFunction.h
        include/ActivationFunction/SigmoidFunction.h
        include/ActivationFunction/ReluFunction.h include/Layer/BaseLayer.h include/Layer/FullyConnectedLayer.h include/NeuralNetwork.h include/MTensor.h include/RandomInitialization/RandomInitialization.h include/LossFunction/LossFunction.h include/OptimizationFunction/OptimizationFunction.h include/RandomInitialization/SimpleRandomInitialization.h include/LossFunction/BinaryCrossEntropyLoss.h include/ActivationFunction/TanFunction.h include/Layer/ConvolutionalLayer.h include/Layer/MaxPoolLayer.h include/ActivationFunction/SoftmaxFunction.h include/LossFunction/CrossEntropyLoss.h include/ActivationFunction/IdentityFunction.h include/LossFunction/MultiClassLoss.h)
set(SRCS
#        src/NN.cpp
        src/SigmoidFunction.cpp src/ReluFunction.cpp src/FullyConnectedLayer.cpp src/RandomInitialization/SimpleRandomInitialization.cpp src/LossFunction/BinaryCrossEntropyLoss.cpp src/NeuralNetwork.cpp src/ActivationFunction/TanFunction.cpp src/Layer/ConvolutionalLayer.cpp src/Layer/MaxPoolLayer.cpp src/ActivationFunction/SoftmaxFunction.cpp src/LossFunction/CrossEntropyLoss.cpp src/ActivationFunction/IdentityFunction.cpp src/LossFunction/MultiClassLoss.cpp)

add_library(NN_lib ${SRCS} ${HDRS})

add_executable(test ../tests/test.cpp tests/test-main.cpp)

target_link_libraries(test NN_lib Catch2::Catch2)

target_link_libraries(NN_lib Eigen3::Eigen spdlog::spdlog)

add_executable(${TEST_TARGET} ../main.cpp)

target_link_libraries(testmain NN_lib)

target_compile_features(${TEST_TARGET} PUBLIC cxx_std_17)
target_compile_features(test PUBLIC cxx_std_17)
set_target_properties(${TEST_TARGET} PROPERTIES CXX_EXTENSIONS OFF)

target_compile_definitions(${TEST_TARGET} PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")

set_target_properties(NN_lib PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        )


if (DOXYGEN_FOUND)
    doxygen_add_docs(
        doxygen
        ${HDRS}
        COMMENT "Generate html pages"
    )
endif (DOXYGEN_FOUND)