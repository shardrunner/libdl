cmake_minimum_required(VERSION 3.9)
project(MBLib VERSION 0.1
        DESCRIPTION "A deep learning library"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS "-march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

#set(PYBIND11_CPP_STANDARD -std=c++1z)
set(TEST_TARGET Mlib_execute)
include_directories(include)
include_directories(extern/)

#find_package(OpenMP REQUIRED)
#find_package(Doxygen)

#add_subdirectory(extern/pybind11)
add_subdirectory(extern/spdlog)
#include_directories(extern/spdlog)

#if(NOT TARGET spdlog)
# Stand-alone build
#find_package(spdlog REQUIRED)
#endif()


# Update submodules (from https://cliutils.gitlab.io/modern-cmake)
find_package(Git QUIET)
if (GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    # Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if (GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMOD_RESULT)
        #execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --remote --merge
        #        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        #        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if (NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif ()
    endif ()
endif ()

if ((NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/Catch2/CMakeLists.txt") OR (NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/Eigen3/CMakeLists.txt") OR (NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/pybind11/CMakeLists.txt") OR (NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/spdlog/CMakeLists.txt"))
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif ()

find_package(MNIST PATHS extern/mnist)
if (NOT MNIST_FOUND)
    message(FATAL_ERROR "MNIST loader could not be found. It is available under https://github.com/wichtounet/mnist")
endif (NOT MNIST_FOUND)

include_directories(${MNIST_INCLUDE_DIR})

#set(DOXYGEN_GENERATE_HTML YES)

# Added submodules (from https://cliutils.gitlab.io/modern-cmake)
#TODO Cmake folder with findeigen cmake_module_path
set(CMAKE_MODULE_PATH /cmake)
add_subdirectory(extern/Catch2)
#set(EIGEN3_INCLUDE_DIR extern/Eigen3)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
#find_package(Eigen3 3.3 REQUIRED MODULE)

# Interprocedural optimization (from https://cliutils.gitlab.io/modern-cmake)
#include(CheckIPOSupported)
#check_ipo_supported(RESULT result)
#if(result)
#    set_target_properties(foo PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
#endif()

set(HDRS
        include/ActivationFunction/ActivationFunction.h
        include/ActivationFunction/SigmoidFunction.h
        include/ActivationFunction/ReluFunction.h include/Layer/BaseLayer.h include/Layer/FullyConnectedLayer.h include/NeuralNetwork.h include/RandomInitialization/RandomInitialization.h include/LossFunction/LossFunction.h include/OptimizationFunction/OptimizationFunction.h include/RandomInitialization/SimpleRandomInitialization.h include/LossFunction/BinaryCrossEntropyLoss.h include/ActivationFunction/TanhFunction.h include/Layer/ConvolutionalLayer.h include/Layer/MaxPoolLayer.h include/ActivationFunction/SoftmaxFunction.h include/ActivationFunction/IdentityFunction.h include/LossFunction/MultiCrossEntropyLoss.h include/RandomInitialization/XavierInitialization.h include/RandomInitialization/HetalInitialization.h include/RandomInitialization/DeterministicInitialization.h include/RandomInitialization/UniformXavierInitialization.h include/RandomInitialization/UniformHeInitialization.h include/ManageLoggers.h include/HelperFunctions.h)
set(SRCS
        src/ActivationFunction/SigmoidFunction.cpp src/ActivationFunction/ReluFunction.cpp src/Layer/FullyConnectedLayer.cpp src/RandomInitialization/SimpleRandomInitialization.cpp src/LossFunction/BinaryCrossEntropyLoss.cpp src/NeuralNetwork.cpp src/ActivationFunction/TanhFunction.cpp src/Layer/ConvolutionalLayer.cpp src/Layer/MaxPoolLayer.cpp src/ActivationFunction/SoftmaxFunction.cpp src/ActivationFunction/IdentityFunction.cpp src/LossFunction/MultiCrossEntropyLoss.cpp src/RandomInitialization/XavierInitialization.cpp src/RandomInitialization/HetalInitialization.cpp src/RandomInitialization/DeterministicInitialization.cpp src/RandomInitialization/UniformXavierInitialization.cpp src/RandomInitialization/UniformHeInitialization.cpp src/ManageLoggers.cpp src/HelperFunctions.cpp)

add_library(Mlib ${SRCS} ${HDRS})

add_executable(Mlib_test ../tests/test_layers.cpp tests/test-main.cpp)

target_link_libraries(Mlib_test Mlib Catch2::Catch2)

target_link_libraries(Mlib Eigen3::Eigen spdlog::spdlog)

add_executable(${TEST_TARGET} ../main.cpp)

target_link_libraries(Mlib_execute PUBLIC Mlib)# OpenMP::OpenMP_CXX)

#if(OpenMP_CXX_FOUND)
#    target_link_libraries(Mlib_execute PUBLIC OpenMP::OpenMP_CXX)
#endif()

target_compile_features(${TEST_TARGET} PUBLIC cxx_std_17)
#target_compile_features(Mlib_test PUBLIC cxx_std_17)
set_target_properties(${TEST_TARGET} PROPERTIES CXX_EXTENSIONS OFF)

target_compile_definitions(${TEST_TARGET} PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")

# Create logs directory
#file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/logs")

set_target_properties(Mlib PROPERTIES
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