cmake_minimum_required(VERSION 3.9)
project(MBLib VERSION 0.1
        DESCRIPTION "A deep learning library"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(EXECUTABLE_TARGET Mlib_execute)

# Set cmake policy for ipo
cmake_policy(SET CMP0069 NEW)

# Set build type to release, if no type is specified
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# Set compiler warning flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wconversion -pedantic")
# Set debugging flags
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
# Set optimization for release (optional with -flto=thin for my clang++, because the check doesn't work
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")# -flto=thin")

#set(PYBIND11_CPP_STANDARD -std=c++1z)

# Set include directorys
include_directories(include)

# Update submodules (from https://cliutils.gitlab.io/modern-cmake)
find_package(Git QUIET)
if (GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    # Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    set(GIT_SUBMODULE OFF)
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

# Add submodule directorys
#add_subdirectory(extern/pybind11)
add_subdirectory(extern/spdlog)
add_subdirectory(extern/Catch2)
#include_directories(extern/spdlog)

#if(NOT TARGET spdlog)
#Stand-alone build
# find_package(spdlog REQUIRED)
#endif()

# Throw error, if submodules not initialized
if ((NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/Catch2/CMakeLists.txt") OR (NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/Eigen3/CMakeLists.txt") OR (NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/pybind11/CMakeLists.txt") OR (NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/spdlog/CMakeLists.txt"))
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif ()

find_package(MNIST PATHS extern/mnist)
if (NOT MNIST_FOUND)
    message(FATAL_ERROR "MNIST loader could not be found. It is available under https://github.com/wichtounet/mnist")
endif (NOT MNIST_FOUND)

include_directories(${MNIST_INCLUDE_DIR})

# Set FindEigen directory in Eigen3 submodule
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/extern/Eigen3/cmake")
# Set Catch2 cmake module directory in Catch2 submodule
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/extern/Catch2/contrib")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/extern/Catch2/CMake")
# Set Eigen include dir
set(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/extern/Eigen3")
# Find Eigen and Doxygen package
find_package(Eigen3 3.3 REQUIRED MODULE)
find_package(Doxygen)

# Set Doxygen to generate docs
set(DOXYGEN_GENERATE_HTML YES)

# Headers
set(HDRS
        include/ActivationFunction/ActivationFunction.h
        include/ActivationFunction/SigmoidFunction.h
        include/ActivationFunction/ReluFunction.h
        include/Layer/BaseLayer.h
        include/Layer/FullyConnectedLayer.h
        include/NeuralNetwork.h
        include/RandomInitialization/RandomInitialization.h
        include/LossFunction/LossFunction.h
        include/OptimizationFunction/OptimizationFunction.h
        include/RandomInitialization/SimpleRandomInitialization.h
        include/LossFunction/BinaryCrossEntropyLoss.h
        include/ActivationFunction/TanhFunction.h
        include/Layer/ConvolutionalLayer.h
        include/ActivationFunction/SoftmaxFunction.h
        include/ActivationFunction/IdentityFunction.h
        include/LossFunction/MultiCrossEntropyLoss.h
        include/RandomInitialization/XavierInitialization.h
        include/RandomInitialization/HetalInitialization.h
        include/RandomInitialization/DeterministicInitialization.h
        include/RandomInitialization/UniformXavierInitialization.h
        include/RandomInitialization/UniformHeInitialization.h
        include/ManageLoggers.h
        include/HelperFunctions.h)
# sources
set(SRCS
        src/ActivationFunction/SigmoidFunction.cpp
        src/ActivationFunction/ReluFunction.cpp
        src/Layer/FullyConnectedLayer.cpp
        src/RandomInitialization/SimpleRandomInitialization.cpp
        src/LossFunction/BinaryCrossEntropyLoss.cpp
        src/NeuralNetwork.cpp
        src/ActivationFunction/TanhFunction.cpp
        src/Layer/ConvolutionalLayer.cpp
        src/ActivationFunction/SoftmaxFunction.cpp
        src/ActivationFunction/IdentityFunction.cpp
        src/LossFunction/MultiCrossEntropyLoss.cpp
        src/RandomInitialization/XavierInitialization.cpp
        src/RandomInitialization/HetalInitialization.cpp
        src/RandomInitialization/DeterministicInitialization.cpp
        src/RandomInitialization/UniformXavierInitialization.cpp
        src/RandomInitialization/UniformHeInitialization.cpp
        src/ManageLoggers.cpp
        src/HelperFunctions.cpp)

add_library(Mlib STATIC ${SRCS} ${HDRS})

target_link_libraries(Mlib PUBLIC Eigen3::Eigen spdlog::spdlog)

add_executable(Mlib_test ../tests/test_layers.cpp tests/test-main.cpp)

target_link_libraries(Mlib_test PUBLIC Mlib Catch2::Catch2)

# Make CTest
include(CTest)
include(Catch)
catch_discover_tests(Mlib_test)
#include(ParseAndAddCatchTests)
#ParseAndAddCatchTests(Mlib_test)

add_executable(${EXECUTABLE_TARGET} ../main.cpp)

target_link_libraries(Mlib_execute PUBLIC Mlib)

target_compile_features(${EXECUTABLE_TARGET} PUBLIC cxx_std_17)

set_target_properties(${EXECUTABLE_TARGET} PROPERTIES CXX_EXTENSIONS OFF)

target_compile_definitions(${EXECUTABLE_TARGET} PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")

# Create logs directory
#file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/logs")

# Set CMake standard for Mlib
set_target_properties(Mlib PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        )

# Check for and enable Interprocedural optimization (flto flag, link optimization) (from https://cliutils.gitlab.io/modern-cmake/chapters/features/small.html)
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_result)
    if (ipo_result)
        message(STATUS "Interprocedural optimization flto activated by cmake: ${ipo_result}")
        set_target_properties(Mlib PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
        set_target_properties(Mlib_test PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
        set_target_properties(Mlib_execute PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
        #set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif ()
endif ()

# Print if no ipo optimizations are done
if (NOT ipo_result)
    message(STATUS "Interprocedural optimization flto activated by cmake: NO")
endif ()

# Generate Doxygen docs
if (DOXYGEN_FOUND)
    doxygen_add_docs(
            doxygen
            ${HDRS}
            COMMENT "Generate html pages"
    )
    message(STATUS "Generated Doxygen docs")
else (DOXYGEN_FOUND)
    message(STATUS "Skipped generating Doxygen docs")
endif (DOXYGEN_FOUND)