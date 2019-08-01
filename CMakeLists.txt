cmake_minimum_required(VERSION 3.9)
project(MBLib VERSION 1.0
        DESCRIPTION "A deep learning library"
        LANGUAGES CXX)

# Set standard to C++ 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(EXECUTABLE_TARGET Mlib_execute)

# Set cmake policy for ipo
cmake_policy(SET CMP0069 NEW)

# Set build type to release, if no type is specified
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMake: ${CMAKE_VERSION}")

# Set -fPIC otherwise pybind won't work with spdlog
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

# Set include directorys
include_directories(include)

# Update submodules (from https://cliutils.gitlab.io/modern-cmake)
find_package(Git QUIET)
if (GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    # Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    #set(GIT_SUBMODULE OFF)
    if (GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMOD_RESULT)
        #execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --remote --merge
        #        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        #        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if (NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(WARNING "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif ()
    endif ()
endif ()

# Throw error, if submodules not initialized
if ((NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/Catch2/CMakeLists.txt") OR (NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/Eigen3/CMakeLists.txt") OR (NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/pybind11/CMakeLists.txt") OR (NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/spdlog/CMakeLists.txt"))
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif ()

# MNist image loader
find_package(MNIST PATHS extern/mnist)
if (NOT MNIST_FOUND)
    message(FATAL_ERROR "MNIST loader could not be found. It is available under https://github.com/wichtounet/mnist")
endif (NOT MNIST_FOUND)

include_directories(${MNIST_INCLUDE_DIR})

# Add submodule directorys
add_subdirectory(extern/pybind11)
set(PYBIND11_CPP_STANDARD -std=c++1z)
add_subdirectory(extern/spdlog)
add_subdirectory(extern/Catch2)

# Set FindEigen directory in Eigen3 submodule
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/extern/Eigen3/cmake")
# Set Catch2 cmake module directory in Catch2 submodule
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/extern/Catch2/contrib")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/extern/Catch2/CMake")
# Set Eigen include dir
set(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/extern/Eigen3")

# Find Eigen
find_package(Eigen3 3.3 REQUIRED MODULE)
if (EIGEN3_VERSION)
    message(STATUS "Eigen Version: ${EIGEN3_VERSION}")
else ()
    message(STATUS "Used installed Eigen version")
endif ()

# Headers
set(HDRS
        include/ActivationFunction/ActivationFunction.h
        include/ActivationFunction/SigmoidFunction.h
        include/ActivationFunction/ReluFunction.h
        include/Layer/Layer.h
        include/Layer/FullyConnectedLayer.h
        include/NeuralNetwork.h
        include/RandomInitialization/RandomInitialization.h
        include/LossFunction/LossFunction.h
        include/OptimizationFunction/OptimizationFunction.h
        include/RandomInitialization/SimpleRandomInitialization.h
        include/LossFunction/BinaryCrossEntropyLoss.h
        include/ActivationFunction/TanhFunction.h
        include/Layer/ConvolutionLayer.h
        include/ActivationFunction/SoftmaxFunction.h
        include/ActivationFunction/IdentityFunction.h
        include/LossFunction/MultiCrossEntropyLoss.h
        include/RandomInitialization/XavierInitialization.h
        include/RandomInitialization/HetalInitialization.h
        include/RandomInitialization/DeterministicInitialization.h
        include/RandomInitialization/UniformXavierInitialization.h
        include/RandomInitialization/UniformHeInitialization.h
        include/HelperFunctions.h
        include/OptimizationFunction/SimpleOptimizer.h
        include/OptimizationFunction/Adam.h)
# sources
set(SRCS
        src/ActivationFunction/SigmoidFunction.cpp
        src/ActivationFunction/ReluFunction.cpp
        src/Layer/FullyConnectedLayer.cpp
        src/RandomInitialization/SimpleRandomInitialization.cpp
        src/LossFunction/BinaryCrossEntropyLoss.cpp
        src/NeuralNetwork.cpp
        src/ActivationFunction/TanhFunction.cpp
        src/Layer/ConvolutionLayer.cpp
        src/ActivationFunction/SoftmaxFunction.cpp
        src/ActivationFunction/IdentityFunction.cpp
        src/LossFunction/MultiCrossEntropyLoss.cpp
        src/RandomInitialization/XavierInitialization.cpp
        src/RandomInitialization/HetalInitialization.cpp
        src/RandomInitialization/DeterministicInitialization.cpp
        src/RandomInitialization/UniformXavierInitialization.cpp
        src/RandomInitialization/UniformHeInitialization.cpp
        src/HelperFunctions.cpp
        src/OptimizationFunction/SimpleOptimizer.cpp
        src/OptimizationFunction/Adam.cpp)
#tests
set(Tests
        tests/TestConvLayer.cpp
        tests/test-main.cpp
        tests/TestActivations.cpp
        tests/TestFCLayer.cpp
        tests/TestLossFunctions.cpp
        tests/TestInitialization.cpp
        tests/TestNetwork.cpp tests/TestOptimizer.cpp)

add_library(Mlib STATIC ${SRCS} ${HDRS})

target_link_libraries(Mlib PUBLIC Eigen3::Eigen spdlog::spdlog)

add_executable(Mlib_test ${Tests})

target_link_libraries(Mlib_test PUBLIC Mlib Catch2::Catch2)

# Make CTest
include(CTest)
include(Catch)
catch_discover_tests(Mlib_test)

add_executable(${EXECUTABLE_TARGET} ../main.cpp)

target_link_libraries(${EXECUTABLE_TARGET} PUBLIC Mlib)

target_compile_definitions(${EXECUTABLE_TARGET} PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")

pybind11_add_module(MlibWrapper pybind/MlibWrapper.cpp)

target_link_libraries(MlibWrapper PUBLIC Eigen3::Eigen spdlog::spdlog Mlib)

# Check for OpenMP and enable it (used for Eigen + im2col)
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    message(STATUS "Enabled OpenMP")
    target_link_libraries(Mlib PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(${EXECUTABLE_TARGET} PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(Mlib_test PUBLIC OpenMP::OpenMP_CXX)
endif ()

# Create logs directory
#file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/logs")

set(IPO_RESULT NO)
# Check for and enable Interprocedural optimization (flto flag, link optimization) (from https://cliutils.gitlab.io/modern-cmake/chapters/features/small.html and https://stackoverflow.com/questions/35922966/lto-with-llvm-and-cmake)
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT IPO_RESULT)
    #if(TRUE)
    if (IPO_RESULT)
        set_target_properties(Mlib PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
        set_target_properties(Mlib_test PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
        set_target_properties(Mlib_execute PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
        #set_target_properties(MlibWrapper PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
        #set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif ()
endif ()

# Print wether ipo optimizations are done
message(STATUS "Interprocedural optimization flto activated by cmake: ${IPO_RESULT}")

# Set general compile and linking options
# Set compiler warning flags; fsized-deallocation are for pybind
set(BUILD_OPTIONS -Wall -Wextra -Wconversion -pedantic -fsized-deallocation)# -fvisibility=hidden)

# Set debug compile and linking options
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(BUILD_OPTIONS ${BUILD_OPTIONS} -O1 -g -fno-omit-frame-pointer -fno-optimize-sibling-calls)
endif ()

# Set release compile and linking options
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    # Set optimization for release (optional with -flto=thin for my clang++, because the check doesn't work; mtune and march optimize for used (native) processor, ffast-math sacrifices accuracy for speed (done in the back now)
    set(BUILD_OPTIONS ${BUILD_OPTIONS} -O3 -march=native -mtune=native)# -ffast-math")# -flto=thin)
endif ()

# Set sanitizer options if needed
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Set to true and uncomment desired sanitizer
    if (FALSE)
        message(STATUS "Using sanitizers")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize-recover=all")
        #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak -fsanitize=address")
        #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined -fsanitize=nullability -fsanitize=integer")
        #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread -fPIE -pie")
        #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=memory -fPIE -pie")
    endif ()
endif ()

# Apply defined build and linking options

# Fall back to CMAKE_CXX_Flags, because CMAKE on server is to old for target_link_options
    #Convert list of options to a  single string
#list(JOIN BUILD_OPTIONS " " BUILD_OPTIONS_STRING)
foreach(line IN LISTS ${BUILD_OPTIONS})
    set(${BUILD_OPTIONS_STRING} "${BUILD_OPTIONS_STRING} line")
endforeach()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BUILD_OPTIONS_STRING}")

#target_link_options(Mlib PUBLIC ${BUILD_OPTIONS})
#target_link_options(MlibWrapper PUBLIC ${BUILD_OPTIONS})

target_compile_options(MlibWrapper PUBLIC ${BUILD_OPTIONS})
target_compile_options(Mlib PUBLIC ${BUILD_OPTIONS})

# Require dot, treat the other components as optional
find_package(Doxygen
        OPTIONAL_COMPONENTS dot mscgen dia)

# Set Doxygen to generate docs
set(DOXYGEN_GENERATE_HTML YES)

# Generate Doxygen docs
if (DOXYGEN_FOUND)
    doxygen_add_docs(
            doxygen
            ${PROJECT_SOURCE_DIR}/include
            COMMENT "Generate html pages"
    )
    message(STATUS "Generated Doxygen docs")
else (DOXYGEN_FOUND)
    message(STATUS "Skipped generating Doxygen docs")
endif (DOXYGEN_FOUND)