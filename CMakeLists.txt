cmake_minimum_required(VERSION 3.9)
project(MBLib VERSION 0.1
        DESCRIPTION "A deep learning library"
        LANGUAGES CXX)

set(TEST_TARGET testmain)
include_directories(include)

find_package(Doxygen
        REQUIRED dot
        OPTIONAL_COMPONENTS mscgen dia)



# Added submodules (from https://cliutils.gitlab.io/modern-cmake)
add_subdirectory(extern/Catch2)
add_subdirectory(extern/Eigen3)
add_subdirectory(extern/pybind11)
add_subdirectory(extern/spdlog)

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
        include/NN.h
        )
set(SRCS
        src/NN.cpp
        )

add_library(NN_lib ${SRCS} ${HDRS})

add_executable(test ../tests/test.cpp)
target_link_libraries(test NN_lib Catch2::Catch2)

add_executable(${TEST_TARGET} ../main.cpp)

target_link_libraries(NN_lib Eigen3::Eigen spdlog::spdlog)
target_link_libraries(testmain NN_lib)

target_compile_features(${TEST_TARGET} PUBLIC cxx_std_17)
set_target_properties(${TEST_TARGET} PROPERTIES CXX_EXTENSIONS OFF)
