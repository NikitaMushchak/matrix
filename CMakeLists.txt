cmake_minimum_required(VERSION 2.6)
set(CMAKE_VERBOSE_MAKEFILE ON)
project(matrix_cuda LANGUAGES CXX CUDA)
#if (msvc) {
#  CMAKE_CXX_FLAGS += -openmp -arch:AVX -D "_CRT_SECURE_NO_WARNINGS"
#  CMAKE_CXX_FLAGS_RELEASE *= -Ox
#}

#if(g++) {
#  CMAKE_CXX_FLAGS += -fopenmp -mavx -fabi-version=0 -ffast-math
#  CMAKE_LFLAGS += -fopenmp
#  CMAKE_CXX_FLAGS_RELEASE *= -O3
#}
# GTest 
#add_compile_options(/W4 /arch:SSE /arch:SSE2 /MT /MTd /MDd)
enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
# Unit Tests
#CUDA

FIND_PACKAGE(CUDA REQUIRED)
INCLUDE_DIRECTORIES( ${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_INCLUDE_DIRS}/../lib/x64)
# add helper cuda tools 
#Open MP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message(">> ${OpenMP_FOUND} ${OpenMP_VERSION}")
    set(OMP_NUM_THREADS=8)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
if(CMAKE_THREAD_LIBS_INIT)
    message(">> ${THREADS_FOUND} ${THREADS_VERSION}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_THREAD_LIBS_INIT}")
endif()
# TODO remove this!!!!!
#include_directories(${PROJECT_SOURCE_DIR}/common/inc)

add_executable(matrix  Matrix.cpp)
target_link_libraries(matrix )

# Add test cpp file

add_executable(testing tests.cpp)
add_test(testing testing)
# Link test executable against gtest & gtest_main
target_link_libraries(testing ${GTEST_BOTH_LIBRARIES})

