cmake_minimum_required(VERSION 2.6)
set(CMAKE_VERBOSE_MAKEFILE ON)
project(matrix_cuda LANGUAGES CXX CUDA)

# GTest 

enable_testing()
find_package(GTest REQUIRED)
if (GTest_FOUND)
    message(">> ${GTest_FOUND} ${GTest_VERSION}")
    include_directories(${GTEST_INCLUDE_DIRS})

endif()    

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
add_executable(testshgl tests.cu)
add_test(testingshgl testingshgl)
# Link test executable against gtest & gtest_main
target_link_libraries(testshgl ${GTEST_BOTH_LIBRARIES} ${CUDA_LIBRARIES})

add_executable(matrix  Matrix.cpp)
target_link_libraries(matrix )

# Add test cpp file

add_executable(testing tests.cpp)
add_test(testing testing)
# Link test executable against gtest & gtest_main
target_link_libraries(testing ${GTEST_BOTH_LIBRARIES})


