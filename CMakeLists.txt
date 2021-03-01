cmake_minimum_required(VERSION 2.6)

project(matrix_cuda LANGUAGES CXX CUDA)

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

# TODO remove this!!!!!
include_directories(${PROJECT_SOURCE_DIR}/common/inc)
# Add test cpp file
add_executable(matrix  Mtrx_Mul_Sh.cu)

target_link_libraries(matrix ${CUDA_LIBRARIES} ${OpenMP_CXX_FLAGS})

# Link test executable against gtest & gtest_main
	
add_executable(testing tests.cu)
add_test(testing testing)
target_link_libraries(testing ${GTEST_BOTH_LIBRARIES} ${CUDA_LIBRARIES} ${OpenMP_CXX_FLAGS})

