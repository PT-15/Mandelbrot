cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(mandelbrot_gpu)

#set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(CUDA REQUIRED)
find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

set(SOURCES
    src/main_gpu.cu)

set(CUDA_NVCC_FLAGS " -gencode arch=compute_50,code=sm_50")

cuda_add_executable(mandelbrot_gpu ${SOURCES})
target_link_libraries(mandelbrot_gpu ${SDL2_LIBRARIES})
