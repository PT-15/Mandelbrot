cmake_minimum_required(VERSION 3.7)

project(mandelbrot)

find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

set(SOURCES
    src/main.cpp)

add_executable(mandelbrot ${SOURCES})
target_link_libraries(mandelbrot ${SDL2_LIBRARIES})