cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(FOGGDD LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -Xcompiler -fopenmp")

set(OpenCV_DIR /home/gustavostahl/CPP_builds/opencv_build/opencv/build)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} eigen)

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit)

add_library(cnpy SHARED "cnpy/cnpy.cpp")

add_executable( FOGGDD FOGGDD.cpp )
target_link_libraries( FOGGDD ${OpenCV_LIBS} cnpy OpenMP::OpenMP_CXX CUDA::nppc)

# This line of code will generate a binary optimized to my CUDA architecture (Pascal).
# Depending on the GPU, this should be changed
set_target_properties(FOGGDD PROPERTIES CUDA_ARCHITECTURES "61")