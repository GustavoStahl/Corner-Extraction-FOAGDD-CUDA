cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(FOAGDD LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -Xcompiler -fopenmp -lineinfo")

set(OpenCV_DIR /home/gustavostahl/CPP_builds/opencv_build/opencv/build)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} eigen)

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit)
find_package(ArrayFire REQUIRED)

add_library( FOAGDD_kernel SHARED FOAGDD_kernels.cu )
target_link_libraries(FOAGDD_kernel)

# This line of code will generate a binary optimized to my CUDA architecture (Pascal).
# Depending on the GPU, this should be changed
set_target_properties(FOAGDD_kernel PROPERTIES CUDA_ARCHITECTURES "61")

add_executable( FOAGDD FOAGDD.cpp )
target_link_libraries( FOAGDD ${OpenCV_LIBS} FOAGDD_kernel ArrayFire::afcuda)

add_executable( FOAGDD_cpu FOAGDD_cpu.cpp )
target_link_libraries( FOAGDD_cpu ${OpenCV_LIBS} OpenMP::OpenMP_CXX )