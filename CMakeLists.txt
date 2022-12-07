cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(FOGGDD LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -Xcompiler -fopenmp -lineinfo")

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} eigen)

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit)
find_package(ArrayFire REQUIRED)

add_library( FOGGDD_kernel SHARED FOGGDD_kernels.cu )
target_link_libraries(FOGGDD_kernel)

# This line of code will generate a binary optimized to my CUDA architecture (Pascal).
# Depending on the GPU, this should be changed
set_target_properties(FOGGDD_kernel PROPERTIES CUDA_ARCHITECTURES "61")

add_executable( FOGGDD FOGGDD.cpp )
target_link_libraries( FOGGDD ${OpenCV_LIBS} FOGGDD_kernel ArrayFire::afcuda)

add_executable( FOAGDD FOAGDD.cpp )
target_link_libraries( FOAGDD ${OpenCV_LIBS} FOGGDD_kernel ArrayFire::afcuda)

add_executable( FOGGDD_cpu FOGGDD_cpu.cpp )
target_link_libraries( FOGGDD_cpu ${OpenCV_LIBS} OpenMP::OpenMP_CXX )