# Corner extraction FOAGDD CUDA
This repository implements the FOAGDD corner extraction method on the GPU using NVIDIA's CUDA.

The chart below shows the speed-up obtained with this implementation.

![alt](https://i.imgur.com/pSWydOk.png)

For more information regarding the optimization process and the improvements obtained regarding the original FOAGDD proposal please refer to the paper [Parallelization of the FOAGDD point of interest extraction technique using the CUDA architecture](https://repositorio.unesp.br/bitstream/handle/11449/239092/stahl_gh_tcc_bauru.pdf?sequence=6&isAllowed=y) (Portuguese version).

## Requirements
C++ libraries
- Eigen
- OpenCV
- OpenMP
- ArrayFire
- CUDAToolkit >= 10.2

Python libraries
- OpenCV

## Instructions
There are three implementations for the FOAGDD:
- `FOAGDD.py`: implementation in Python using Numpy and OpenCV to accelerate the computations.
- `FOAGDD_cpu.cpp`: implementation in CPP using OpenMP to accelerate some `for` loops
- `FOAGDD.cpp`: implementation in CPP leveraging CUDA kernels to accelerate some computations.

To build the binaries for the C++ code run the following commands:
```
mkdir build
cd build
cmake ..
make
cd ..
```
After the binaries are ready, run them with: 
```
./build/FOAGDD <number-of-iterations> <image-path>
```
> **Note:** the number of iterations doesn't affect the corner extraction quality, it's used to benchmark the implementation.


If everything goes well, the algorithm should extract the corners in the input image using the FOAGDD technique. The corner points are drawn in the image and saved on disk in the path `data/result.jpg`. Below you can see the results for two samples from the University of South Florida dataset.

Sample 140 | Sample 208
:-------------------------:|:-------------------------:
![alt](https://i.imgur.com/N5jd3vt.png) | ![alt](https://i.imgur.com/y5XwGS7.png)
