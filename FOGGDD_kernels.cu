#include <stdio.h>
#include <iostream>
#include "helper_cuda.h"

#define DIRECTIONS_N 8
#define MASK_LEN 37
#define BLOCK_SIZE 32

__device__ float determinant(const float matrix[], const size_t rank)
{
    float ratio, det=1.0;

    float triang_mat[DIRECTIONS_N*DIRECTIONS_N];
    for(size_t i=0; i<rank*rank; i++)
        triang_mat[i] = matrix[i];

    for(size_t i=0; i<rank; i++)
    {
         if(triang_mat[i*rank + i] == 0.0)
         {
              printf("Mathematical Error!");
         }
         for(size_t j=i+1; j< rank; j++)
         {
              ratio = triang_mat[j*rank+i]/triang_mat[i*rank+i];

              for(size_t k=0; k< rank; k++)
              {
                     triang_mat[j*rank + k] = triang_mat[j*rank + k] - ratio*triang_mat[i*rank + k];
              }
         }
    }

    for(size_t i=0; i< rank; i++)
    {
        det *= triang_mat[i*rank + i];
    }

    return det;
}

__device__ float trace(const float matrix[], const size_t rank)
{
    float sum_fdiag=0.0;

    // Iter through the first diagonal
    for(size_t i=0; i<rank; i++)
    {
        sum_fdiag += matrix[i*rank + i];
    }

    return sum_fdiag;
}

__global__ void d_first_corner_measures(const float* im_templates, 
                                        const size_t im_templates_pitch,
                                        float* corner_measure,
                                        const size_t corner_measure_pitch,
                                        const size_t width, 
                                        const size_t height, 
                                        const size_t directions_n, 
                                        const size_t filter_size, 
                                        const float eps)
{
    const size_t padding_size = filter_size/2; // floor division
    const size_t padding_size_twice = filter_size - 1;

    // NOTE: the use of ptrdiff_t is due to signed and unsigned conversions
    const size_t col_global = threadIdx.x + blockIdx.x * (blockDim.x - (ptrdiff_t)padding_size_twice);
    const size_t row_global = threadIdx.y + blockIdx.y * (blockDim.y - (ptrdiff_t)padding_size_twice);

    // Check if thread is outside the image "padded" region
    if (col_global >= width + padding_size_twice || row_global >= height + padding_size_twice)
        return;

    __shared__ float im_template_shr[BLOCK_SIZE][BLOCK_SIZE];

    const ptrdiff_t col_global_shifted = col_global - (ptrdiff_t)padding_size;
    const ptrdiff_t row_global_shifted = row_global - (ptrdiff_t)padding_size;

    bool is_padding_zeros = col_global_shifted < 0 || col_global_shifted >= (ptrdiff_t)width || 
                            row_global_shifted < 0 || row_global_shifted >= (ptrdiff_t)height;

    const size_t col_local = threadIdx.x;
    const size_t row_local = threadIdx.y;

    if (is_padding_zeros) 
    {
        im_template_shr[row_local][col_local] = 0.f;
        return;
    }       

    const ptrdiff_t col_local_shifted = (ptrdiff_t)col_local - (ptrdiff_t)padding_size;
    const ptrdiff_t row_local_shifted = (ptrdiff_t)row_local - (ptrdiff_t)padding_size;    

    bool is_padding = col_local_shifted < 0 || col_local_shifted >= blockDim.x - (ptrdiff_t)padding_size_twice || 
                      row_local_shifted < 0 || row_local_shifted >= blockDim.y - (ptrdiff_t)padding_size_twice;
      
    // const size_t mask_len = filter_size*filter_size - 12; //TODO fix this
    // 0, 1, 5, 6, 7, 13, 35, 41, 42, 43, 47, 48
    // int skip_idxs[12] = {0, 1, filter_size-2, 
    //                      filter_size-1, filter_size, 2*filter_size-1, 
    //                      filter_size*(filter_size-2), filter_size*(filter_size-1)-1, filter_size*(filter_size-1),
    //                      filter_size*(filter_size-1)+1, filter_size*filter_size-2, filter_size*filter_size-1
    //                     };

    float templates_slice[DIRECTIONS_N * MASK_LEN];   

    for(size_t direction_idx = 0; direction_idx < directions_n; direction_idx++)
    {
        const size_t row_offset = direction_idx * height;

        float val = *((float*)((char*)im_templates + (row_offset + row_global_shifted) * im_templates_pitch) + col_global_shifted);
        im_template_shr[row_local][col_local] = val;

        __syncthreads();    

        if(! is_padding)
        {
            size_t mask_count = 0;
            for(size_t pad_row=0; pad_row < filter_size; pad_row++)
            {
                const size_t curr_row = row_local_shifted + pad_row;
                for(size_t pad_col=0; pad_col < filter_size; pad_col++)
                {
                    const size_t mask_idx_curr = pad_row * filter_size + pad_col;
    
                    if(mask_idx_curr == 0 || mask_idx_curr == 1 || mask_idx_curr == 5 || 
                       mask_idx_curr == 6 || mask_idx_curr == 7 || mask_idx_curr == 13 || 
                       mask_idx_curr == 35 || mask_idx_curr == 41 || mask_idx_curr == 42 || 
                       mask_idx_curr == 43 || mask_idx_curr == 47 || mask_idx_curr == 48) 
                       continue;
    
                    const size_t curr_col = col_local_shifted + pad_col;
                    
                    const size_t template_slice_idx = direction_idx * MASK_LEN + mask_count;
   
                    templates_slice[template_slice_idx] = im_template_shr[curr_row][curr_col];
    
                    mask_count++;
                }
            } 
        }
        __syncthreads();    
    }

    if(is_padding)
        return;

    // Matrix multiplication
    float template_symmetric[DIRECTIONS_N * DIRECTIONS_N];
    for (size_t i = 0; i < directions_n; i++) 
    {
        for (size_t j = 0; j < directions_n; j++) 
        {
            template_symmetric[i*directions_n + j] = 0.f;
            for (size_t k = 0; k < MASK_LEN; k++)
            {
                // Equivalent to A @ A.T
                template_symmetric[i*directions_n + j] += templates_slice[i*MASK_LEN + k] * templates_slice[j*MASK_LEN + k];
            }
        }
    }

    const float det = determinant(template_symmetric, directions_n);
    const float trc = trace(template_symmetric, directions_n);

    float *corner_measure_row = (float*)((char*)corner_measure + row_global_shifted * corner_measure_pitch);
    corner_measure_row[col_global_shifted] = det / (trc + eps);
}

extern "C"
void compute_templates()
{
}

extern "C"
void sequential_corner_measures()
{
}

extern "C"
int init_cuda_device(int argc, const char **argv)
{
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
        exit(EXIT_FAILURE);
    }

    int dev = findCudaDevice(argc, argv);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    std::cerr << "cudaSetDevice GPU " << dev << " = " << deviceProp.name << std::endl;

    checkCudaErrors(cudaSetDevice(dev));

    return dev;
}

/*
im_templates size: directions_n x width x height (flatten)
*/
extern "C"
float* first_corner_measures(const float *im_templates, 
                             const size_t width, 
                             const size_t height, 
                             const size_t directions_n, 
                             const size_t filter_size, 
                             const float eps)
{
    size_t d_im_templates_pitch, d_corner_measures_pitch;

    float *d_im_templates, *d_corner_measures, *h_corner_measures;

    checkCudaErrors(cudaMallocPitch(&d_im_templates, 
                                     &d_im_templates_pitch, 
                                     sizeof(float) * width, 
                                     height * directions_n));

    checkCudaErrors(cudaMemcpy2DAsync(d_im_templates, 
                                      d_im_templates_pitch, 
                                      im_templates, 
                                      sizeof(float)*width, 
                                      sizeof(float)*width, 
                                      height * directions_n, 
                                      cudaMemcpyHostToDevice));  
                                      
    checkCudaErrors(cudaMallocPitch(&d_corner_measures, 
                                    &d_corner_measures_pitch, 
                                    sizeof(float) * width, 
                                    height));    
    checkCudaErrors(cudaMallocHost(&h_corner_measures, sizeof(float) * width * height));

    ptrdiff_t useful_region = BLOCK_SIZE - filter_size + 1;
    if(useful_region < 0)
    {
        printf("No useful region\n");
    }

    // int THREADS = 16;
    dim3 block_dim(BLOCK_SIZE,BLOCK_SIZE);
    dim3 grid_dim((width+useful_region-1)/useful_region, (height+useful_region-1)/useful_region);
    d_first_corner_measures<<<grid_dim, block_dim>>>(d_im_templates, 
                                                     d_im_templates_pitch, 
                                                     d_corner_measures, 
                                                     d_corner_measures_pitch, 
                                                     width, 
                                                     height, 
                                                     directions_n, 
                                                     filter_size, 
                                                     eps);
                                                     
    checkCudaErrors(cudaMemcpy2D(h_corner_measures, 
                                  sizeof(float)*width, 
                                  d_corner_measures, 
                                  d_corner_measures_pitch, 
                                  sizeof(float)*width, 
                                  height, 
                                  cudaMemcpyDeviceToHost)); 

    checkCudaErrors(cudaFree(d_im_templates));
    checkCudaErrors(cudaFree(d_corner_measures));

    return h_corner_measures;
}