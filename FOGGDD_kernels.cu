#include <stdio.h>
#include <iostream>

#include "helper_cuda.h"
#include "helper_npp.h"

#define DIRECTIONS_MAX 8
#define FILTER_MAX 7
#define BLOCK_SIZE 32

/* 
   ##.##
   #...#
   ..... = 12 corners
   #...#
   ##.## 
*/
#define CORNER_NUM 12
#define MASK_MAX FILTER_MAX*FILTER_MAX - CORNER_NUM

__device__ float determinant(float triang_mat[][DIRECTIONS_MAX], const size_t rank)
{
    float ratio, det=1.f;

    for(size_t i=0; i<rank; i++)
    {
        // if(triang_mat[i][i] == 0.0) {printf("Mathematical Error!");}
        
        for(size_t j=i+1; j<rank; j++)
        {
            ratio = triang_mat[j][i]/triang_mat[i][i];

            for(size_t k=0; k<rank; k++)
            {
                triang_mat[j][k] = triang_mat[j][k] - ratio*triang_mat[i][k];
            }
        }
    }

    for(size_t i=0; i<rank; i++)
    {
        det *= triang_mat[i][i];
    }

    return det;
}

__device__ float trace(const float matrix[][DIRECTIONS_MAX], const size_t rank)
{
    float sum_fdiag=0.f;

    // Iter through the first diagonal
    for(size_t i=0; i<rank; i++)
    {
        sum_fdiag += matrix[i][i];
    }

    return sum_fdiag;
}

__constant__ uchar2 noncorner_coords[MASK_MAX];

__global__ void 
__launch_bounds__(BLOCK_SIZE*BLOCK_SIZE)
d_first_corner_measures(const float* im_templates, 
                        const size_t im_templates_pitch,
                        float* corner_measure,
                        const size_t corner_measure_pitch,
                        const size_t width, 
                        const size_t height, 
                        const size_t directions_n, 
                        const size_t filter_size, 
                        const float eps)
{    
    const int padding_size = filter_size/2; // floor division

    // NOTE: the use of ptrdiff_t is due to signed and unsigned conversions
    const int col_global = threadIdx.x + blockIdx.x * blockDim.x;
    const int row_global = threadIdx.y + blockIdx.y * blockDim.y;

    // Check if thread is outside the image "padded" region
    if (col_global >= width || row_global >= height)
        return;

    __shared__ float im_template_shr[DIRECTIONS_MAX][BLOCK_SIZE+FILTER_MAX-1][BLOCK_SIZE+FILTER_MAX-1];

    const int non_pad_x = threadIdx.x + padding_size;
    const int non_pad_y = threadIdx.y + padding_size;

    bool is_padding = false;
    bool is_padding_zeros = false;

    const int left_shift = threadIdx.x - padding_size;
    const int right_shift = threadIdx.x + padding_size;
    const int top_shift = threadIdx.y - padding_size;
    const int bottom_shift = threadIdx.y + padding_size;

    is_padding = (left_shift < 0 || right_shift >= BLOCK_SIZE ||
                  top_shift < 0 || bottom_shift >= BLOCK_SIZE);

    if(is_padding)
    {
        is_padding_zeros = col_global - padding_size < 0 || col_global + padding_size >= width ||
                           row_global - padding_size < 0 || row_global + padding_size >= height;        
    }

    // Copy 'directions_n' tiles into shared memory
    float val;
    for (size_t direction_idx = 0; direction_idx < directions_n; direction_idx++)
    {
        if(is_padding)
        {
            int col_offset, row_offset;

            if(left_shift < 0)
            {
                col_offset = -padding_size;
                row_offset = 0;

                if(is_padding_zeros) { val = 0.f; }
                else { val = *((float*)((char*)im_templates + (direction_idx * height + row_global + row_offset) * im_templates_pitch) + col_global + col_offset); }

                im_template_shr[direction_idx][non_pad_y + row_offset][non_pad_x + col_offset] = val;
            }

            if(left_shift < 0 && top_shift < 0)
            {
                col_offset = -padding_size;
                row_offset = -padding_size;

                if(is_padding_zeros) { val = 0.f; }
                else { val = *((float*)((char*)im_templates + (direction_idx * height + row_global + row_offset) * im_templates_pitch) + col_global + col_offset); }

                im_template_shr[direction_idx][non_pad_y + row_offset][non_pad_x + col_offset] = val;
            }

            if(left_shift < 0 && bottom_shift >= BLOCK_SIZE)
            {
                col_offset = -padding_size;
                row_offset = padding_size;

                if(is_padding_zeros) { val = 0.f; }
                else { val = *((float*)((char*)im_templates + (direction_idx * height + row_global + row_offset) * im_templates_pitch) + col_global + col_offset); }

                im_template_shr[direction_idx][non_pad_y + row_offset][non_pad_x + col_offset] = val;
            }

            if(right_shift >= BLOCK_SIZE)
            {
                col_offset = padding_size;
                row_offset = 0;

                if(is_padding_zeros) { val = 0.f; }
                else { val = *((float*)((char*)im_templates + (direction_idx * height + row_global + row_offset) * im_templates_pitch) + col_global + col_offset); }

                im_template_shr[direction_idx][non_pad_y + row_offset][non_pad_x + col_offset] = val;
            }

            if(right_shift >= BLOCK_SIZE && top_shift < 0)
            {
                col_offset = padding_size;
                row_offset = -padding_size;

                if(is_padding_zeros) { val = 0.f; }
                else { val = *((float*)((char*)im_templates + (direction_idx * height + row_global + row_offset) * im_templates_pitch) + col_global + col_offset); }

                im_template_shr[direction_idx][non_pad_y + row_offset][non_pad_x + col_offset] = val;
            }

            if(right_shift >= BLOCK_SIZE && bottom_shift >= BLOCK_SIZE)
            {
                col_offset = padding_size;
                row_offset = padding_size;

                if(is_padding_zeros) { val = 0.f; }
                else { val = *((float*)((char*)im_templates + (direction_idx * height + row_global + row_offset) * im_templates_pitch) + col_global + col_offset); }

                im_template_shr[direction_idx][non_pad_y + row_offset][non_pad_x + col_offset] = val;
            }

            if(top_shift < 0)
            {
                col_offset = 0;
                row_offset = -padding_size;

                if(is_padding_zeros) { val = 0.f; }
                else { val = *((float*)((char*)im_templates + (direction_idx * height + row_global + row_offset) * im_templates_pitch) + col_global + col_offset); }

                im_template_shr[direction_idx][non_pad_y + row_offset][non_pad_x + col_offset] = val;
            }

            if(bottom_shift >= BLOCK_SIZE)
            {
                col_offset = 0;
                row_offset = padding_size;

                if(is_padding_zeros) { val = 0.f; }
                else { val = *((float*)((char*)im_templates + (direction_idx * height + row_global + row_offset) * im_templates_pitch) + col_global + col_offset); }

                im_template_shr[direction_idx][non_pad_y + row_offset][non_pad_x + col_offset] = val;
            }
        }

        val = *((float*)((char*)im_templates + (direction_idx * height + row_global) * im_templates_pitch) + col_global);
        im_template_shr[direction_idx][non_pad_y][non_pad_x] = val;
    }

    __syncthreads(); 

    float template_symmetric[DIRECTIONS_MAX][DIRECTIONS_MAX];
    // Initialise the result matrix
    for (size_t i = 0; i < directions_n; i++) 
    {
        for (size_t j = 0; j < directions_n; j++) 
        {
            template_symmetric[i][j] = 0.f;
        }
    }         

    const size_t mask_len = filter_size*filter_size - CORNER_NUM; 

    // Add the A * At contributions for this pixel
    // Avoid extra loopings by noting that the matrix is symmetrical, we will mirror it after
    for (size_t i = 0; i < directions_n; i++) 
    {
        for (size_t j = i; j < directions_n; j++) 
        {
            // Loop through the xy kernel
            // We have precomputed the valid kernel coord with corners removed in noncorner_coords
            for (size_t k = 0; k < mask_len; k++)
            {
                const size_t curr_row = threadIdx.y + noncorner_coords[k].y;
                const size_t curr_col = threadIdx.x + noncorner_coords[k].x;     
                
                template_symmetric[i][j] += im_template_shr[i][curr_row][curr_col] * im_template_shr[j][curr_row][curr_col];
            }
        }
    }

    // Mirror the matrix about the diagonal
    for (size_t i = 0; i < directions_n; i++) 
    {
        for (size_t j = 0; j < i; j++) 
        {
            template_symmetric[i][j] = template_symmetric[j][i];
        }
    }    

    const float trc = trace(template_symmetric, directions_n);
    // to save registers, the input matrix isn't coppied; thus, changed inplace
    const float det = determinant(template_symmetric, directions_n); 

    float *corner_measure_row = (float*)((char*)corner_measure + row_global * corner_measure_pitch);
    corner_measure_row[col_global] = det / (trc + eps);
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

void compute_noncorner_coords(size_t filter_size)
{
    size_t corner_indexes[] = {0, 1, filter_size-2, 
                               filter_size-1, filter_size, 2*filter_size-1, 
                               filter_size*(filter_size-2), filter_size*(filter_size-1)-1, filter_size*(filter_size-1),
                               filter_size*(filter_size-1)+1, filter_size*filter_size-2, filter_size*filter_size-1}; 

    uchar2 h_noncorner_coords[MASK_MAX];
    size_t noncorner_count = 0;
    for(unsigned i=0; i<filter_size; i++)
    {
        for(unsigned j=0; j<filter_size; j++)
        {
            bool is_corner = false;
            #pragma unroll
            for(auto corner_index : corner_indexes)
            {
                if(i * filter_size + j == corner_index)
                {
                    is_corner = true;
                    break;
                }
            }

            if(is_corner)
            {
                continue;
            }

            h_noncorner_coords[noncorner_count] = make_uchar2(i, j);
            noncorner_count += 1;
        }
    }

    checkCudaErrors(cudaMemcpyToSymbol(noncorner_coords, h_noncorner_coords, sizeof(uchar2)*noncorner_count));
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
    compute_noncorner_coords(filter_size);

    size_t d_im_templates_pitch, d_corner_measures_pitch;

    float *d_im_templates, *d_corner_measures, *h_corner_measures;

    checkCudaErrors(cudaMallocPitch(&d_im_templates, 
                                    &d_im_templates_pitch, 
                                    sizeof(float) * width, 
                                    height * directions_n));
    checkCudaErrors(cudaMemcpy2D(d_im_templates, 
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
    dim3 grid_dim((width+BLOCK_SIZE-1)/BLOCK_SIZE, (height+BLOCK_SIZE-1)/BLOCK_SIZE);
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