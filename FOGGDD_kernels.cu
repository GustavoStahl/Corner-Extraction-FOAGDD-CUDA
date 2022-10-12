#include <stdio.h>
#include <iostream>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cout << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cout << cudaGetErrorString(err) << " " << func << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cout << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cout << cudaGetErrorString(err) << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

__device__ float determinant(const float matrix[], const size_t rank)
{
    float ratio, det=1.0;

    float triang_mat[8*8];
    for(size_t i=0; i<64; i++)
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

__global__ void d_first_corner_measures(const float* const im_templates, 
                                        const size_t im_templates_pitch,
                                        float* corner_measure,
                                        const size_t corner_measure_pitch,
                                        const size_t width, 
                                        const size_t height, 
                                        const size_t directions_n, 
                                        const size_t patch_size, 
                                        const float eps)
{
    const size_t col = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col >= width || row >= height)
        return;

    const size_t start_col = col + patch_size/2 + 1;
    const size_t start_row = row + patch_size/2 + 1;

    const size_t mask_len = patch_size*patch_size - 12; //TODO fix this
    // 0, 1, 5, 6, 7, 13, 35, 41, 42, 43, 47, 48
    // int skip_idxs[12] = {0, 1, patch_size-2, 
    //                      patch_size-1, patch_size, 2*patch_size-1, 
    //                      patch_size*(patch_size-2), patch_size*(patch_size-1)-1, patch_size*(patch_size-1),
    //                      patch_size*(patch_size-1)+1, patch_size*patch_size-2, patch_size*patch_size-1
    //                     };

    const size_t height_padded = height + 2*patch_size;

    float templates_slice[8 * 37];

    for(size_t direction_idx = 0; direction_idx < directions_n; direction_idx++)
    {
        size_t mask_count = 0;
        const size_t row_offset = direction_idx * height_padded;
        for(size_t pad_row=0; pad_row < patch_size; pad_row++)
        {
            const size_t curr_row = start_row + pad_row;
            for(size_t pad_col=0; pad_col < patch_size; pad_col++)
            {
                const size_t mask_idx_curr = pad_row * patch_size + pad_col;

                if(mask_idx_curr == 0 || mask_idx_curr == 1 || mask_idx_curr == 5 || 
                   mask_idx_curr == 6 || mask_idx_curr == 7 || mask_idx_curr == 13 || 
                   mask_idx_curr == 35 || mask_idx_curr == 41 || mask_idx_curr == 42 || 
                   mask_idx_curr == 43 || mask_idx_curr == 47 || mask_idx_curr == 48) 
                   continue;

                const size_t curr_col = start_col + pad_col;
                
                const size_t template_slice_idx = direction_idx * mask_len + mask_count;
                const size_t template_row = (row_offset + curr_row) * im_templates_pitch;

                templates_slice[template_slice_idx] = *((float*)((char*)im_templates + template_row) + curr_col);

                mask_count++;
            }
        } 
    }


    // Matrix multiplication
    float template_symmetric[8 * 8];
    for (size_t i = 0; i < directions_n; i++) 
    {
        for (size_t j = 0; j < directions_n; j++) 
        {
            template_symmetric[i*directions_n + j] = 0.0;
            for (size_t k = 0; k < mask_len; k++)
            {
                // Equivalent to A @ A.T
                template_symmetric[i*directions_n + j] += templates_slice[i*mask_len + k] * templates_slice[j*mask_len + k];
            }
        }
    }

    const float det = determinant(template_symmetric, directions_n);
    const float trc = trace(template_symmetric, directions_n);

    /*
    if(col == 255 && row == 255)
    {
        for(int i=0; i<64; i++)
            printf("%f ", template_symmetric[i]);
        printf("\n");
        printf("%lf %lf %lf\n", det, trc, det / (trc + eps));
    }
    */

    // int idx = row * width + col;
    // corner_measure[idx] = det / (trc + eps);

    float *corner_measure_row = (float*)((char*)corner_measure + row * corner_measure_pitch);
    corner_measure_row[col] = det / (trc + eps);;
}

extern "C"
void compute_templates()
{
}

extern "C"
void sequential_corner_measures()
{
}

/*
im_templates size: directions_n x width x height (flatten)
*/
extern "C"
float* first_corner_measures(const float *im_templates, 
                             const size_t width, 
                             const size_t height, 
                             const size_t directions_n, 
                             const size_t patch_size, 
                             const float eps)
{
    cudaSetDevice(0);

    size_t width_padded = width + 2 * patch_size, 
           height_padded = height + 2 * patch_size;

    size_t d_im_templates_pitch, d_corner_measures_pitch;;

    float *d_im_templates, *d_corner_measures, *h_corner_measures;

    CHECK_CUDA_ERROR(cudaMallocHost(&h_corner_measures, sizeof(float) * width * height));
    CHECK_CUDA_ERROR(cudaMallocPitch(&d_corner_measures, 
                                     &d_corner_measures_pitch, 
                                     sizeof(float) * width, 
                                     height));    
    CHECK_CUDA_ERROR(cudaMallocPitch(&d_im_templates, 
                                     &d_im_templates_pitch, 
                                     sizeof(float) * width_padded, 
                                     height_padded * directions_n));

    CHECK_CUDA_ERROR(cudaMemcpy2DAsync(d_im_templates, 
                                       d_im_templates_pitch, 
                                       im_templates, 
                                       sizeof(float)*width_padded, 
                                       sizeof(float)*width_padded, 
                                       height_padded * directions_n, 
                                       cudaMemcpyHostToDevice));                              

    int THREADS = 32;
    dim3 block_dim(THREADS,THREADS);
    dim3 grid_dim((width+THREADS-1)/THREADS, (height+THREADS-1)/THREADS);
    d_first_corner_measures<<<grid_dim, block_dim>>>(d_im_templates, 
                                                     d_im_templates_pitch, 
                                                     d_corner_measures, 
                                                     d_corner_measures_pitch, 
                                                     width, 
                                                     height, 
                                                     directions_n, 
                                                     patch_size, 
                                                     eps);
                                                     
    CHECK_CUDA_ERROR(cudaMemcpy2DAsync(h_corner_measures, 
                                       sizeof(float)*width, 
                                       d_corner_measures, 
                                       d_corner_measures_pitch, 
                                       sizeof(float)*width, 
                                       height, 
                                       cudaMemcpyDeviceToHost)); 

    CHECK_CUDA_ERROR(cudaFree(d_im_templates));
    CHECK_CUDA_ERROR(cudaFree(d_corner_measures));

    CHECK_LAST_CUDA_ERROR();

    return h_corner_measures;
}