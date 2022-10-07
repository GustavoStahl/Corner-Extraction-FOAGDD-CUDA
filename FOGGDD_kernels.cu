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

__device__ double determinant(double matrix[], int rank)
{
    double ratio, det=1.0;

    double triang_mat[8*8];
    for(int i=0; i<64; i++)
        triang_mat[i] = matrix[i];

    for(int i=0; i<rank; i++)
    {
         if(triang_mat[i*rank + i] == 0.0)
         {
              printf("Mathematical Error!");
         }
         for(int j=i+1; j< rank; j++)
         {
              ratio = triang_mat[j*rank+i]/triang_mat[i*rank+i];

              for(int k=0; k< rank; k++)
              {
                     triang_mat[j*rank + k] = triang_mat[j*rank + k] - ratio*triang_mat[i*rank + k];
              }
         }
    }

    for(int i=0;i< rank;i++)
    {
        det *= triang_mat[i*rank + i];
    }

    return det;
}

__device__ double trace(double matrix[], int rank)
{
    double sum_fdiag=0.0;

    // Iter through the first diagonal
    for(int i=0; i<rank; i++)
    {
        sum_fdiag += matrix[i*rank + i];
    }

    return sum_fdiag;
}

__global__ void d_first_corner_measures(double *im_templates, 
                                        double *corner_measure,
                                        int width, 
                                        int height, 
                                        int directions_n, 
                                        int patch_size, 
                                        double eps)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    // if(col == 255 && row == 255)
    // {
    //     long long int start = clock64();
    //     long long int stop = clock64();
    //     printf("%lli\n", stop-start);
    // }

    if (col >= width || row >= height)
        return;

    int start_col = col + patch_size/2 + 1;
    int start_row = row + patch_size/2 + 1;

    int mask_len = patch_size*patch_size - 12; //TODO fix this
    // 0, 1, 5, 6, 7, 13, 35, 41, 42, 43, 47, 48
    // int skip_idxs[12] = {0, 1, patch_size-2, 
    //                      patch_size-1, patch_size, 2*patch_size-1, 
    //                      patch_size*(patch_size-2), patch_size*(patch_size-1)-1, patch_size*(patch_size-1),
    //                      patch_size*(patch_size-1)+1, patch_size*patch_size-2, patch_size*patch_size-1
    //                     };

    int width_padded = width + 2*patch_size,
        height_padded = height + 2*patch_size;
    int pitch = width_padded * height_padded;

    double templates_slice[8 * 37];
    for(int direction_idx = 0; direction_idx < directions_n; direction_idx++)
    {
        double *direction_image = &im_templates[direction_idx * pitch]; //TODO put this in shared memory??

        int mask_count = 0;
        for(int pad_row=0; pad_row < patch_size; pad_row++)
        {
            int curr_row = start_row + pad_row;
            for(int pad_col=0; pad_col < patch_size; pad_col++)
            {
                int mask_idx_curr = pad_row * patch_size + pad_col;

                if(mask_idx_curr == 0 || mask_idx_curr == 1 || mask_idx_curr == 5 || 
                   mask_idx_curr == 6 || mask_idx_curr == 7 || mask_idx_curr == 13 || 
                   mask_idx_curr == 35 || mask_idx_curr == 41 || mask_idx_curr == 42 || 
                   mask_idx_curr == 43 || mask_idx_curr == 47 || mask_idx_curr == 48) 
                   continue;

                int curr_col = start_col + pad_col;
                int idx = curr_row * width_padded + curr_col;

                templates_slice[direction_idx * mask_len + mask_count] = direction_image[idx]; 

                mask_count++;
            }
        } 
    }

    // Matrix multiplication
    double template_symmetric[8 * 8];
    for (int i = 0; i < directions_n; i++) 
    {
        for (int j = 0; j < directions_n; j++) 
        {
            template_symmetric[i*directions_n + j] = 0.0;
            for (int k = 0; k < mask_len; k++)
            {
                // Equivalent to A @ A.T
                template_symmetric[i*directions_n + j] += templates_slice[i*mask_len + k] * templates_slice[j*mask_len + k];
            }
        }
    }

    double det = determinant(template_symmetric, directions_n);
    double trc = trace(template_symmetric, directions_n);

    /*
    if(col == 255 && row == 255)
    {
        for(int i=0; i<64; i++)
            printf("%f ", template_symmetric[i]);
        printf("\n");
        printf("%lf %lf %lf\n", det, trc, det / (trc + eps));
    }
    */

    int idx = row * width + col;
    corner_measure[idx] = det / (trc + eps);
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
double* first_corner_measures(double *im_templates, int width, int height, int directions_n, int patch_size, double eps)
{
    int width_padded = width + 2 * patch_size,
        height_padded = height + 2 * patch_size;

    double *d_im_templates, *d_corner_measures, *h_corner_measures;

    cudaMalloc(&d_im_templates, sizeof(double) * width_padded * height_padded * directions_n);
    cudaMalloc(&d_corner_measures, sizeof(double) * width * height);
    cudaMallocHost(&h_corner_measures, sizeof(double) * width * height);

    CHECK_CUDA_ERROR(cudaMemcpy(d_im_templates, im_templates, sizeof(double) * width_padded * height_padded * directions_n, cudaMemcpyHostToDevice));

    int THREADS = 32;
    dim3 block_dim(THREADS,THREADS);
    dim3 grid_dim((width+THREADS-1)/THREADS, (height+THREADS-1)/THREADS);
    d_first_corner_measures<<<grid_dim, block_dim>>>(d_im_templates, d_corner_measures, width, height, directions_n, patch_size, eps);
    cudaDeviceSynchronize();

    CHECK_CUDA_ERROR(cudaMemcpy(h_corner_measures, d_corner_measures, sizeof(double) * width * height, cudaMemcpyDeviceToHost));
    CHECK_LAST_CUDA_ERROR();

    return h_corner_measures;
}