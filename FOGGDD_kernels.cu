#include <stdio.h>
#include <iostream>
#include <npp.h>

#include "helper_cuda.h"
#include "helper_npp.h"

#define DIRECTIONS_MAX 8
#define FILTER_MAX 7
#define BLOCK_SIZE 25

/* 
   ##.##
   #...#
   ..... = 12 corners
   #...#
   ##.## 
*/
#define CORNER_NUM 12
#define MASK_MAX FILTER_MAX*FILTER_MAX - CORNER_NUM

__device__ float determinant(const float matrix[][DIRECTIONS_MAX], const size_t rank)
{
    float ratio, det=1.f;

    float triang_mat[DIRECTIONS_MAX][DIRECTIONS_MAX];
    for(size_t i=0; i<rank; i++)
        for(size_t j=0; j<rank; j++)
            triang_mat[i][j] = matrix[i][j];

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

__constant__ size_t skip_idxes[CORNER_NUM];
__device__ bool is_skipable(const size_t idx)
{
    #pragma unroll
    for(size_t i=0; i<CORNER_NUM; i++)
    {
        if(skip_idxes[i] == idx)
        {
            return true;
        }
    }
    return false;
}

/* possible overhead: 
   * not enough registers 
   * uncoalesced memory access
   * unsufficient streaming multiprocessor warps
*/

__global__ void 
__launch_bounds__(BLOCK_SIZE*BLOCK_SIZE, 3)
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

    if (is_padding_zeros) 
    {
        im_template_shr[threadIdx.y][threadIdx.x] = 0.f;
        return;
    }       

    const ptrdiff_t col_local_shifted = threadIdx.x - (ptrdiff_t)padding_size;
    const ptrdiff_t row_local_shifted = threadIdx.y - (ptrdiff_t)padding_size;    

    bool is_padding = col_local_shifted < 0 || col_local_shifted >= blockDim.x - (ptrdiff_t)padding_size_twice || 
                      row_local_shifted < 0 || row_local_shifted >= blockDim.y - (ptrdiff_t)padding_size_twice;
      
    const size_t mask_len = filter_size*filter_size - CORNER_NUM; 

    float templates_slice[DIRECTIONS_MAX][MASK_MAX];   

    for(size_t direction_idx = 0; direction_idx < directions_n; direction_idx++)
    {

        float val = *((float*)((char*)im_templates + (direction_idx * height + row_global_shifted) * im_templates_pitch) + col_global_shifted);
        im_template_shr[threadIdx.y][threadIdx.x] = val;

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
    
                    if(is_skipable(mask_idx_curr)) 
                       continue;
    
                    const size_t curr_col = col_local_shifted + pad_col;
                       
                    templates_slice[direction_idx][mask_count] = im_template_shr[curr_row][curr_col];
    
                    mask_count++;
                }
            } 
        }
        __syncthreads();    
    }

    if(is_padding)
        return;

    // Matrix multiplication
    float template_symmetric[DIRECTIONS_MAX][DIRECTIONS_MAX];
    for (size_t i = 0; i < directions_n; i++) 
    {
        for (size_t j = 0; j < directions_n; j++) 
        {
            template_symmetric[i][j] = 0.f;
            for (size_t k = 0; k < mask_len; k++)
            {
                // Equivalent to A @ A.T
                template_symmetric[i][j] += templates_slice[i][k] * templates_slice[j][k];
            }
        }
    }

    const float det = determinant(template_symmetric, directions_n);
    const float trc = trace(template_symmetric, directions_n);

    float *corner_measure_row = (float*)((char*)corner_measure + row_global_shifted * corner_measure_pitch);
    corner_measure_row[col_global_shifted] = det / (trc + eps);
}

// __device__ Npp32s nSrcStep;
// __device__ Npp32f *pSrc;

extern "C"
float* set_filter_src_image(const float *h_pSrc, 
                            const int width, 
                            const int height,
                            int &nSrcStep)
{
    // Npp32s nSrcStep_tmp;
    Npp32f *d_pSrc;

    d_pSrc = nppiMalloc_32f_C1(width, height, &nSrcStep); 
    checkCudaErrors(cudaMemcpy2D(d_pSrc, 
                                 nSrcStep, 
                                 h_pSrc, 
                                 sizeof(float)*width, 
                                 sizeof(float)*width, 
                                 height, 
                                 cudaMemcpyHostToDevice));   

    return d_pSrc;                                 
                                 
    // checkCudaErrors(cudaMemcpyToSymbol(pSrc, pSrc_tmp, width*height*nSrcStep_tmp, 0, cudaMemcpyDeviceToDevice));
    // checkCudaErrors(cudaMemcpyToSymbol(&nSrcStep, &nSrcStep_tmp, sizeof(Npp32s), 0, cudaMemcpyDeviceToDevice));
}

extern "C"
float* compute_templates(float* pSrc,
                         const int pSrcStep,
                         const int width, 
                         const int height, 
                         const float *conv_filter, 
                         const int filter_size)
{
    Npp32s nDstStep;

    Npp32f *pDst, *pKernel;
    float* im_template;                               

    cudaMalloc((void**)&pKernel, sizeof(Npp32f)*filter_size*filter_size);
    cudaMemcpy(pKernel, conv_filter, sizeof(Npp32f)*filter_size*filter_size, cudaMemcpyHostToDevice);

    pDst = nppiMalloc_32f_C1(width, height, &nDstStep); 

    checkCudaErrors(cudaMallocHost(&im_template, sizeof(float) * width * height));

    NppiSize oSrcSize = {width, height};
    NppiPoint oSrcOffset = {0, 0};
    NppiSize oSizeROI = {width, height};
    NppiSize oKernelSize = {filter_size, filter_size};
    NppiPoint oAnchor = {filter_size/2, filter_size/2};

    // Npp32s nSrcStep_val;
    // Npp32f *pSrc_ptr;
    // checkCudaErrors(cudaMemcpyFromSymbol(&nSrcStep_val, nSrcStep, sizeof(Npp32s)));
    // checkCudaErrors(cudaGetSymbolAddress((void**)&pSrc_ptr, pSrc));

    NPP_CHECK_NPP(nppiFilterBorder_32f_C1R(pSrc, 
                                           pSrcStep,
                                           oSrcSize,
                                           oSrcOffset,
                                           pDst,
                                           nDstStep,
                                           oSizeROI,
                                           pKernel,
                                           oKernelSize,
                                           oAnchor,
                                           NPP_BORDER_REPLICATE));

    // NPP_CHECK_NPP(nppiAbs_32f_C1R(pDst, nDstStep, pDst, nDstStep, oSizeROI));
                             
    checkCudaErrors(cudaMemcpy2D(im_template, 
                                 sizeof(float)*width, 
                                 pDst, 
                                 nDstStep, 
                                 sizeof(Npp32f)*width, 
                                 height, 
                                 cudaMemcpyDeviceToHost)); 

    // nppiFree(pSrc);
    nppiFree(pDst);
    nppiFree(pKernel);

    return im_template;                                    
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

void update_skip_idxes(size_t filter_size)
{
    size_t h_skip_idxes[] = {0, 1, filter_size-2, 
                             filter_size-1, filter_size, 2*filter_size-1, 
                             filter_size*(filter_size-2), filter_size*(filter_size-1)-1, filter_size*(filter_size-1),
                             filter_size*(filter_size-1)+1, filter_size*filter_size-2, filter_size*filter_size-1};    
    checkCudaErrors(cudaMemcpyToSymbol(skip_idxes, h_skip_idxes, sizeof(size_t)*CORNER_NUM));
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
    update_skip_idxes(filter_size);

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