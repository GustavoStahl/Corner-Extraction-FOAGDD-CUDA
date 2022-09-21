#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>

#include "cnpy/cnpy.h"
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>
#include <npp.h>

#include <opencv2/cudafilters.hpp>
#include <opencv2/core/cuda.hpp>

std::vector<std::vector<cv::Mat>> compute_templates(const cv::Mat &im_padded, int directions_n, std::vector<double> sigmas, double rho, int lattice_size)
{
    Eigen::Matrix<double,2,2,Eigen::RowMajor> rho_mat {{rho, 0.0}, {0.0, 1/rho}};
    auto lattice_xx = Eigen::RowVectorXd::LinSpaced(lattice_size, -(lattice_size/2), lattice_size/2).replicate(lattice_size,1);
    auto lattice_yy = Eigen::VectorXd::LinSpaced(lattice_size, -(lattice_size/2), lattice_size/2).replicate(1,lattice_size);

    std::vector<std::vector<cv::Mat>> im_templates(directions_n, std::vector<cv::Mat>(sigmas.size()));
    for(size_t direction_idx=0; direction_idx < directions_n; direction_idx++)
    {
        double theta = direction_idx * M_PI / directions_n;
        Eigen::RowVector2d theta_2d {cos(theta), sin(theta)};
        Eigen::Matrix<double,2,2,Eigen::RowMajor> R {{ cos(theta), sin(theta)}, 
                                                     {-sin(theta), cos(theta)}};
        auto R_T = R.transpose();

        for(size_t sigma_idx=0; sigma_idx < sigmas.size(); sigma_idx++)
        {
            double sigma = sigmas[sigma_idx];

            Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> anigs_direction(lattice_size,lattice_size);

            #pragma omp parallel for collapse(2)
            for(size_t i=0; i<lattice_size; i++)
            {
                for(size_t j=0; j<lattice_size; j++)
                {
                    Eigen::Vector2d n {lattice_xx(i,j), lattice_yy(i,j)}; // [nx, ny].T

                    double agk = 1/(2 * M_PI * sigma * sigma) * exp(-1/(2 * sigma) *  n.transpose() * R_T * rho_mat * R * n);
                    auto agdd = -rho * theta_2d.dot(n) * agk;

                    anigs_direction(i,j) = agdd;
                }
            }
            anigs_direction.array() -= anigs_direction.sum() / anigs_direction.size();
            anigs_direction.transposeInPlace(); //TODO check how to remove this

            cv::Mat conv_filter, im_template;
            cv::eigen2cv(anigs_direction, conv_filter);

            cv::flip(conv_filter,conv_filter,-1);

            // conv_filter_gpu.upload(conv_filter);
            // cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createLinearFilter(im_padded_gpu.type(), 
            //                                                                 im_padded_gpu.type(), 
            //                                                                 conv_filter, 
            //                                                                 cv::Point(-1,-1), 
            //                                                                 cv::BORDER_CONSTANT);
            // filter->apply(im_padded_gpu, im_template_gpu);
            // im_template_gpu.download(im_template);

            cv::filter2D(im_padded, im_template, -1, conv_filter, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
            im_templates[direction_idx][sigma_idx] = im_template;
        }
    }

    return im_templates;
}

cv::Mat foggdd(const cv::Mat &img)
{
    double rho = 1.5, eps = 2.22e-16, threshold = pow(10, 8.4);
    std::vector<double> sigmas = {1.5, 3.0, 4.5};
    int directions_n = 8, nonma_radius = 5;
    int lattice_size = 31; // consider the origin in the lattice

    cv::Mat img_gray;
    if(img.channels() != 1)
        cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    else
        img.copyTo(img_gray);

    img_gray.convertTo(img_gray, CV_64F);

    int rows = img_gray.rows;
    int cols = img_gray.cols;

    int patch_size = 7;
    cv::Mat im_padded;
    cv::copyMakeBorder(img_gray, im_padded, patch_size, patch_size, patch_size, patch_size, cv::BORDER_REFLECT);

    // cv::cuda::GpuMat im_padded_gpu, conv_filter_gpu, im_template_gpu;
    // im_padded_gpu.upload(im_padded);

    auto start = std::chrono::steady_clock::now();
    std::vector<std::vector<cv::Mat>> im_templates = compute_templates(im_padded, directions_n, sigmas, rho, lattice_size);
    auto end = std::chrono::steady_clock::now();
    std::cout << "Computed templates: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    return cv::Mat();

}

void cnpy2eigen(std::string data_fname, cv::Mat &out_mat){
    cnpy::NpyArray npy_data = cnpy::npy_load(data_fname);
    int data_row = npy_data.shape[0];
    int data_col = npy_data.shape[1];
    double* ptr = static_cast<double *>(malloc(data_row * data_col * sizeof(double)));
    memcpy(ptr, npy_data.data<double>(), data_row * data_col * sizeof(double));
    out_mat = cv::Mat(data_col, data_row, CV_8U, ptr); // CV_64F is equivalent double
}

int main(int argc, char **argv)
{
    Eigen::initParallel();
    std::cout << "Eigen will be using: " << Eigen::nbThreads() << " threads\n";

    cv::Mat img = cv::imread("../17.bmp");
    cv::Mat points_of_interest;

    cv::Mat corner_measure;
    cnpy2eigen("../gt_arrays/measure_nonma.npy", corner_measure);

    auto start = std::chrono::steady_clock::now();
    points_of_interest = foggdd(img);
    auto end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time in milliseconds: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

}