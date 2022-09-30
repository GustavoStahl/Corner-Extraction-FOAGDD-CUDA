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

cv::Mat nonma(cv::Mat cim, double threshold, size_t radius)
{
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(radius, radius));
    cv::Mat mx, cimmx, cimmx_nonzero;
    cv::dilate(cim, mx, kernel);

    cv::Mat bordermask = cv::Mat::zeros(cim.size(), CV_8U);
    size_t width = cim.cols - 2*radius-1;
    size_t height = cim.rows - 2*radius-1;
    cv::Rect roi(radius+1, radius+1, width, height);
    bordermask(roi) = 1;

    cv::bitwise_and(cim == mx, cim > threshold, cimmx, bordermask);
    cv::findNonZero(cimmx, cimmx_nonzero);    
    return cimmx_nonzero;
}

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
            cv::absdiff(im_template, cv::Scalar::all(0), im_template);
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

    // cnpy::npy_save("../pred_arrays/templates.npy", reinterpret_cast<double *>(im_templates[0][0].data), {im_templates[0][0].rows, im_templates[0][0].cols}, "w");

    cv::Mat mask = cv::Mat::ones(patch_size, patch_size, CV_8U), mask_indexes;
    mask.at<uint8_t>(0,0) = 0;
    mask.at<uint8_t>(0,1) = 0;
    mask.at<uint8_t>(0,patch_size-1) = 0;
    mask.at<uint8_t>(0,patch_size-2) = 0;
    mask.at<uint8_t>(1,0) = 0;
    mask.at<uint8_t>(1,patch_size-1) = 0;
    mask.at<uint8_t>(patch_size-2,0) = 0;
    mask.at<uint8_t>(patch_size-2,patch_size-1) = 0;
    mask.at<uint8_t>(patch_size-1,0) = 0;
    mask.at<uint8_t>(patch_size-1,1) = 0;
    mask.at<uint8_t>(patch_size-1,patch_size-1) = 0;
    mask.at<uint8_t>(patch_size-1,patch_size-2) = 0;
    cv::findNonZero(mask, mask_indexes);
    size_t mask_len = mask_indexes.total();

    cv::Mat corner_measure(rows, cols, CV_64F);
    start = std::chrono::steady_clock::now();
    #pragma omp parallel for collapse(2)
    for(size_t i=0; i<rows; i++)
    {
        for(size_t j=0; j<cols; j++)
        {
            cv::Rect roi(j + patch_size - 3,i + patch_size - 3,patch_size,patch_size);
            cv::Mat templates_slice(directions_n, mask_len, CV_64F);
            for(size_t d=0; d<directions_n; d++)
            {
                cv::Mat im_template = im_templates[d][0](roi);
                for(size_t mask_i=0; mask_i < mask_len; mask_i++)
                {
                    cv::Point point = mask_indexes.at<cv::Point>(mask_i);
                    templates_slice.at<double>(d,mask_i) = im_template.at<double>(point);
                }
            }
            cv::Mat template_symmetric(directions_n, directions_n, CV_64F);
            //NOTE this matrix is symmetric, thus it has real eigenvalues and eigenvectors
            cv::mulTransposed(templates_slice, template_symmetric, false); // templates_slice * templates_slice.T
            //NOTE approximation of: product of eigenvalues / sum of eigenvalues
            corner_measure.at<double>(i,j) = cv::determinant(template_symmetric) / (cv::trace(template_symmetric)[0] + eps);
        }
    }
    end = std::chrono::steady_clock::now();
    std::cout << "Iter through the first scale: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    start = std::chrono::steady_clock::now();
    cv::Mat points_of_interest = nonma(corner_measure, threshold, nonma_radius);
    end = std::chrono::steady_clock::now();
    std::cout << "Nonma: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    // cnpy::npy_save("../pred_arrays/measure_nonma.npy", reinterpret_cast<double *>(corner_measure.data), {corner_measure.rows, corner_measure.cols});

    start = std::chrono::steady_clock::now();
    for(size_t sigma_idx=1; sigma_idx < sigmas.size(); sigma_idx++)
    {
        std::vector<cv::Point> points_of_interest_filtered;
        size_t points_of_interest_len = points_of_interest.total();
        for(size_t point_idx=0; point_idx<points_of_interest_len; point_idx++)
        {
            cv::Point point = points_of_interest.at<cv::Point>(point_idx);
            int i = point.y, j = point.x;
            int y = i + patch_size - 3, x = j + patch_size - 3;

            cv::Rect roi(x,y,patch_size,patch_size);
            cv::Mat templates_slice(directions_n, mask_len, CV_64F);
            for(size_t d=0; d<directions_n; d++)
            {
                cv::Mat im_template = im_templates[d][sigma_idx](roi);
                for(size_t mask_i=0; mask_i < mask_len; mask_i++)
                {
                    cv::Point point = mask_indexes.at<cv::Point>(mask_i);
                    templates_slice.at<double>(d,mask_i) = im_template.at<double>(point);
                }
            }
            cv::Mat template_symmetric(directions_n, directions_n, CV_64F);
            //NOTE this matrix is symmetric, thus it has real eigenvalues and eigenvectors
            cv::mulTransposed(templates_slice, template_symmetric, false); // templates_slice * templates_slice.T
            //NOTE approximation of: product of eigenvalues / sum of eigenvalues
            double measure = cv::determinant(template_symmetric) / (cv::trace(template_symmetric)[0] + eps);   
            if (measure > threshold)
            {
                points_of_interest_filtered.push_back(point);
            }
        }
        points_of_interest = cv::Mat(points_of_interest_filtered, true);
    }    
    end = std::chrono::steady_clock::now();
    std::cout << "Iter through scales: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    return points_of_interest;

    // std::vector<int> output; 
    // for(size_t point_idx=0; point_idx < points_of_interest.total(); point_idx++)
    // {
    //     cv::Point point = points_of_interest.at<cv::Point>(point_idx);
    //     output.push_back(point.y);
    //     output.push_back(point.x);
    // }
    // cnpy::npy_save("../pred_arrays/measure_3.npy", &output[0], {output.size()});
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

    for(size_t point_idx=0; point_idx < points_of_interest.total(); point_idx++)
    {
        cv::Point point = points_of_interest.at<cv::Point>(point_idx);
        cv::drawMarker(img, point, cv::Scalar(0,0,255), cv::MARKER_SQUARE, 2, 1, cv::LINE_AA);
    }
    cv::namedWindow("jonas", 0);
    cv::imshow("jonas", img);
    cv::waitKey(0);
    cv::destroyAllWindows();
}