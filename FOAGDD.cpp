#include <stdio.h>
#include <math.h>
#include <chrono>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>
#include <arrayfire.h>

#include "FOAGDD.h"

extern "C" int init_cuda_device(int argc, const char **argv);
extern "C" float* first_corner_measures(float *im_templates, size_t width, size_t height, size_t directions_n, size_t patch_size, float eps);

FOAGDD::FOAGDD(size_t directions_n, std::vector<float> sigmas, float rho, float threshold)
:directions_n(directions_n), sigmas(sigmas), rho(rho), threshold(threshold)
{
    af::setBackend(AF_BACKEND_CUDA);

    mask_indexes = compute_noncorner_coords(patch_size);
    mask_len = mask_indexes.total();

    im_filters = compute_filters(directions_n, sigmas, rho, lattice_size);

    size_t sigmas_n = sigmas.size();
    std::vector<cv::Mat> im_filters_collapsed(directions_n * sigmas_n);
    for(size_t direction_idx=0; direction_idx < directions_n; direction_idx++)
    {
        for(size_t sigma_idx=0; sigma_idx < sigmas_n; sigma_idx++)
        {
            im_filters_collapsed[direction_idx * sigmas_n + sigma_idx] = im_filters[direction_idx][sigma_idx];
        }
    }   
    cv::Mat im_filters_concat;
    cv::vconcat(im_filters_collapsed.data(), im_filters_collapsed.size(), im_filters_concat);    

    im_filters_gpu = af::array(lattice_size, lattice_size, 1, directions_n * sigmas_n, reinterpret_cast<float*>(im_filters_concat.data));
    af::transposeInPlace(im_filters_gpu);
}

FOAGDD::~FOAGDD()
{
    af::freePinned(this->pim_templates);
}

cv::Mat FOAGDD::compute_noncorner_coords(int fingerprint_size)
{
    cv::Mat fingerprint = cv::Mat::ones(fingerprint_size, fingerprint_size, CV_8U);
    cv::Mat non_corner_coords;

    fingerprint.at<unsigned char>(0,0) = 0;
    fingerprint.at<unsigned char>(0,1) = 0;
    fingerprint.at<unsigned char>(0,fingerprint_size-1) = 0;
    fingerprint.at<unsigned char>(0,fingerprint_size-2) = 0;
    fingerprint.at<unsigned char>(1,0) = 0;
    fingerprint.at<unsigned char>(1,fingerprint_size-1) = 0;
    fingerprint.at<unsigned char>(fingerprint_size-2,0) = 0;
    fingerprint.at<unsigned char>(fingerprint_size-2,fingerprint_size-1) = 0;
    fingerprint.at<unsigned char>(fingerprint_size-1,0) = 0;
    fingerprint.at<unsigned char>(fingerprint_size-1,1) = 0;
    fingerprint.at<unsigned char>(fingerprint_size-1,fingerprint_size-1) = 0;
    fingerprint.at<unsigned char>(fingerprint_size-1,fingerprint_size-2) = 0;
    cv::findNonZero(fingerprint, non_corner_coords);    

    return non_corner_coords;
}

std::vector<std::vector<cv::Mat>> FOAGDD::compute_filters(int directions_n, 
                                                          std::vector<float> sigmas, 
                                                          float rho, 
                                                          int lattice_size)
{
    Eigen::Matrix<float,2,2,Eigen::RowMajor> rho_mat {{rho, 0.0}, {0.0, 1/rho}};
    auto lattice_xx = Eigen::RowVectorXf::LinSpaced(lattice_size, -(lattice_size/2), lattice_size/2).replicate(lattice_size,1);
    auto lattice_yy = Eigen::VectorXf::LinSpaced(lattice_size, -(lattice_size/2), lattice_size/2).replicate(1,lattice_size);

    std::vector<std::vector<cv::Mat>> im_filters(directions_n, std::vector<cv::Mat>(sigmas.size()));
    for(size_t direction_idx=0; direction_idx < directions_n; direction_idx++)
    {
        float theta = direction_idx * M_PI / directions_n;
        Eigen::RowVector2f theta_2d {cos(theta), sin(theta)};
        Eigen::Matrix<float,2,2,Eigen::RowMajor> R {{ cos(theta), sin(theta)}, 
                                                    {-sin(theta), cos(theta)}};
        auto R_T = R.transpose();

        for(size_t sigma_idx=0; sigma_idx < sigmas.size(); sigma_idx++)
        {
            float sigma = sigmas[sigma_idx];

            Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> anigs_direction(lattice_size,lattice_size);

            for(size_t i=0; i<lattice_size; i++)
            {
                for(size_t j=0; j<lattice_size; j++)
                {
                    Eigen::Vector2f n {lattice_xx(i,j), lattice_yy(i,j)}; // [nx, ny].T

                    float agk = 1/(2 * M_PI * sigma * sigma) * exp(-1/(2 * sigma) *  n.transpose() * R_T * rho_mat * R * n);
                    auto agdd = -rho * theta_2d.dot(n) * agk;

                    anigs_direction(i,j) = agdd;
                }
            }
            anigs_direction.array() -= anigs_direction.sum() / anigs_direction.size();
            anigs_direction.transposeInPlace(); //TODO check how to remove this

            cv::Mat conv_filter;
            cv::eigen2cv(anigs_direction, conv_filter);

            im_filters[direction_idx][sigma_idx] = conv_filter; 
        }
    }
    return im_filters;
}    

cv::Mat FOAGDD::nonma(cv::Mat cim, float threshold, size_t radius)
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

std::vector<std::vector<cv::Mat>> FOAGDD::compute_templates(cv::Mat& im_gray)
{
    cv::Mat im_padded;
    cv::copyMakeBorder(im_gray, 
                       im_padded, 
                       this->patch_size, 
                       this->patch_size, 
                       this->patch_size, 
                       this->patch_size, 
                       cv::BORDER_REFLECT);

    size_t im_padded_width = im_padded.cols;
    size_t im_padded_height = im_padded.rows;

    af::array im_padded_gpu(im_padded_width, im_padded_height, reinterpret_cast<float*>(im_padded.data));
    im_padded_gpu = af::transpose(im_padded_gpu);

    af::array im_templates_gpu = af::abs(af::convolve2(im_padded_gpu, 
                                                       this->im_filters_gpu));
    im_templates_gpu = af::transpose(im_templates_gpu);
   
    im_templates_gpu.host(this->pim_templates);

    size_t sigmas_n = sigmas.size();
    std::vector<std::vector<cv::Mat>> im_templates(this->directions_n, std::vector<cv::Mat>(sigmas_n));
    for(size_t direction_idx=0; direction_idx < directions_n; direction_idx++)
    {
        for(size_t sigma_idx=0; sigma_idx < sigmas_n; sigma_idx++)
        {
            size_t shift = (direction_idx * sigmas_n + sigma_idx) * im_padded_height * im_padded_width;            
            im_templates[direction_idx][sigma_idx] = cv::Mat(im_padded_height, 
                                                             im_padded_width, 
                                                             CV_32F, 
                                                             this->pim_templates + shift);
        }
    }

    return im_templates;
}

void FOAGDD::preallocate_gpu_mem(size_t width, size_t height)
{
    if(this->pim_templates && width == this->width && height == this->height)
    {
        return;
    }

    this->pim_templates = af::pinned<float>((width + 2*this->patch_size) * 
                                            (height + 2*this->patch_size) * 
                                            this->directions_n * 
                                            this->sigmas.size());

    this->width = width;
    this->height = height;
}

cv::Mat FOAGDD::find_features(const cv::Mat &image)
{
    cv::Mat image_gray;
    if(image.channels() != 1)
        cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
    else
        image.copyTo(image_gray);

    image_gray.convertTo(image_gray, CV_32F);

    size_t image_width = image_gray.cols;
    size_t image_height = image_gray.rows;

    preallocate_gpu_mem(image_width, image_height); // preallocate if necessary

    std::vector<std::vector<cv::Mat>> im_templates = compute_templates(image_gray);

    std::vector<cv::Mat> im_templates_temp(im_templates.size());
    for(int i=0; i<im_templates.size(); i++)
    {
        im_templates_temp[i] = im_templates[i][0](cv::Rect(this->patch_size, 
                                                           this->patch_size, 
                                                           image_width, 
                                                           image_height));
    }
    cv::Mat templates_vstack;
    cv::vconcat(im_templates_temp.data(), im_templates_temp.size(), templates_vstack);

    float* pcorner_measures = first_corner_measures(reinterpret_cast<float *>(templates_vstack.data),
                                                    image_width,
                                                    image_height,
                                                    this->directions_n,
                                                    this->patch_size,
                                                    this->eps);
    cv::Mat corner_measure_cuda_mat(image_height, image_width, CV_32F, pcorner_measures);

    cv::Mat points_of_interest = nonma(corner_measure_cuda_mat, this->threshold, this->nonma_radius);

    for(size_t sigma_idx=1; sigma_idx < this->sigmas.size(); sigma_idx++)
    {
        std::vector<cv::Point> points_of_interest_filtered;
        size_t points_of_interest_len = points_of_interest.total();
        for(size_t point_idx=0; point_idx<points_of_interest_len; point_idx++)
        {
            cv::Point point = points_of_interest.at<cv::Point>(point_idx);
            int i = point.y, j = point.x;
            int y = i + this->patch_size - 3;
            int x = j + this->patch_size - 3;

            cv::Rect roi(x, y, this->patch_size, this->patch_size);
            cv::Mat templates_slice(this->directions_n, this->mask_len, CV_32F);

            for(size_t d=0; d < this->directions_n; d++)
            {
                cv::Mat im_template = im_templates[d][sigma_idx](roi);
                for(size_t mask_i=0; mask_i < this->mask_len; mask_i++)
                {
                    cv::Point point = this->mask_indexes.at<cv::Point>(mask_i);
                    templates_slice.at<float>(d,mask_i) = im_template.at<float>(point);
                }
            }

            cv::Mat template_symmetric(this->directions_n, this->directions_n, CV_32F);
            //NOTE this matrix is symmetric, thus it has real eigenvalues and eigenvectors
            cv::mulTransposed(templates_slice, template_symmetric, false); // templates_slice * templates_slice.T
            //NOTE approximation of: product of eigenvalues / sum of eigenvalues
            float measure = cv::determinant(template_symmetric) / (cv::trace(template_symmetric)[0] + this->eps);   
            if (measure > this->threshold)
            {
                points_of_interest_filtered.push_back(point);
            }
        }
        points_of_interest = cv::Mat(points_of_interest_filtered, true);
    }        

    return points_of_interest;
}

int main(int argc, const char **argv)
{
    init_cuda_device(argc, argv);

    size_t num_iters = 1;
    std::string image_path = "../data/17.bmp";

    if(argc >= 2)
    {
        num_iters = std::stoi(argv[1]);
        std::cout << "[INFO] Iter number provided: " << num_iters << "\n";
    }
    if(argc >= 3)
    {
        image_path = argv[2];
    }

    cv::Mat img = cv::imread(image_path);
    cv::Mat points_of_interest;

    float rho = 1.5, threshold = pow(10, 8.4);
    std::vector<float> sigmas = {1.5, 3.0, 4.5};
    int directions_n = 8;

    FOAGDD foagdd(directions_n, sigmas, rho, threshold);

    float time_taken = 0.f;
    for(size_t i=0; i<num_iters; i++)
    {
        auto start = std::chrono::steady_clock::now();
        points_of_interest = foagdd.find_features(img);
        auto end = std::chrono::steady_clock::now();
        if(i > 0)
            time_taken += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }
    if(num_iters != 1)
    {
        num_iters -= 1.f;
    }

    std::cout << "Average elapsed time in milliseconds: " << time_taken/(num_iters) << " ms\n";
    std::cout << "Points of interest found: " << points_of_interest.size() << "\n";

    for(size_t point_idx=0; point_idx < points_of_interest.total(); point_idx++)
    {
        cv::Point point = points_of_interest.at<cv::Point>(point_idx);
        cv::drawMarker(img, point, cv::Scalar(0,0,255), cv::MARKER_SQUARE, 2, 1, cv::LINE_AA);
    }

    cv::imwrite("../data/result.jpg", img);
}
