#include <stdio.h>
#include <math.h>
#include <chrono>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>
#include <arrayfire.h>

extern "C" int init_cuda_device(int argc, const char **argv);
extern "C" float* first_corner_measures(float *im_templates, size_t width, size_t height, size_t directions_n, size_t patch_size, float eps);

cv::Mat nonma(cv::Mat cim, float threshold, size_t radius)
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

std::vector<std::vector<cv::Mat>> compute_filters(int directions_n, 
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

std::vector<std::vector<cv::Mat>> compute_templates(cv::Mat& im_gray, int patch_size, std::vector<std::vector<cv::Mat>> im_filters)
{
    cv::Mat im_padded;
    cv::copyMakeBorder(im_gray, im_padded, patch_size, patch_size, patch_size, patch_size, cv::BORDER_REFLECT);

    size_t directions_n = im_filters.size(), sigmas_n = im_filters[0].size(), filter_size = im_filters[0][0].rows;

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

    af::array im_padded_gpu(im_padded.cols, im_padded.rows, reinterpret_cast<float*>(im_padded.data));
    af::transposeInPlace(im_padded_gpu);

    af::array im_filters_gpu(filter_size, filter_size, directions_n * sigmas_n, reinterpret_cast<float*>(im_filters_concat.data));
    af::transposeInPlace(im_filters_gpu);

    af::array im_templates_gpu = af::abs(af::convolve2(im_padded_gpu, im_filters_gpu));
    af::transposeInPlace(im_templates_gpu);
    
    float* pim_templates = im_templates_gpu.host<float>();

    std::vector<std::vector<cv::Mat>> im_templates(directions_n, std::vector<cv::Mat>(sigmas_n));
    for(size_t direction_idx=0; direction_idx < directions_n; direction_idx++)
    {
        for(size_t sigma_idx=0; sigma_idx < sigmas_n; sigma_idx++)
        {
            size_t shift = (direction_idx * sigmas_n + sigma_idx) * im_padded.rows * im_padded.cols;            
            im_templates[direction_idx][sigma_idx] = cv::Mat(im_padded.rows, im_padded.cols, CV_32F, pim_templates + shift);
        }
    }

    return im_templates;
}

// std::vector<std::vector<cv::Mat>> compute_templates(cv::Mat& im_gray, int patch_size, std::vector<std::vector<cv::Mat>> im_filters)
// {
//     cv::Mat im_padded;
//     cv::copyMakeBorder(im_gray, im_padded, patch_size, patch_size, patch_size, patch_size, cv::BORDER_REFLECT);

//     int im_padded_step;
//     float* im_padded_gpu = set_filter_src_image(reinterpret_cast<float*>(im_padded.data), 
//                                                 im_padded.cols, 
//                                                 im_padded.rows,
//                                                 im_padded_step);

//     size_t directions_n = im_filters.size(), sigmas_n = im_filters[0].size();

//     std::vector<std::vector<cv::Mat>> im_templates(directions_n, std::vector<cv::Mat>(sigmas_n));

//     std::vector<float*> conv_filters(directions_n * sigmas_n);

//     for(size_t direction_idx=0; direction_idx < directions_n; direction_idx++)
//     {
//         for(size_t sigma_idx=0; sigma_idx < sigmas_n; sigma_idx++)
//         {
//             conv_filters[direction_idx*sigmas_n+sigma_idx] = reinterpret_cast<float*>(im_filters[direction_idx][sigma_idx].data);
//         }
//     }

//     // auto start = std::chrono::steady_clock::now();
//     std::vector<float*> im_template_ptr = compute_templates(im_padded_gpu,
//                                                             im_padded_step,
//                                                             im_padded.cols, 
//                                                             im_padded.rows, 
//                                                             conv_filters,
//                                                             im_filters[0][0].rows);
//         // auto end = std::chrono::steady_clock::now();
//         // std::cout << "template CUDA: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
//     for(size_t direction_idx=0; direction_idx < directions_n; direction_idx++)
//     {     
//         for(size_t sigma_idx=0; sigma_idx < sigmas_n; sigma_idx++)
//         {
//             cv::Mat im_template = cv::Mat(im_padded.rows, im_padded.cols, CV_32F, im_template_ptr[direction_idx*sigmas_n + sigma_idx]);
//             cv::absdiff(im_template, cv::Scalar::all(0), im_template);    

//             im_templates[direction_idx][sigma_idx] = im_template; 
//         }
//     }

//     return im_templates;
// }

cv::Mat foggdd(const cv::Mat &img)
{
    float rho = 1.5, eps = 2.22e-16, threshold = pow(10, 8.4);
    std::vector<float> sigmas = {1.5, 3.0, 4.5};
    int directions_n = 8, nonma_radius = 5;
    int lattice_size = 31; // consider the origin in the lattice

    cv::Mat img_gray;
    if(img.channels() != 1)
        cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    else
        img.copyTo(img_gray);

    img_gray.convertTo(img_gray, CV_32F);

    int rows = img_gray.rows;
    int cols = img_gray.cols;

    int patch_size = 7;

    auto start = std::chrono::steady_clock::now();
    std::vector<std::vector<cv::Mat>> im_filters = compute_filters(directions_n,sigmas,rho,lattice_size);
    auto end = std::chrono::steady_clock::now();
    std::cout << "Compute filters: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    start = std::chrono::steady_clock::now();
    std::vector<std::vector<cv::Mat>> im_templates = compute_templates(img_gray, patch_size, im_filters);
    end = std::chrono::steady_clock::now();
    std::cout << "Compute templates: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    start = std::chrono::steady_clock::now();
    std::vector<cv::Mat> im_templates_temp(im_templates.size());
    for(int i=0; i<im_templates.size(); i++)
    {
        im_templates_temp[i] = im_templates[i][0](cv::Rect(patch_size, patch_size, rows, cols));
    }
    cv::Mat templates_vstack;
    cv::vconcat(im_templates_temp.data(), im_templates_temp.size(), templates_vstack);
    end = std::chrono::steady_clock::now();
    std::cout << "Create CUDA input: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    start = std::chrono::steady_clock::now();
    float *corner_measure_ptr = first_corner_measures(reinterpret_cast<float *>(templates_vstack.data), 
                                                      (size_t)cols, 
                                                      (size_t)rows, 
                                                      (size_t)directions_n, 
                                                      (size_t)patch_size, 
                                                      eps);
    end = std::chrono::steady_clock::now();
    std::cout << "CUDA: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    cv::Mat corner_measure_cuda_mat(rows, cols, CV_32F, corner_measure_ptr);
    cv::Mat points_of_interest = nonma(corner_measure_cuda_mat, threshold, nonma_radius);
    std::cout << "Points of interest found in CUDA array: " << points_of_interest.size() << "\n";

    start = std::chrono::steady_clock::now();
    cv::Mat mask = cv::Mat::ones(patch_size, patch_size, CV_8U), mask_indexes;
    mask.at<unsigned char>(0,0) = 0;
    mask.at<unsigned char>(0,1) = 0;
    mask.at<unsigned char>(0,patch_size-1) = 0;
    mask.at<unsigned char>(0,patch_size-2) = 0;
    mask.at<unsigned char>(1,0) = 0;
    mask.at<unsigned char>(1,patch_size-1) = 0;
    mask.at<unsigned char>(patch_size-2,0) = 0;
    mask.at<unsigned char>(patch_size-2,patch_size-1) = 0;
    mask.at<unsigned char>(patch_size-1,0) = 0;
    mask.at<unsigned char>(patch_size-1,1) = 0;
    mask.at<unsigned char>(patch_size-1,patch_size-1) = 0;
    mask.at<unsigned char>(patch_size-1,patch_size-2) = 0;
    cv::findNonZero(mask, mask_indexes);
    size_t mask_len = mask_indexes.total();
    end = std::chrono::steady_clock::now();
    std::cout << "Create mask: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

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
            cv::Mat templates_slice(directions_n, mask_len, CV_32F);
            for(size_t d=0; d<directions_n; d++)
            {
                cv::Mat im_template = im_templates[d][sigma_idx](roi);
                for(size_t mask_i=0; mask_i < mask_len; mask_i++)
                {
                    cv::Point point = mask_indexes.at<cv::Point>(mask_i);
                    templates_slice.at<float>(d,mask_i) = im_template.at<float>(point);
                }
            }
            cv::Mat template_symmetric(directions_n, directions_n, CV_32F);
            //NOTE this matrix is symmetric, thus it has real eigenvalues and eigenvectors
            cv::mulTransposed(templates_slice, template_symmetric, false); // templates_slice * templates_slice.T
            //NOTE approximation of: product of eigenvalues / sum of eigenvalues
            float measure = cv::determinant(template_symmetric) / (cv::trace(template_symmetric)[0] + eps);   
            if (measure > threshold)
            {
                points_of_interest_filtered.push_back(point);
            }
        }
        points_of_interest = cv::Mat(points_of_interest_filtered, true);
    }    
    end = std::chrono::steady_clock::now();
    std::cout << "Iter through scales: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    start = std::chrono::steady_clock::now();
    std::vector<int> output; 
    for(size_t point_idx=0; point_idx < points_of_interest.total(); point_idx++)
    {
        cv::Point point = points_of_interest.at<cv::Point>(point_idx);
        output.push_back(point.y);
        output.push_back(point.x);
    }
    end = std::chrono::steady_clock::now();
    std::cout << "Create output: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    return points_of_interest;
}

int main(int argc, const char **argv)
{
    init_cuda_device(argc, argv);

    cv::Mat img = cv::imread("../data/17.bmp");
    cv::Mat points_of_interest;

    for(size_t i=0; i<5; i++)
    {
        auto start = std::chrono::steady_clock::now();
        points_of_interest = foggdd(img);
        auto end = std::chrono::steady_clock::now();
        std::cout << "Elapsed time in milliseconds: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    }

    for(size_t point_idx=0; point_idx < points_of_interest.total(); point_idx++)
    {
        cv::Point point = points_of_interest.at<cv::Point>(point_idx);
        cv::drawMarker(img, point, cv::Scalar(0,0,255), cv::MARKER_SQUARE, 2, 1, cv::LINE_AA);
    }

    cv::imwrite("../data/result.jpg", img);

    // cv::namedWindow("jonas", 0);
    // cv::imshow("jonas", img);
    // cv::waitKey(0);
    // cv::destroyAllWindows();
}