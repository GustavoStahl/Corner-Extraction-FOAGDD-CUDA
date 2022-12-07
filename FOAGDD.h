#ifndef FOGGDD_H
#define FOGGDD_H

#include <stdio.h>
#include <math.h>
#include <chrono>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>
#include <arrayfire.h>

class FOAGDD
{
    public:
    FOAGDD(size_t directions_n, std::vector<float> sigmas, float rho, float threshold);
    cv::Mat find_features(const cv::Mat &image);

    private:

    size_t width, height;

    int lattice_size = 31; // consider the origin in the lattice
    int patch_size = 7;
    int nonma_radius = 5;
    float rho, eps = 2.22e-16;

    float threshold;
    std::vector<float> sigmas;
    size_t directions_n;
    size_t mask_len;
    cv::Mat mask_indexes;

    std::vector<std::vector<cv::Mat>> im_filters;
    af::array im_filters_gpu;

    std::vector<std::vector<cv::Mat>> compute_filters(int directions_n, 
                                                      std::vector<float> sigmas, 
                                                      float rho, 
                                                      int lattice_size);
    cv::Mat nonma(cv::Mat cim, float threshold, size_t radius);
    std::vector<std::vector<cv::Mat>> compute_templates(cv::Mat& im_gray);
    cv::Mat compute_noncorner_coords(int fingerprint_size);
    void preallocate_gpu_mem(size_t width, size_t height);
};

#endif