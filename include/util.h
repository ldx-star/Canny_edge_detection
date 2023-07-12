//
// Created by liangdaxin on 23-6-20.
//

#ifndef CANNY_EDGE_DETECTION_UTIL_H
#define CANNY_EDGE_DETECTION_UTIL_H
#include <opencv2/opencv.hpp>
#include <math.h>

class util {
public:
    static void Sobel(const cv::Mat& img, cv::Mat& x_derivative, cv::Mat& y_derivative);
    static void Prewitt(const cv::Mat& img, cv::Mat& x_derivative, cv::Mat& y_derivative);
    static void get_gradient(const cv::Mat& img, cv::Mat& magnitude, cv::Mat& direction );
    static void gaussian_kernel(const int& width, cv::Mat& out_mat);
    static void img_normal(cv::Mat& img);
    static void img_show(cv::Mat img,cv::Mat& out_img);

};


#endif //CANNY_EDGE_DETECTION_UTIL_H
