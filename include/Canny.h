//
// Created by liangdaxin on 23-6-20.
//

#ifndef CANNY_EDGE_DETECTION_CANNY_H
#define CANNY_EDGE_DETECTION_CANNY_H
#include <opencv2/opencv.hpp>

class Canny {
public:
    const int &gaussian_kernel_width_;
    cv::Mat &img_;
    const float &high_threshold_;
    const float &low_threshold_;
public:
    Canny(const int &gaussian_kernel_width,cv::Mat &img, const float &high_threshold, const float &low_threshold);
    void Canny_edge_detect();
private:
    void Gaussian_blur();
    void Get_gradient_img();
    void Non_maximum_suppression();
    void Hysteresis_thresholding();
private:
    cv::Mat Gaussian_img_;
    cv::Mat magnitude_;
    cv::Mat direction_;


};


#endif //CANNY_EDGE_DETECTION_CANNY_H
