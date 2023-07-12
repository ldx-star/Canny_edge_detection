//
// Created by liangdaxin on 23-6-20.
//

#include "../include/util.h"

void util::img_normal(cv::Mat &img) {
    img.convertTo(img, CV_32F);
    int height = img.rows;
    int width = img.cols;
    int max = 0;
    int min = 9999999;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int value = img.at<float>(i, j);
            if(value == 0){
                continue;
            }
            max = max > value ? max : value;
            min = min < value ? min : value;
        }
    }
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if(img.at<float>(i, j) == 0){
                continue;
            }
            img.at<float>(i, j) = (img.at<float>(i, j) - min) / (max - min);
        }
    }
}

void util::Sobel(const cv::Mat &img, cv::Mat &x_derivative, cv::Mat &y_derivative) {
    cv::Mat x_kernel = (cv::Mat_<int8_t>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    cv::Mat y_kernel = (cv::Mat_<int8_t>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
    x_derivative = cv::Mat::zeros(img.size(), CV_8U);
    y_derivative = cv::Mat::zeros(img.size(), CV_8U);
    cv::filter2D(img, x_derivative, CV_8U, x_kernel);
    cv::filter2D(img, y_derivative, CV_8U, y_kernel);
}

void util::Prewitt(const cv::Mat &img, cv::Mat &x_derivative, cv::Mat &y_derivative) {
    cv::Mat x_kernel = (cv::Mat_<int8_t>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
    cv::Mat y_kernel = (cv::Mat_<int8_t>(3, 3) << 1, 1, 1, 0, 0, 0, -1, -1, -1);
    x_derivative = cv::Mat::zeros(img.size(), CV_8U);
    y_derivative = cv::Mat::zeros(img.size(), CV_8U);
    cv::filter2D(img, x_derivative, CV_8U, x_kernel);
    cv::filter2D(img, y_derivative, CV_8U, y_kernel);
}

void util::get_gradient(const cv::Mat &img, cv::Mat &magnitude, cv::Mat &direction) {
    cv::Mat x_kernel = (cv::Mat_<int>(1, 2) << -1, 1);
    cv::Mat y_kernel = (cv::Mat_<int>(2, 1) << -1, 1);
    cv::Mat x_derivative = cv::Mat::zeros(img.size(), CV_32F);
    cv::Mat y_derivative = cv::Mat::zeros(img.size(), CV_32F);

    int height = img.rows;
    int width = img.cols;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (j == 0) {
                x_derivative.at<float>(i, j) = 1;
            } else {
                x_derivative.at<float>(i, j) = img.at<uint8_t>(i, j - 1) * x_kernel.at<int>(0, 0) +
                                               img.at<uint8_t>(i, j) * x_kernel.at<int>(0, 1);
            }
            if (i == 0) {
                y_derivative.at<float>(i, j) = 1;
            } else {
                y_derivative.at<float>(i, j) = img.at<uint8_t>(i - 1, j) * y_kernel.at<int>(0, 0) +
                                               img.at<uint8_t>(i, j) * y_kernel.at<int>(1, 0);
            }
        }
    }
    magnitude = cv::Mat::zeros(img.size(), CV_32F);
    direction = cv::Mat::zeros(img.size(), CV_32F);
    cv::cartToPolar(y_derivative, x_derivative, magnitude, direction);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float x_d = x_derivative.at<float>(i, j);
            float y_d = y_derivative.at<float>(i, j);
            auto a =  sqrt(float(x_d * x_d + y_d * y_d));
            magnitude.at<float>(i, j) = sqrt(float(x_d * x_d + y_d * y_d));
            direction.at<float>(i, j) = atan2(float(y_d), float(x_d));
        }
    }
    magnitude.convertTo(magnitude, CV_8U);
}

void util::gaussian_kernel(const int &width, cv::Mat &out_mat) {
    //sigma =( width - 1) / 2
    int sigma = (width - 1) / 2;
    if (width % 2 == 0) {
        //暂时只接受奇数，后续完善
        std::cout << "width必须为奇数" << std::endl;
        exit(1);
    }
    float sum = 0;
    int num = width / 2;
    out_mat = cv::Mat::zeros(width, width, CV_32F);
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            float value = (1 / (2 * M_PI * std::pow(sigma, 2))) * (std::pow(M_E, -1 * (std::pow(i - num, 2) +
                                                                                       std::pow(j - num, 2)) / (2 *
                                                                                                                std::pow(
                                                                                                                        sigma,
                                                                                                                        2))));
            out_mat.at<float>(i, j) = value;
            sum += value;
        }
    }
    out_mat *= 1 / sum;
}

void util::img_show(cv::Mat img, cv::Mat &out_img) {
    float max = 0;
    float min = 999999;
    int height = img.rows;
    int wight = img.cols;
    out_img = cv::Mat::zeros(img.size(),CV_8U);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < wight; j++) {
            max = max > img.at<float>(i,j) ? max :img.at<float>(i,j);
            min = min < img.at<float>(i,j) ? min : img.at<float>(i,j);
        }
    }
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < wight; j++) {
            auto value = (img.at<float>(i,j) - min) / (max - min);
            out_img.at<uint8_t>(i,j) = (unsigned int)(value * 255);
        }
    }
}