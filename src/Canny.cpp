//
// Created by liangdaxin on 23-6-20.
//

#include "../include/Canny.h"
#include "../include/util.h"

Canny::Canny(const int &gaussian_kernel_width, cv::Mat &img, const float &high_threshold,
             const float &low_threshold)
        : gaussian_kernel_width_(gaussian_kernel_width), img_(img),
          high_threshold_(high_threshold), low_threshold_(low_threshold) {
    assert(img.data != nullptr);
}

void Canny::Gaussian_blur() {
    cv::Mat gaussian_kernel;
    util::gaussian_kernel(gaussian_kernel_width_, gaussian_kernel);
    Gaussian_img_ = cv::Mat::zeros(img_.size(), CV_8U);
    cv::filter2D(img_, Gaussian_img_, CV_8U, gaussian_kernel);
}

void Canny::Get_gradient_img() {
    util::get_gradient(Gaussian_img_, magnitude_, direction_);
}

void Canny::Non_maximum_suppression() {
    //沿着梯度方向进行非极大值抑制
    int height = magnitude_.rows;
    int width = magnitude_.cols;
    const cv::Mat &magnitude = magnitude_;
    const cv::Mat &angle = direction_;
    cv::Mat result = cv::Mat::zeros(img_.size(), CV_32F);
    for (int i = 1; i < height; i++) {
        for (int j = 1; j < width; j++) {
            float g1, g2, g3, g4;
            bool type1 = false, type2 = false, type3 = false, type4 = false;
            if (abs(float(magnitude.at<uint8_t>(i, j))) <= 4) {
                result.at<float>(i, j) = 0;
            } else {
                //看角度
                if (angle.at<float>(i, j) >= 0 && angle.at<float>(i, j) <= M_PI / 4 &&
                    angle.at<float>(i, j) <= -M_PI * 3 / 4 && angle.at<float>(i, j) >= -M_PI) {
                    /*
                     *                g3
                     *      g2    c   g4
                     *      g1
                     */
                    g1 = magnitude.at<uint8_t>(i + 1, j - 1);
                    g2 = magnitude.at<uint8_t>(i, j - 1);
                    g3 = magnitude.at<uint8_t>(i - 1, j + 1);
                    g4 = magnitude.at<uint8_t>(i, j + 1);
                    type1 = true;
                } else if (angle.at<float>(i, j) >= M_PI * 3 / 4 && angle.at<float>(i, j) <= M_PI &&
                           angle.at<float>(i, j) <= 0 && angle.at<float>(i, j) >= -M_PI / 4) {
                    /*
                     *      g1
                     *      g2    c   g4
                     *                g3
                     */
                    g1 = magnitude.at<uint8_t>(i - 1, j - 1);
                    g2 = magnitude.at<uint8_t>(i, j - 1);
                    g3 = magnitude.at<uint8_t>(i + 1, j + 1);
                    g4 = magnitude.at<uint8_t>(i, j + 1);
                    type2 = true;
                } else if (angle.at<float>(i, j) >= M_PI / 4 && angle.at<float>(i, j) <= M_PI / 2 &&
                           angle.at<float>(i, j) >= -M_PI * 3 / 4 && angle.at<float>(i, j) <= -M_PI / 2) {
                    /*
                     *           g2  g1
                     *           c
                     *      g3   g4
                     */
                    g1 = magnitude.at<uint8_t>(i - 1, j + 1);
                    g2 = magnitude.at<uint8_t>(i - 1, j);
                    g3 = magnitude.at<uint8_t>(i + 1, j - 1);
                    g4 = magnitude.at<uint8_t>(i + 1, j);
                    type3 = true;
                } else {
                    /*
                *           g1  g2
                    *           c
                    *           g4  g3
                    */
                    g1 = magnitude.at<uint8_t>(i - 1, j - 1);
                    g2 = magnitude.at<uint8_t>(i - 1, j);
                    g3 = magnitude.at<uint8_t>(i + 1, j + 1);
                    g4 = magnitude.at<uint8_t>(i + 1, j + 1);
                    type4 = true;
                }
            }
            float temp1, temp2;
            if (type1 || type2) {
                temp1 = abs(angle.at<float>(i, j) / M_PI) * g1 + (1 - abs(angle.at<float>(i, j) / M_PI)) * g2;
                temp2 = abs(angle.at<float>(i, j) / M_PI) * g3 + (1 - abs(angle.at<float>(i, j) / M_PI)) * g4;
            } else {
                temp1 = abs(angle.at<float>(i, j) / M_PI) * g2 + (1 - abs(angle.at<float>(i, j) / M_PI)) * g1;
                temp2 = abs(angle.at<float>(i, j) / M_PI) * g4 + (1 - abs(angle.at<float>(i, j) / M_PI)) * g3;
            }

            if (magnitude.at<uint8_t>(i, j) >= temp1 && magnitude.at<uint8_t>(i, j) >= temp2) {
                result.at<float>(i, j) = magnitude.at<uint8_t>(i, j);
            } else {
                result.at<float>(i, j) = 0;
            }
        }
    }
    img_ = result.clone();
}

void Canny::Hysteresis_thresholding() {
    // 沿梯度的垂直方向，进行滞后阈值法，用强边延伸弱边
    int height = img_.rows;
    int width = img_.cols;
    auto origin = img_;
    auto angle = direction_;
    cv::Mat result = cv::Mat::zeros(img_.size(), CV_32F);
    auto high_threshold = high_threshold_;
    auto low_threshold = low_threshold_;
    for (int i = 1; i < height; i++) {
        for (int j = 1; j < width; j++) {
            float *g1, *g2, *g3, *g4;
            if (origin.at<float>(i, j) >= high_threshold) {
                result.at<float>(i, j) = origin.at<float>(i, j);
                //根据角度调整低于low_threshold的值
                if (angle.at<float>(i, j) >= 0 && angle.at<float>(i, j) <= M_PI / 4 &&
                    angle.at<float>(i, j) <= -M_PI * 3 / 4 && angle.at<float>(i, j) >= -M_PI) {
                    /*
                     *     g1   g2
                     *          c
                     *          g4   g3
                     */
                    g1 = &origin.at<float>(i - 1, j - 1);
                    g2 = &origin.at<float>(i - 1, j);
                    g3 = &origin.at<float>(i + 1, j + 1);
                    g4 = &origin.at<float>(i + 1, j);
                } else if (angle.at<float>(i, j) >= M_PI * 3 / 4 && angle.at<float>(i, j) <= M_PI &&
                           angle.at<float>(i, j) <= 0 && angle.at<float>(i, j) >= -M_PI / 4) {
                    /*
                      *         g2 g1
                      *         c
                      *      g3 g4
                      */
                    g1 = &origin.at<float>(i - 1, j + 1);
                    g2 = &origin.at<float>(i - 1, j);
                    g3 = &origin.at<float>(i + 1, j - 1);
                    g4 = &origin.at<float>(i + 1, j);
                } else if (angle.at<float>(i, j) >= M_PI / 4 && angle.at<float>(i, j) <= M_PI / 2 &&
                           angle.at<float>(i, j) >= -M_PI * 3 / 4 && angle.at<float>(i, j) <= -M_PI / 2) {
                    /*
                     *         g1
                     *         g2 c g4
                     *              g3
                     */
                    g1 = &origin.at<float>(i - 1, j - 1);
                    g2 = &origin.at<float>(i, j - 1);
                    g3 = &origin.at<float>(i + 1, j + 1);
                    g4 = &origin.at<float>(i, j + 1);
                } else {
                    /*
                *                   g3
                    *       g2  c   g4
                    *       g1
                    */
                    g1 = &origin.at<float>(i + 1, j - 1);
                    g2 = &origin.at<float>(i, j - 1);
                    g3 = &origin.at<float>(i - 1, j + 1);
                    g4 = &origin.at<float>(i, j + 1);
                }
            }
            if(g1 && *g1 > low_threshold && *g1 < high_threshold){
                *g1 = high_threshold;
            }else if(g2 && *g2 > low_threshold && *g2 < high_threshold){
                *g2 = high_threshold;
            }else if(g3 && *g3 > low_threshold && *g3 < high_threshold){
                *g3 = high_threshold;
            }else if(g4 && *g1 > low_threshold && *g1 < high_threshold){
                *g4 = high_threshold;
            }
        }
    }
    int count = 0;
    for(int i = 0 ;i < height;i++){
        for(int j = 0; j <width;j++){
            if(result.at<float>(i,j) != 0){
                count++;
            }
        }
    }
    std::cout << count << std::endl;
    img_ = result.clone();
}

void Canny::Canny_edge_detect() {
    Gaussian_blur();
    Get_gradient_img();
    Non_maximum_suppression();
    Hysteresis_thresholding();
    util::img_normal(img_);
    img_ *= 255;
    img_.convertTo(img_,CV_8U);
}