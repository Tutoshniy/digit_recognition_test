#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <opencv2/opencv.hpp>
#include <vector>

class ImageProcessor {
public:
    static cv::Mat preprocess_image(const cv::Mat& image);
    static cv::Mat create_target_vector(int digit, int num_classes = 10);
    
    // Генерация тестовых данных (вместо реального MNIST)
    static std::vector<cv::Mat> generate_test_images(int count);
    static std::vector<cv::Mat> generate_test_targets(int count);
};

#endif