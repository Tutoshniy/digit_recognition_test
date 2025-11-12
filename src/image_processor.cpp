#include "image_processor.h"
#include <random>
#include <iostream>
#include <cmath>
#include <algorithm>

cv::Mat ImageProcessor::preprocess_image(const cv::Mat& image) {
    cv::Mat processed;
    
    if (image.channels() > 1) {
        cv::cvtColor(image, processed, cv::COLOR_BGR2GRAY);
    } else {
        processed = image.clone();
    }
    
    cv::resize(processed, processed, cv::Size(28, 28));
    processed.convertTo(processed, CV_64F, 1.0 / 255.0);
    
    return processed.reshape(1, 1);
}

cv::Mat ImageProcessor::create_target_vector(int digit, int num_classes) {
    cv::Mat target = cv::Mat::zeros(num_classes, 1, CV_64F);
    target.at<double>(digit, 0) = 1.0;
    return target;
}

std::vector<cv::Mat> ImageProcessor::generate_test_images(int count) {
    std::vector<cv::Mat> images;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> noise(0.0, 0.15);
    std::uniform_real_distribution<> variation(0.7, 1.0);
    
    std::cout << "Generating LARGE dataset with " << count << " samples..." << std::endl;
    
    // Создаем много вариаций каждой цифры
    int samples_per_digit = count / 10;
    
    for (int digit = 0; digit < 10; ++digit) {
        for (int variant = 0; variant < samples_per_digit; ++variant) {
            cv::Mat image = cv::Mat::zeros(1, 784, CV_64F);
            double scale = variation(gen);
            double offset_x = (variation(gen) - 0.5) * 4;
            double offset_y = (variation(gen) - 0.5) * 4;
            
            for (int j = 0; j < 784; ++j) {
                int row = j / 28;
                int col = j % 28;
                double val = 0.0;
                
                // С учетом смещения и масштаба
                double centered_row = (row - 14 + offset_y) / scale;
                double centered_col = (col - 14 + offset_x) / scale;
                
                switch (digit) {
                    case 0: // Круг
                        if (centered_row*centered_row + centered_col*centered_col < 100) 
                            val = 1.0;
                        break;
                    case 1: // Вертикальная линия
                        if (abs(centered_col) < 3 && centered_row > -10 && centered_row < 10) 
                            val = 1.0;
                        break;
                    case 2: // Двойка
                        if ((centered_row > -9 && centered_row < -5) || 
                            (centered_row > -2 && centered_row < 2) ||
                            (centered_row > 5 && centered_row < 9)) val = 1.0;
                        if (centered_row >= -5 && centered_row <= -2 && centered_col > 4) val = 1.0;
                        if (centered_row >= 2 && centered_row <= 5 && centered_col < -4) val = 1.0;
                        break;
                    case 3: // Тройка
                        if ((centered_row > -9 && centered_row < -5) || 
                            (centered_row > -2 && centered_row < 2) ||
                            (centered_row > 5 && centered_row < 9)) val = 1.0;
                        if (centered_col > 4 && centered_row > -5 && centered_row < 5) val = 1.0;
                        break;
                    case 4: // Четверка
                        if (centered_col > 4 && centered_row < 0) val = 1.0;
                        if (centered_row > -2 && centered_row < 2) val = 1.0;
                        if (centered_col < -4 && centered_row > -2) val = 1.0;
                        break;
                    case 5: // Пятерка
                        if ((centered_row > -9 && centered_row < -5) || 
                            (centered_row > -2 && centered_row < 2) ||
                            (centered_row > 5 && centered_row < 9)) val = 1.0;
                        if (centered_row >= -5 && centered_row <= -2 && centered_col < -4) val = 1.0;
                        if (centered_row >= 2 && centered_row <= 5 && centered_col > 4) val = 1.0;
                        break;
                    case 6: // Шестерка
                        if ((centered_row > -9 && centered_row < -5) || 
                            (centered_row > -2 && centered_row < 2) ||
                            (centered_row > 5 && centered_row < 9)) val = 1.0;
                        if (centered_col < -4 && centered_row > -5 && centered_row < 5) val = 1.0;
                        if (centered_row >= 2 && centered_row <= 5 && centered_col > 4) val = 1.0;
                        break;
                    case 7: // Семерка
                        if (centered_row > -9 && centered_row < -5) val = 1.0;
                        if (centered_col > 4 && centered_row > -5) val = 1.0;
                        break;
                    case 8: // Восьмерка
                        if ((centered_row+5)*(centered_row+5) + centered_col*centered_col < 30) val = 1.0;
                        if ((centered_row-5)*(centered_row-5) + centered_col*centered_col < 30) val = 1.0;
                        break;
                    case 9: // Девятка
                        if ((centered_row > -9 && centered_row < -5) || 
                            (centered_row > -2 && centered_row < 2) ||
                            (centered_row > 5 && centered_row < 9)) val = 1.0;
                        if (centered_col > 4 && centered_row > -5 && centered_row < 5) val = 1.0;
                        if (centered_row >= -5 && centered_row <= -2 && centered_col < -4) val = 1.0;
                        break;
                }
                
                val = std::min(1.0, val + noise(gen));
                image.at<double>(0, j) = val;
            }
            
            images.push_back(image);
        }
    }
    
    // Перемешиваем данные
    std::shuffle(images.begin(), images.end(), gen);
    
    std::cout << "Generated " << images.size() << " training samples with variations" << std::endl;
    return images;
}

std::vector<cv::Mat> ImageProcessor::generate_test_targets(int count) {
    std::vector<cv::Mat> targets;
    
    int samples_per_digit = count / 10;
    
    for (int digit = 0; digit < 10; ++digit) {
        for (int variant = 0; variant < samples_per_digit; ++variant) {
            targets.push_back(create_target_vector(digit));
        }
    }
    
    // Перемешиваем цели
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(targets.begin(), targets.end(), gen);
    
    return targets;
}