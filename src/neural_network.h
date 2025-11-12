#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& layers, double lr = 0.1);
    
    // Основные методы
    int predict(const cv::Mat& input);
    double train(const std::vector<cv::Mat>& inputs, 
                 const std::vector<cv::Mat>& targets, 
                 int epochs, int batch_size = 32);
    
    // Сохранение и загрузка модели
    void save_model(const std::string& filename);
    void load_model(const std::string& filename);

private:
    std::vector<int> layer_sizes;
    std::vector<cv::Mat> weights;
    std::vector<cv::Mat> biases;
    double learning_rate;
    
    // Вспомогательные методы
    cv::Mat sigmoid(const cv::Mat& x);
    cv::Mat sigmoid_derivative(const cv::Mat& x);
    cv::Mat softmax(const cv::Mat& x);
    std::vector<cv::Mat> forward(const cv::Mat& input);
    void backward(const cv::Mat& input, const cv::Mat& target);
};

#endif