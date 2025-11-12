#include "neural_network.h"
#include <random>
#include <iostream>
#include <fstream>
#include <cmath>

NeuralNetwork::NeuralNetwork(const std::vector<int>& layers, double lr) 
    : layer_sizes(layers), learning_rate(lr) {
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> d(-0.5, 0.5);
    
    for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
        cv::Mat w(layer_sizes[i], layer_sizes[i + 1], CV_64F);
        cv::Mat b(1, layer_sizes[i + 1], CV_64F);
        
        for (int r = 0; r < w.rows; ++r) {
            for (int c = 0; c < w.cols; ++c) {
                w.at<double>(r, c) = d(gen);
            }
        }
        
        for (int c = 0; c < b.cols; ++c) {
            b.at<double>(0, c) = 0.1;
        }
        
        weights.push_back(w);
        biases.push_back(b);
    }
}

cv::Mat NeuralNetwork::sigmoid(const cv::Mat& x) {
    cv::Mat result;
    cv::exp(-x, result);
    return 1.0 / (1.0 + result);
}

cv::Mat NeuralNetwork::sigmoid_derivative(const cv::Mat& x) {
    cv::Mat s = sigmoid(x);
    return s.mul(1 - s);
}

cv::Mat NeuralNetwork::softmax(const cv::Mat& x) {
    double maxVal;
    cv::minMaxLoc(x, nullptr, &maxVal);
    
    cv::Mat exp_x;
    cv::exp(x - maxVal, exp_x);
    
    cv::Mat sum;
    cv::reduce(exp_x, sum, 1, cv::REDUCE_SUM);
    
    cv::Mat result(x.rows, x.cols, CV_64F);
    for (int i = 0; i < x.rows; ++i) {
        double row_sum = sum.at<double>(i, 0);
        for (int j = 0; j < x.cols; ++j) {
            result.at<double>(i, j) = exp_x.at<double>(i, j) / row_sum;
        }
    }
    
    return result;
}

std::vector<cv::Mat> NeuralNetwork::forward(const cv::Mat& input) {
    std::vector<cv::Mat> activations;
    cv::Mat current = input.clone();
    activations.push_back(current);
    
    for (size_t i = 0; i < weights.size(); ++i) {
        cv::Mat z = current * weights[i] + biases[i];
        
        if (i == weights.size() - 1) {
            current = softmax(z);
        } else {
            current = sigmoid(z);
        }
        
        activations.push_back(current);
    }
    
    return activations;
}

void NeuralNetwork::backward(const cv::Mat& input, const cv::Mat& target) {
    auto activations = forward(input);
    std::vector<cv::Mat> deltas(weights.size());
    
    cv::Mat output = activations.back();
    cv::Mat target_t = target.t();
    cv::Mat output_error = output - target_t;
    deltas.back() = output_error;
    
    for (int i = weights.size() - 2; i >= 0; --i) {
        cv::Mat error = deltas[i + 1] * weights[i + 1].t();
        cv::Mat sp = sigmoid_derivative(activations[i + 1]);
        deltas[i] = error.mul(sp);
    }
    
    for (size_t i = 0; i < weights.size(); ++i) {
        cv::Mat dw = activations[i].t() * deltas[i];
        weights[i] -= learning_rate * dw;
        biases[i] -= learning_rate * deltas[i];
    }
}

int NeuralNetwork::predict(const cv::Mat& input) {
    try {
        auto activations = forward(input);
        cv::Mat output = activations.back();
        
        cv::Point max_loc;
        cv::minMaxLoc(output, nullptr, nullptr, nullptr, &max_loc);
        return max_loc.x;
        
    } catch (const std::exception& e) {
        std::cerr << "Prediction error: " << e.what() << std::endl;
        return -1;
    }
}

double NeuralNetwork::train(const std::vector<cv::Mat>& inputs, 
                           const std::vector<cv::Mat>& targets, 
                           int epochs, int batch_size) {
    
    if (inputs.empty() || targets.empty()) {
        std::cerr << "Error: No training data!" << std::endl;
        return 0.0;
    }
    
    std::cout << "Training: " << inputs.size() << " samples, " << epochs << " epochs" << std::endl;
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        int correct = 0;
        
        for (size_t i = 0; i < inputs.size(); ++i) {
            try {
                backward(inputs[i], targets[i]);
                
                int prediction = predict(inputs[i]);
                int actual = -1;
                for (int j = 0; j < targets[i].rows; ++j) {
                    if (targets[i].at<double>(j, 0) > 0.5) {
                        actual = j;
                        break;
                    }
                }
                
                if (prediction == actual && actual != -1) {
                    correct++;
                }
                
                auto activations = forward(inputs[i]);
                cv::Mat output = activations.back();
                cv::Mat log_output;
                cv::log(output + 1e-8, log_output);
                cv::Mat loss_mat = -targets[i].t() * log_output.t();
                total_loss += loss_mat.at<double>(0, 0);
                
            } catch (const std::exception& e) {
                continue;
            }
        }
        
        double accuracy = static_cast<double>(correct) / inputs.size();
        double avg_loss = total_loss / inputs.size();
        
        std::cout << "Epoch " << (epoch + 1) << "/" << epochs 
                  << " - Loss: " << avg_loss 
                  << " - Accuracy: " << (accuracy * 100) << "%" << std::endl;
    }
    
    return 0.0;
}

void NeuralNetwork::save_model(const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    
    fs << "layer_sizes" << layer_sizes;
    fs << "learning_rate" << learning_rate;
    
    for (size_t i = 0; i < weights.size(); ++i) {
        fs << "weight_" + std::to_string(i) << weights[i];
        fs << "bias_" + std::to_string(i) << biases[i];
    }
    
    fs.release();
    std::cout << "Model saved: " << filename << std::endl;
}

void NeuralNetwork::load_model(const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    
    cv::FileNode node = fs["layer_sizes"];
    if (node.isSeq()) {
        layer_sizes.clear();
        for (auto it = node.begin(); it != node.end(); ++it) {
            layer_sizes.push_back((int)*it);
        }
    }
    
    fs["learning_rate"] >> learning_rate;
    
    weights.clear();
    biases.clear();
    
    for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
        cv::Mat w, b;
        fs["weight_" + std::to_string(i)] >> w;
        fs["bias_" + std::to_string(i)] >> b;
        
        weights.push_back(w);
        biases.push_back(b);
    }
    
    fs.release();
    std::cout << "Model loaded: " << filename << std::endl;
}