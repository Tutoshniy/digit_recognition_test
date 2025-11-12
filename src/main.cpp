#include <iostream>
#include <opencv2/opencv.hpp>
#include "neural_network.h"
#include "image_processor.h"
#include "drawing_interface.h"

int main() {
    std::cout << "=== LARGE DATASET Digit Recognition ===" << std::endl;
    
    try {
        NeuralNetwork nn({784, 128, 64, 10}, 0.01);
        
        std::cout << "Generating LARGE training dataset..." << std::endl;
        auto images = ImageProcessor::generate_test_images(500);
        auto targets = ImageProcessor::generate_test_targets(500);
        
        std::cout << "Starting training with LARGE dataset..." << std::endl;
        nn.train(images, targets, 70);
        
        // Проверка
        std::cout << "\n=== Final Test ===" << std::endl;
        int correct = 0;
        for (int i = 0; i < 20; i++) {
            int pred = nn.predict(images[i]);
            int actual = 0;
            for (int j = 0; j < targets[i].rows; j++) {
                if (targets[i].at<double>(j, 0) > 0.5) {
                    actual = j;
                    break;
                }
            }
            std::cout << "Digit " << actual << " -> Prediction: " << pred;
            if (pred == actual) {
                std::cout << " ✓";
                correct++;
            }
            std::cout << std::endl;
        }
        std::cout << "Test accuracy: " << (correct * 5) << "%" << std::endl;
        
        nn.save_model("large_dataset_model.xml");
        
        std::cout << "\nStarting drawing interface..." << std::endl;
        DrawingInterface drawingInterface(nn);
        drawingInterface.run();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}