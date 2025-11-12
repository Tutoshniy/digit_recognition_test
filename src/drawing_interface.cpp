#include "drawing_interface.h"
#include <iostream>

DrawingInterface::DrawingInterface(NeuralNetwork& nn) 
    : neuralNetwork(nn), drawing(false) {
    initializeCanvas();
}

void DrawingInterface::initializeCanvas() {
    canvas = cv::Mat::zeros(280, 280, CV_8UC1);
    drawing = false;
}

void DrawingInterface::onMouse(int event, int x, int y, int flags, void* param) {
    DrawingInterface* interface = static_cast<DrawingInterface*>(param);
    
    if (x < 0 || y < 0 || x >= interface->canvas.cols || y >= interface->canvas.rows) {
        return;
    }

    switch (event) {
        case cv::EVENT_LBUTTONDOWN:
            interface->drawing = true;
            interface->lastPoint = cv::Point(x, y);
            cv::circle(interface->canvas, interface->lastPoint, 15, cv::Scalar(255), -1);
            break;
            
        case cv::EVENT_MOUSEMOVE:
            if (interface->drawing) {
                cv::line(interface->canvas, interface->lastPoint, cv::Point(x, y), cv::Scalar(255), 30);
                interface->lastPoint = cv::Point(x, y);
            }
            break;
            
        case cv::EVENT_LBUTTONUP:
            interface->drawing = false;
            break;
            
        case cv::EVENT_RBUTTONDOWN:
            interface->canvas = cv::Mat::zeros(280, 280, CV_8UC1);
            break;
    }
}

void DrawingInterface::mouseCallback(int event, int x, int y, int flags, void* param) {
    DrawingInterface* interface = static_cast<DrawingInterface*>(param);
    interface->onMouse(event, x, y, flags, param);
}

cv::Mat DrawingInterface::preprocessDrawing() {
    try {
        // Масштабируем до 28x28
        cv::Mat resized;
        cv::resize(canvas, resized, cv::Size(28, 28));
        
        // Инвертируем: черный фон, белая цифра -> белый фон, черная цифра
        cv::Mat inverted;
        cv::bitwise_not(resized, inverted);
        
        // Конвертируем в double и нормализуем
        cv::Mat normalized;
        inverted.convertTo(normalized, CV_64F, 1.0 / 255.0);
        
        // Выравниваем в вектор [1 x 784]
        return normalized.reshape(1, 1);
        
    } catch (const cv::Exception& e) {
        std::cerr << "Preprocessing error: " << e.what() << std::endl;
        return cv::Mat::zeros(1, 784, CV_64F);
    }
}

void DrawingInterface::run() {
    // Создаем окно побольше
    cv::namedWindow("Draw Digit - Press Q to quit", cv::WINDOW_NORMAL);
    cv::resizeWindow("Draw Digit - Press Q to quit", 600, 500);
    cv::setMouseCallback("Draw Digit - Press Q to quit", mouseCallback, this);

    std::cout << "=== Drawing Interface ===" << std::endl;
    std::cout << "Left click: Draw" << std::endl;
    std::cout << "Right click: Clear" << std::endl;
    std::cout << "Press 'c': Clear" << std::endl;
    std::cout << "Press 'q': Quit" << std::endl;

    while (true) {
        // Создаем большое изображение для отображения
        cv::Mat display = cv::Mat::zeros(500, 600, CV_8UC3);
        
        // Копируем холст в центр
        cv::Mat colorCanvas;
        cv::cvtColor(canvas, colorCanvas, cv::COLOR_GRAY2BGR);
        cv::Rect roi(160, 60, 280, 280);
        colorCanvas.copyTo(display(roi));
        
        // Добавляем рамку
        cv::rectangle(display, roi, cv::Scalar(0, 255, 0), 2);
        
        // Распознаем
        cv::Mat processed = preprocessDrawing();
        int prediction = -1;
        if (!processed.empty()) {
            prediction = neuralNetwork.predict(processed);
        }
        
        // Отображаем предсказание КРУПНЫМ шрифтом
        std::string predictionText = "Prediction: " + std::to_string(prediction);
        cv::putText(display, predictionText, cv::Point(50, 40), 
                   cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 255, 255), 3);
        
        // Отображаем инструкции
        cv::putText(display, "Instructions:", cv::Point(50, 380), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        cv::putText(display, "Left click: Draw", cv::Point(50, 410), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1);
        cv::putText(display, "Right click: Clear", cv::Point(50, 430), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1);
        cv::putText(display, "Press 'c': Clear canvas", cv::Point(50, 450), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1);
        cv::putText(display, "Press 'q': Quit", cv::Point(50, 470), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1);
        
        cv::imshow("Draw Digit - Press Q to quit", display);
        
        int key = cv::waitKey(30);
        
        if (key == 'q' || key == 27) break;
        if (key == 'c') canvas = cv::Mat::zeros(280, 280, CV_8UC1);
    }
    
    cv::destroyAllWindows();
}