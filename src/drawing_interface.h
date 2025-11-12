#ifndef DRAWING_INTERFACE_H
#define DRAWING_INTERFACE_H

#include <opencv2/opencv.hpp>
#include "neural_network.h"

class DrawingInterface {
private:
    cv::Mat canvas;
    cv::Mat display;
    bool drawing;
    cv::Point lastPoint;
    NeuralNetwork& neuralNetwork;

public:
    DrawingInterface(NeuralNetwork& nn);
    void run();
    
private:
    void initializeCanvas();
    void drawCircle(cv::Point center);
    void onMouse(int event, int x, int y, int flags, void* param);
    cv::Mat preprocessDrawing();
    static void mouseCallback(int event, int x, int y, int flags, void* param);
};

#endif