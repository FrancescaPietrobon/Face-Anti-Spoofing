#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

#ifndef FUNCTIONS_H_INCLUDED
#define FUNCTIONS_H_INCLUDED

string make_prediction(Mat img, dnn::Net cvNet, Ptr<ml::RTrees> svm);


#endif