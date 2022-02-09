#ifndef MY_FUNCTIONS_H
#define MY_FUNCTIONS_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <iostream>
#include <string>

#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

using namespace cv;
using namespace std;
using namespace dlib;


string make_prediction(Mat img, dnn::Net cvNet, Ptr<ml::RTrees> svm);

void face_detection(frontal_face_detector detector, Mat img);

#endif