#ifndef UTILITIES_H_
#define UTILITIES_H_

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

void print_status(Mat *frame, string message, bool black = true);

bool camera_disconnection(bool bSuccess);

bool close_webcam();

//Mat print_image(VideoCapture cap, string message);


#endif