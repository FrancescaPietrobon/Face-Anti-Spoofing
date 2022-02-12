#ifndef ANTISPOOFINGDETECTION_H_
#define ANTISPOOFINGDETECTION_H_

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

class AntiSpoofingDetection
{
    public:
        string single_prediction(Mat img, dnn::Net snn, Ptr<ml::RTrees> ml); 
        string multiple_prediction(string frames_path, dnn::Net snn, Ptr<ml::RTrees> ml);
        void print_status(Mat *frame, string message);
    private:
        float value_prediction(Mat img, dnn::Net snn, Ptr<ml::RTrees> ml);

};


#endif