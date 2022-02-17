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
        AntiSpoofingDetection(Mat face, dnn::Net snn, Ptr<ml::RTrees> ml, string pred);
        string single_prediction(); 
        string multiple_prediction(string frames_path);
        Mat face;
        string pred;
    private:
        float value_prediction();
        dnn::Net snn;
        Ptr<ml::RTrees> ml;
};


#endif