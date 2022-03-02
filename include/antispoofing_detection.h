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
        AntiSpoofingDetection(dnn::Net snn, Ptr<ml::RTrees> ml, int n_img, string frames_path);
        Mat face;
        dnn::Net snn;
        Ptr<ml::RTrees> ml;
        int n_img;
        string frames_path;
        string pred = "Null";
        string single_prediction(); 
        string multiple_prediction();
        int multiple_prediction_par(int world_rank, int world_size);
        int compute_sum_real(int *sum_real, int world_size);
        int value_prediction();
        
    private:
        int one_pred(int i, int count_real);
};


#endif