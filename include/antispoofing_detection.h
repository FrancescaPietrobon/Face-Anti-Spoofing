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
        AntiSpoofingDetection(dnn::Net snn, Ptr<ml::RTrees> ml, int n_img, string frames_path, int world_rank);
        Mat face;
        dnn::Net snn;
        Ptr<ml::RTrees> ml;
        int n_img;
        string frames_path;
        int world_rank;
        string pred = "Null";
        string single_prediction(); 
        string multiple_prediction();
        //int *create_indexes(int elements_per_proc, int world_size);
        //int compute_real(int *sub_indexes, int elements_per_proc);
        int compute_sum_real(int *sum_real, int world_size);
        int value_prediction();
};


#endif