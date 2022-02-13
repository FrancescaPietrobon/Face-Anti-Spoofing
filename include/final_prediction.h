#ifndef FINAL_PREDICTION_H_
#define FINAL_PREDICTION_H_

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


class FinalPrediction: public AntiSpoofingDetection, public FaceDetection
{
    public:
        void predict_image(string img_path, frontal_face_detector detector, dnn::Net snn, Ptr<ml::RTrees> ml);

};


#endif