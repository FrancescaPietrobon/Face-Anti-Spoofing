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

#include "../include/antispoofing_detection.h"
#include "../include/face_detection.h"

using namespace cv;
using namespace std;
using namespace dlib;


class FinalPrediction
{
    public:
        FinalPrediction(FaceDetection face_detector, AntiSpoofingDetection antispoofing_detector);
        void predict_image();
        int predict_images(int n_img, string frames_path);
        int predict_realtime();
        FaceDetection face_detector;
        AntiSpoofingDetection antispoofing_detector;
    private:
        bool camera_disconnection(bool bSuccess);
        bool close_webcam();
        
};


#endif