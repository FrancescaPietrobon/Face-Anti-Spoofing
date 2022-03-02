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
        FinalPrediction(FaceDetection *face_detector, AntiSpoofingDetection *antispoofing_detector);
        FaceDetection *face_detector;
        AntiSpoofingDetection *antispoofing_detector;
        void predict_image();
        int predict_realtime();
        int predict_multiple_frames(string frames_path, int world_rank, int world_size);
        
    private:
        int predict_images(string frames_path);
        int predict_images_par(string frames_path, int world_rank, int world_size);

};


#endif