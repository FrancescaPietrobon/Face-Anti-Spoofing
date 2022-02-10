#ifndef FACE_DETECTION_H_
#define FACE_DETECTION_H_

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


namespace FaceDetection
{
    std::vector<dlib::rectangle> detect_rectangle(frontal_face_detector detector, dlib::cv_image<dlib::bgr_pixel> temp);
    std::vector<full_object_detection> detect_shape(shape_predictor pose_model, frontal_face_detector detector, Mat temp);
    dlib::cv_image<dlib::bgr_pixel> OpenCVMatTodlib(Mat temp);
    cv::Rect dlibRectangleToOpenCV(dlib::rectangle r);
    dlib::rectangle openCVRectToDlib(cv::Rect r);

};


#endif