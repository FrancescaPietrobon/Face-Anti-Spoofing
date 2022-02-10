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
    cv::Rect expand_face_rect(cv::Rect rect);
    Mat extract_face_rect(frontal_face_detector detector, Mat temp);
    std::vector<dlib::rectangle> detect_rectangle(frontal_face_detector detector, Mat temp);
    std::vector<full_object_detection> detect_shape(shape_predictor pose_model, frontal_face_detector detector, Mat temp);
    void CVprint_rectangle(frontal_face_detector detector, Mat temp, string pred = "Null");
    void print_rectangle(Mat img, std::vector<dlib::rectangle> faces, string pred = "Null");
    void print_shape(Mat img, std::vector<full_object_detection> faces);
    dlib::cv_image<dlib::bgr_pixel> OpenCVMatTodlib(Mat temp);
    cv::Rect dlibRectangleToOpenCV(dlib::rectangle r);
    dlib::rectangle openCVRectToDlib(cv::Rect r);
    Mat laplacian_plot(Mat img);
    string blur_detection(Mat img);

};


#endif