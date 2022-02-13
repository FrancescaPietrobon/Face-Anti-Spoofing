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


class FaceDetection
{
    public:
        FaceDetection(frontal_face_detector detector, Mat img, Mat cropedImage, bool blurred);
        Mat extract_rectangle();
        bool blur_detection();
        void print_rectangle_cv(string pred = "Null");
        Mat cropedImage;
        Mat img;
        bool blurred;

    private:
        dlib::rectangle detect_rectangle();
        cv::Rect dlib_rectangle_to_cv(dlib::rectangle r);
        dlib::cv_image<dlib::bgr_pixel> cv_mat_to_dlib();
        cv::Rect expand_rectangle(cv::Rect rect);
        Mat compute_laplacian();
        frontal_face_detector detector;
        
        
};


#endif