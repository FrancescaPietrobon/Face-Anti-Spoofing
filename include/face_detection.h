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
        FaceDetection(frontal_face_detector detector, Mat img, Mat cropedImage, bool blurred, VideoCapture cap, int ROI_dim);
        Mat extract_rectangle();
        bool blur_detection();
        void print_rectangle_cv(string pred = "Null");
        Mat cropedImage;
        Mat img;
        bool blurred;
        VideoCapture cap;
        int ROI_dim;
        cv::Rect rect;
        bool out_of_bounds_top();
        bool out_of_bounds_bottom();
        bool out_of_bounds_right();
        bool out_of_bounds_left();
        bool out_of_bounds();
        cv::Rect detect_rectangle();
        cv::Rect extract_ROI();

    private:
        
        cv::Rect dlib_rectangle_to_cv(dlib::rectangle r);
        dlib::cv_image<dlib::bgr_pixel> cv_mat_to_dlib();
        cv::Rect expand_rectangle(cv::Rect rect);
        Mat compute_laplacian();
        frontal_face_detector detector;
        cv::Rect rectExp;
        
        int x_rect_center;
        int y_rect_center;
        int width_screen;
        int height_screen;
        int x_screen_center;
        int y_screen_center;

        
        
};


#endif