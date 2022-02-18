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
        FaceDetection(frontal_face_detector detector, VideoCapture cap, int ROI_dim);
        Mat img;
        VideoCapture cap;
        bool detect_rectangle();
        bool out_of_bounds();
        Mat extract_rectangle();
        bool blur_detection();
        void print_rectangle_cv(string pred = "Null");
        
    private:
        cv::Rect rectExp;
        Mat cropedImage;
        frontal_face_detector detector;
        int ROI_dim;
        cv::Rect rect;
        bool detected_ROI = true;
        bool blurred = false;
        int x_rect_center;
        int y_rect_center;
        const int width_screen = cap.get(CAP_PROP_FRAME_WIDTH);
        const int height_screen = cap.get(CAP_PROP_FRAME_HEIGHT);
        const int x_screen_center = width_screen/2;
        const int y_screen_center = height_screen/2;
        bool ROI_out_of_bounds_top();
        bool ROI_out_of_bounds_bottom();
        bool ROI_out_of_bounds_right();
        bool ROI_out_of_bounds_left();
        bool face_out_of_bounds_top();
        bool face_out_of_bounds_bottom();
        bool face_out_of_bounds_right();
        bool face_out_of_bounds_left();
        cv::Rect extract_ROI();
        cv::Rect dlib_rectangle_to_cv(dlib::rectangle r);
        dlib::cv_image<dlib::bgr_pixel> cv_mat_to_dlib();
        cv::Rect expand_rectangle(cv::Rect rect);
        Mat compute_laplacian();
};


#endif