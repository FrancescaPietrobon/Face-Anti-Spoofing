#include "../include/face_detection.h"

using namespace cv;
using namespace std;
using namespace dlib;


FaceDetection::FaceDetection(frontal_face_detector _detector, Mat _img, Mat _cropedImage):
detector(_detector), img(_img), cropedImage(_cropedImage) {};


Mat FaceDetection::extract_rectangle()
{
    // Detect face in dlib rectangle
    dlib::rectangle faceRectDlib = FaceDetection::detect_rectangle();

    // Convert dlib rectangle in OpenCV rectangle
    cv::Rect faceRectCV = FaceDetection::dlib_rectangle_to_cv(faceRectDlib);

    cv::Rect ExpfaceRectCV = expand_rectangle(faceRectCV);

    cropedImage = img(ExpfaceRectCV);

    return cropedImage;
}


dlib::rectangle FaceDetection::detect_rectangle()
{
    // http://dlib.net/webcam_face_pose_ex.cpp.html

    // Make the image bigger by a factor of two.  This is useful since the face detector looks for faces that are about 80 by 80 pixels or larger. 
    //pyramid_up(img);

    // Convert OpenCV image format to Dlib's image format
    dlib::cv_image<dlib::bgr_pixel> cv_img = FaceDetection::cv_mat_to_dlib();

    // Detect faces 
    std::vector<dlib::rectangle> faces = detector(cv_img);

    if (faces.size() > 1)
        std::cerr << "More than one face" << std::endl;

    return faces[0]; 
}


dlib::cv_image<dlib::bgr_pixel> FaceDetection::cv_mat_to_dlib()
{
    // Turn OpenCV's Mat into something dlib can deal with
    auto ipl_img = cvIplImage(img); //_IplImage type
    auto cv_img = cv_image<bgr_pixel>(&ipl_img); //dlib::cv_image<dlib::bgr_pixel> type
    return cv_img;
}


cv::Rect FaceDetection::dlib_rectangle_to_cv(dlib::rectangle r)
{
    return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right() + 1, r.bottom() + 1));
}


cv::Rect FaceDetection::expand_rectangle(cv::Rect rect)
{
    // CONTROLLA OUT OF BOUNDS!!!

    return Rect(rect.x - 70, rect.y - 70, rect.width + 140, rect.height + 140);
}


void FaceDetection::print_rectangle_cv(bool blurred, string pred)
{
    //https://learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/

    // Detect faces in the image
	dlib::rectangle faceRect = FaceDetection::detect_rectangle();

    int frameHeight = 100;

	int x1 = faceRect.left();
	int y1 = faceRect.top();
	int x2 = faceRect.right();
	int y2 = faceRect.bottom();

    Rect faceRectCV = FaceDetection::dlib_rectangle_to_cv(faceRect);
    Rect ExpfaceRectCV = FaceDetection::expand_rectangle(faceRectCV);

    int x = ExpfaceRectCV.x;
	int y = ExpfaceRectCV.y;
	int width = ExpfaceRectCV.width;
	int height = ExpfaceRectCV.height;

    // Plot face detected
	cv::rectangle(img, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0), (int)(frameHeight/150.0), 4);

    // Plot face detected expanded
    cv::rectangle(img, Point(x, y), Point(x + width, y + height), Scalar(255,0,0), (int)(frameHeight/150.0), 4);

    // Plot result of the prediction if exists otherwise plot blurred if the image is blurred
    if (pred != "Null")
        putText(img, pred,  Point(x1 + int((x2-x1)/2) - 5, y1 - 3), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
    else if (blurred)
        putText(img, "Blurred",  Point(x1 + int((x2-x1)/2) - 5, y1 - 3), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

    // Plot dimension of the rectangle
    string dim = to_string(x2-x1) + string("x") + to_string(y2-y1);
    putText(img, dim,  Point(x2, y2), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

    imshow( "Image", img );
    //imwrite("/home/fra/Project/Frames/frame" + std::to_string(j+1) +".jpg", temp);
    
}


bool FaceDetection::blur_detection()
{
    // https://stackoverflow.com/questions/24080123/opencv-with-laplacian-formula-to-detect-image-is-blur-or-not-in-ios
    // https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/

    Mat laplacianImage = FaceDetection::compute_laplacian();

    Mat gray;
    cvtColor(cropedImage, gray, COLOR_BGR2GRAY);

    Laplacian(gray, laplacianImage, CV_64F);
    Scalar mean, stddev;
    meanStdDev(laplacianImage, mean, stddev, Mat());
    double variance = stddev.val[0] * stddev.val[0];

    double threshold = 6.5; //6.5 before

    bool blurred = true;

    if (variance >= threshold)
        blurred = false;

    /*
    string text = "Not Blurry";

    if (variance <= threshold)
        text = "Blurry";
    */

    return blurred;
}


Mat FaceDetection::compute_laplacian()
{
    //https://docs.opencv.org/3.4/d5/db5/tutorial_laplace_operator.html
            
    Mat src, src_gray, dst, abs_dst;
    int kernel_size = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    // Reduce noise by blurring with a Gaussian filter ( kernel size = 3 )
    GaussianBlur(cropedImage, cropedImage, Size(3, 3), 0, 0, BORDER_DEFAULT);

    cvtColor(cropedImage, src_gray, COLOR_BGR2GRAY); // Convert the image to grayscale

    Laplacian(src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);

    // converting back to CV_8U
    convertScaleAbs(dst, abs_dst);

    return abs_dst;
}








/* POSSIBLE USEFUL FUNCTIONS

std::vector<full_object_detection> FaceDetection::detect_shape(shape_predictor pose_model, frontal_face_detector detector, Mat temp)
{
    dlib::cv_image<dlib::bgr_pixel> cimg = FaceDetection::cv_mat_to_dlib(temp);

    // Detect faces 
    dlib::rectangle face = FaceDetection::detect_rectangle(detector, temp);

    // Find the pose of each face.
    std::vector<full_object_detection> shapes;

    shapes.push_back(pose_model(cimg, face));

    return shapes;
    
}


void FaceDetection::print_rectangle_dlib(Mat img, std::vector<dlib::rectangle> faces, string pred)
{
    dlib::cv_image<dlib::bgr_pixel> cimg = FaceDetection::cv_mat_to_dlib(img);

    image_window win;
    win.clear_overlay();
    win.set_image(cimg);
    if (pred == "Null")
        win.add_overlay(faces);
    else
        win.add_overlay(dlib::image_window::overlay_rect(faces[0], dlib::rgb_pixel(0, 0, 255), pred));
    waitKey(5000);

}


void FaceDetection::print_shape(Mat img, std::vector<full_object_detection> faces)
{
    dlib::cv_image<dlib::bgr_pixel> cimg = FaceDetection::cv_mat_to_dlib(img);

    image_window win;
    win.clear_overlay();
    win.set_image(cimg);
    win.add_overlay(render_face_detections(faces));
    waitKey(100);

}


dlib::rectangle FaceDetection::cv_rectangle_to_dlib(cv::Rect r)
{
    return dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1, (long)r.br().y - 1);
}


*/