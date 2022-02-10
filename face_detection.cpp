#include "face_detection.h"

using namespace cv;
using namespace std;
using namespace dlib;


cv::Rect FaceDetection::expand_face_rect(cv::Rect rect)
{
    //Rect(X,Y,Width,Height)
    //return Rect(max(rect.x - int(rect.width/2), 0), max(rect.y - int(rect.height/2), 0), rect.width * 2, rect.height * 2);
    return Rect(rect.x - 70, rect.y - 70, rect.width + 140, rect.height + 140);
}


Mat FaceDetection::extract_face_rect(frontal_face_detector detector, Mat temp)
{
    std::vector<dlib::rectangle> faceRectsDlib = FaceDetection::detect_rectangle(detector, temp);

    if (faceRectsDlib.size() > 1)
    {
        std::cerr << "More than one face" << std::endl;
    }

    cv::Rect faceRectsCV = FaceDetection::dlibRectangleToOpenCV(faceRectsDlib[0]);

    cv::Rect ExpfaceRectsCV = expand_face_rect(faceRectsCV);

    Mat cropedImage = temp(ExpfaceRectsCV);

    //Mat cropedImage = croppedRef(temp, ExpfaceRectsCV);

    return cropedImage;

    /* To save cropped image
    cv::Mat cropped;
    // Copy the data into new matrix
    croppedRef.copyTo(cropped);
    */


    /* If manage multiple faces
    std::vector<cv::Rect> faceRectsCV;

    for ( size_t i = 0; i < faceRectsDlib.size(); i++ )
	{
        faceRectsCV.push_back(FaceDetection::dlibRectangleToOpenCV(dlib::rectangle faceRectsDlib[i]));
    }
    */

}


void FaceDetection::CVprint_rectangle(frontal_face_detector detector, Mat temp, string pred)
{
    //https://learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/

    // Detect faces in the image
	std::vector<dlib::rectangle> faceRects = FaceDetection::detect_rectangle(detector, temp);

    int frameHeight = 100;

	for ( size_t i = 0; i < faceRects.size(); i++ )
	{
	    int x1 = faceRects[i].left();
	    int y1 = faceRects[i].top();
	    int x2 = faceRects[i].right();
	    int y2 = faceRects[i].bottom();

        Rect faceRectsCV = FaceDetection::dlibRectangleToOpenCV(faceRects[i]);
        Rect faceResctExp = FaceDetection::expand_face_rect(faceRectsCV);

        int x = faceResctExp.x;
	    int y = faceResctExp.y;
	    int width = faceResctExp.width;
	    int height = faceResctExp.height;

        // Plot face detected
	    cv::rectangle(temp, Point(x1, y1), Point(x2, y2), Scalar(0,255,0), (int)(frameHeight/150.0), 4);
        // Plot face detected expanded
        cv::rectangle(temp, Point(x, y), Point(x + width, y + height), Scalar(0,255,0), (int)(frameHeight/150.0), 4);

        putText(temp, pred,  Point(x1 + int((x2-x1)/2) - 5, y1 - 3), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
        string dim = to_string(x2-x1) + string("x") + to_string(y2-y1);
        putText(temp, dim,  Point(x2, y2), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
        imshow( "Image", temp );
        //imwrite("/home/fra/Project/Frames/frame" + std::to_string(j+1) +".jpg", temp);
	}
    

}

std::vector<dlib::rectangle> FaceDetection::detect_rectangle(frontal_face_detector detector, Mat temp)
{
    // http://dlib.net/webcam_face_pose_ex.cpp.html

    // Make the image bigger by a factor of two.  This is useful since the face detector looks for faces that are about 80 by 80 pixels or larger. 
    //pyramid_up(temp);

    // Convert OpenCV image format to Dlib's image format
    dlib::cv_image<dlib::bgr_pixel> cimg = FaceDetection::OpenCVMatTodlib(temp);

    // Detect faces 
    std::vector<dlib::rectangle> faces = detector(cimg);

    return faces;
    
}

std::vector<full_object_detection> FaceDetection::detect_shape(shape_predictor pose_model, frontal_face_detector detector, Mat temp)
{
    dlib::cv_image<dlib::bgr_pixel> cimg = FaceDetection::OpenCVMatTodlib(temp);

    // Detect faces 
    std::vector<dlib::rectangle> faces = FaceDetection::detect_rectangle(detector, temp);

    // Find the pose of each face.
    std::vector<full_object_detection> shapes;

    
    for (unsigned long i = 0; i < faces.size(); ++i)
        shapes.push_back(pose_model(cimg, faces[i]));
    

    return shapes;
    
}


void FaceDetection::print_rectangle(Mat img, std::vector<dlib::rectangle> faces, string pred)
{
    dlib::cv_image<dlib::bgr_pixel> cimg = FaceDetection::OpenCVMatTodlib(img);

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
    dlib::cv_image<dlib::bgr_pixel> cimg = FaceDetection::OpenCVMatTodlib(img);

    image_window win;
    win.clear_overlay();
    win.set_image(cimg);
    win.add_overlay(render_face_detections(faces));
    waitKey(100);

}


dlib::cv_image<dlib::bgr_pixel> FaceDetection::OpenCVMatTodlib(Mat temp)
{
    // Turn OpenCV's Mat into something dlib can deal with
    auto ipl_img = cvIplImage(temp); //_IplImage type
    auto cimg = cv_image<bgr_pixel>(&ipl_img); //dlib::cv_image<dlib::bgr_pixel> type
    return cimg;
}


cv::Rect FaceDetection::dlibRectangleToOpenCV(dlib::rectangle r)
{
    return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right() + 1, r.bottom() + 1));
}


dlib::rectangle FaceDetection::openCVRectToDlib(cv::Rect r)
{
    return dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1, (long)r.br().y - 1);
}


Mat FaceDetection::laplacian_plot(Mat img)
{
    //https://docs.opencv.org/3.4/d5/db5/tutorial_laplace_operator.html
            
    Mat src, src_gray, dst, abs_dst;
    int kernel_size = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    // Reduce noise by blurring with a Gaussian filter ( kernel size = 3 )
    GaussianBlur(img, img, Size(3, 3), 0, 0, BORDER_DEFAULT);

    cvtColor(img, src_gray, COLOR_BGR2GRAY); // Convert the image to grayscale

    Laplacian(src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);

    // converting back to CV_8U
    convertScaleAbs(dst, abs_dst);

    return abs_dst;
}


string FaceDetection::blur_detection(Mat img)
{
    // https://stackoverflow.com/questions/24080123/opencv-with-laplacian-formula-to-detect-image-is-blur-or-not-in-ios
    // https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/

    Mat laplacianImage = FaceDetection::laplacian_plot(img);

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    Laplacian(gray, laplacianImage, CV_64F);
    Scalar mean, stddev; // 0:1st channel, 1:2nd channel and 2:3rd channel
    meanStdDev(laplacianImage, mean, stddev, Mat());
    double variance = stddev.val[0] * stddev.val[0];

    double threshold = 5;

    string text = "Not Blurry";

    if (variance <= threshold)
        text = "Blurry";

    return text;
}


