#include "../include/face_detection.h"
#include "../include/utilities.h"

using namespace cv;
using namespace std;
using namespace dlib;


FaceDetection::FaceDetection(frontal_face_detector _detector, Mat _img, Mat _cropedImage, VideoCapture _cap, int _ROI_dim):
detector(_detector), img(_img), cropedImage(_cropedImage), cap(_cap), ROI_dim(_ROI_dim)
{
    /*
    cv::Rect rectExp;
    cv::Rect rect;

    int x_rect_center;
    int y_rect_center;

    
    int width_screen = cap.get(CAP_PROP_FRAME_WIDTH);
    int height_screen = cap.get(CAP_PROP_FRAME_HEIGHT);

    
    const int x_screen_center = width_screen/2;
    cout << x_screen_center << endl;
    const int y_screen_center = height_screen/2;
    cout << y_screen_center << endl;
    */
    /*
    int width_screen = cap.get(CAP_PROP_FRAME_WIDTH);
    cout << width_screen << endl;
    int height_screen = cap.get(CAP_PROP_FRAME_HEIGHT);
    cout << height_screen << endl;
    int x_screen_center = width_screen/2;
    cout << x_screen_center << endl;
    int y_screen_center = height_screen/2;
    cout << y_screen_center << endl;
    
    // If the screen is too small the ROI will be as big as the screen otherwise it is fixed
    if (width_screen<_ROI_dim || height_screen<_ROI_dim)
        ROI_dim = min(width_screen, height_screen);
    else
        ROI_dim = _ROI_dim;

    cout << ROI_dim << endl;
    
    
    int x_rect_center = rect.x + rect.width/2;
    int y_rect_center = rect.y + rect.height/2;


    */
};


bool FaceDetection::out_of_bounds_top() {return ((y_rect_center + ROI_dim/2) > height_screen);}

bool FaceDetection::out_of_bounds_bottom() {return ((y_rect_center - ROI_dim/2) < 0);}

bool FaceDetection::out_of_bounds_right() {return ((x_rect_center + ROI_dim/2) > width_screen);}

bool FaceDetection::out_of_bounds_left() {return ((x_rect_center - ROI_dim/2) < 0);}

bool FaceDetection::out_of_bounds()
{   
    detect_rectangle();
    // If the screen is too small the ROI will be as big as the screen otherwise it is fixed
    if (width_screen<ROI_dim || height_screen<ROI_dim)
        ROI_dim = min(width_screen, height_screen);
    else
        ROI_dim = ROI_dim;

    // If ROI is out of bounds in all sides or in height (such can happened in phones) or in width (such can happened in pc webcam),
    // the message of closeness to the screen is printed
    if ((FaceDetection::out_of_bounds_top() && FaceDetection::out_of_bounds_bottom() &&
         FaceDetection::out_of_bounds_right() && FaceDetection::out_of_bounds_left())
        || (FaceDetection::out_of_bounds_top() && FaceDetection::out_of_bounds_bottom())
        || (FaceDetection::out_of_bounds_right() && FaceDetection::out_of_bounds_left()))
    {
        print_status(&img, "The face is too close to the webcam.", false);
        return 1;
    }
    // If the center of the face and the center of the screen are away more than the half of the
    // minimum between the width and the height of the screen, the message of non centering is printed
    else if (abs(x_screen_center - x_rect_center) > int(min(width_screen, height_screen)/2)) //before /2
    {
        cout << x_rect_center << endl;
        cout << x_screen_center << endl;
        cout << abs(x_screen_center - x_rect_center) << endl;
        cout << int(min(width_screen, height_screen)/2) << endl;
        cout << width_screen << endl;
        cout << height_screen << endl;
        print_status(&img, "The face is not centered in the screen", false);
        return 1;
    }
    return 0;
}


cv::Rect FaceDetection::extract_ROI()
{    
    return Rect(x_rect_center - ROI_dim/2, y_rect_center - ROI_dim/2, ROI_dim, ROI_dim);
    //return Rect(rect.x - 70, rect.y - 70, rect.width + 140, rect.height + 140);
}


Mat FaceDetection::extract_rectangle()
{
    // Expand the rectangle extracting the ROI
    rectExp = FaceDetection::extract_ROI();
    cropedImage = img(rectExp);

    /*
    VideoCapture cap;
    cap.open(0, CAP_ANY);
    imshow( "Image", cropedImage );
    waitKey(5000);
    */

    return cropedImage;
}


void FaceDetection::detect_rectangle()
{
    // http://dlib.net/webcam_face_pose_ex.cpp.html

    // Make the image bigger by a factor of two.  This is useful since the face detector looks for faces that are about 80 by 80 pixels or larger. 
    //pyramid_up(img);

    // Convert OpenCV image format to Dlib's image format
    dlib::cv_image<dlib::bgr_pixel> cv_img = FaceDetection::cv_mat_to_dlib();

    // Detect faces 
    std::vector<dlib::rectangle> faces = detector(cv_img);

    if (faces.size() > 1)
    {
        std::cerr << "More than one face" << std::endl;
        print_status(&img, "More than one face, the first is extracted", false);
    }

    // Convert dlib rectangle in OpenCV rectangle
    rect = FaceDetection::dlib_rectangle_to_cv(faces[0]);

    x_rect_center = rect.x + rect.width/2;
    cout << x_rect_center << endl;
    y_rect_center = rect.y + rect.height/2;
    cout << y_rect_center << endl;
  
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

/*
cv::Rect FaceDetection::expand_rectangle(cv::Rect rect)
{
    // CONTROLLA OUT OF BOUNDS!!!

    return Rect(rect.x - 70, rect.y - 70, rect.width + 140, rect.height + 140);
}
*/

void FaceDetection::print_rectangle_cv(string pred)
{
    //https://learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/

    int frameHeight = 100; //?? A COSA SERVE

    // Plot face detected
	cv::rectangle(img, Point(rect.x, rect.y), Point(rect.x + rect.width, rect.y + rect.height), Scalar(0, 255, 0), (int)(frameHeight/150.0), 4);

    // Plot face detected expanded
    cv::rectangle(img, Point(rectExp.x, rectExp.y), Point(rectExp.x + rectExp.width, rectExp.y + rectExp.height), Scalar(255,0,0), (int)(frameHeight/150.0), 4);

    // Plot result of the prediction if exists otherwise plot blurred if the image is blurred
    if (pred != "Null")
        putText(img, pred,  Point(x_screen_center - 10, y_screen_center), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
    else if (blurred)
        putText(img, "Blurred",  Point(x_screen_center - 10, y_screen_center), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

    // Plot dimension of the rectangle
    string dim = to_string(rect.width) + string("x") + to_string(rect.height);
    putText(img, dim,  Point(rect.x + rect.width, rect.y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

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

    double threshold = 5.5; //6.5 before

    blurred = true;

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