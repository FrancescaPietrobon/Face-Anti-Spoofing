#include "face_detection.h"

using namespace cv;
using namespace std;
using namespace dlib;


void FaceDetection::CVprint_rectangle(frontal_face_detector detector, Mat temp)
{
    //https://learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/

    // Convert OpenCV image format to Dlib's image format
    dlib::cv_image<dlib::bgr_pixel> cimg = FaceDetection::OpenCVMatTodlib(temp);

    // Detect faces in the image
	std::vector<dlib::rectangle> faceRects = detector(cimg);

    int frameHeight = 100;

	for ( size_t i = 0; i < faceRects.size(); i++ )
	{
	  int x1 = faceRects[i].left();
	  int y1 = faceRects[i].top();
	  int x2 = faceRects[i].right();
	  int y2 = faceRects[i].bottom();
	  cv::rectangle(temp, Point(x1, y1), Point(x2, y2), Scalar(0,255,0), (int)(frameHeight/150.0), 4);
      imshow( "result", temp );
      waitKey(5000);
	}
    

}

std::vector<dlib::rectangle> FaceDetection::detect_rectangle(frontal_face_detector detector, Mat temp)
{
    // http://dlib.net/webcam_face_pose_ex.cpp.html

    // Make the image bigger by a factor of two.  This is useful since the face detector looks for faces that are about 80 by 80 pixels or larger. 
    //pyramid_up(temp);

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
        win.add_overlay(dlib::image_window::overlay_rect(faces[0], dlib::rgb_pixel(0, 0, 255), pred)); //if also with prediction
    waitKey(5000);

}

void FaceDetection::print_shape(Mat img, std::vector<full_object_detection> faces)
{
    dlib::cv_image<dlib::bgr_pixel> cimg = FaceDetection::OpenCVMatTodlib(img);

    image_window win;
    win.clear_overlay();
    win.set_image(cimg);
    win.add_overlay(render_face_detections(faces));
    waitKey(5000);

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


    /*

    // Find the pose of each face.
    std::vector<full_object_detection> shapes;

    for (unsigned long i = 0; i < faces.size(); ++i)
        //cout << i << endl;
        shapes.push_back(pose_model(cimg, faces[i]));

    

    // Display it all on the screen
    win.clear_overlay();
    win.set_image(cimg);
    win.add_overlay(faces, rgb_pixel(255,0,0)); //to display rectangle
    win.add_overlay(render_face_detections(shapes)); //to display shape

    waitKey(0);
    */



