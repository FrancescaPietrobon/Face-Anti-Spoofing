#include "my_functions.h"

using namespace cv;
using namespace std;
using namespace dlib;



string make_prediction(Mat img, dnn::Net cvNet, Ptr<ml::RTrees> svm)
{
        // SNN prediction      

        //image, output size, mean values to subtract from channels, swap first and last channels, cropped after resize or not, depth of output blob
        Mat blob = dnn::blobFromImage(img, 1, Size(256, 256), Scalar(0,0,0), true, false, CV_32F);
        cvNet.setInput(blob);
        Mat features = cvNet.forward();
        //cout << "features = " << endl << " "  << features << endl << endl;

        // ML Model prediction
        float prediction = svm->predict(features);

        string output;
        if (prediction == 0)
            output = "Real";
        else
            output = "Fake";
        
        return output;
}


void face_detection(frontal_face_detector detector, Mat temp)
{
    // http://dlib.net/webcam_face_pose_ex.cpp.html

    image_window win;

    shape_predictor pose_model;
    deserialize("/home/fra/PROGETTO_PACS/Face-Anti-Spoofing/models/shape_predictor_68_face_landmarks.dat") >> pose_model;

    // Make the image bigger by a factor of two.  This is useful since
    // the face detector looks for faces that are about 80 by 80 pixels
    // or larger.  Therefore, if you want to find faces that are smaller
    // than that then you need to upsample the image as we do here by
    // calling pyramid_up().  So this will allow it to detect faces that
    // are at least 40 by 40 pixels in size.  We could call pyramid_up()
    // again to find even smaller faces, but note that every time we
    // upsample the image we make the detector run slower since it must
    // process a larger image.
    //pyramid_up(temp);


    // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
    // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
    // long as temp is valid.  Also don't do anything to temp that would cause it
    // to reallocate the memory which stores the image as that will make cimg
    // contain dangling pointers.  This basically means you shouldn't modify temp
    // while using cimg.
    //cv_image<bgr_pixel> cimg(temp);
    //IplImage ipl_img = cvIplImage(temp);
    //dlib::cv_image<dlib::bgr_pixel> cimg(&ipl_img);

    //Mat temp = dnn::blobFromImage(img, 1, Size(256, 256), Scalar(0,0,0), true, false, CV_32F);

    auto ipl_img = cvIplImage(temp);
    auto cimg = cv_image<bgr_pixel>(&ipl_img);

    // Detect faces 
    std::vector<dlib::rectangle> faces = detector(cimg);

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


    /*

    https://learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/

    frontal_face_detector hogFaceDetector = get_frontal_face_detector();

   // Convert OpenCV image format to Dlib's image format
	cv_image<bgr_pixel> dlibIm(frameDlibHogSmall);

    // Detect faces in the image
	std::vector<dlib::rectangle> faceRects = hogFaceDetector(dlibIm);

	for ( size_t i = 0; i < faceRects.size(); i++ )
	{
	  int x1 = faceRects[i].left();
	  int y1 = faceRects[i].top();
	  int x2 = faceRects[i].right();
	  int y2 = faceRects[i].bottom();
	  cv::rectangle(frameDlibHog, Point(x1, y1), Point(x2, y2), Scalar(0,255,0), (int)(frameHeight/150.0), 4);
	}
    */
}