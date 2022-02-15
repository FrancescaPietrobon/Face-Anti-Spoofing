#include "../include/final_prediction.h"
#include "../include/antispoofing_detection.h"
#include "../include/face_detection.h"
#include "../include/utilities.h"

using namespace cv;
using namespace std;
using namespace dlib;       
        
 
FinalPrediction::FinalPrediction(FaceDetection _face_detector, AntiSpoofingDetection _antispoofing_detector):
face_detector(_face_detector), antispoofing_detector(_antispoofing_detector) {};


void FinalPrediction::predict_image()
{
    //face_detector.rect = face_detector.detect_rectangle();

    if (face_detector.out_of_bounds())
        print_status(&face_detector.img, "Face out of bounds");
    else
    {
        // Extract only the face
        antispoofing_detector.face = face_detector.extract_rectangle();

        // If the face is not blurred print make the prediction and print them, otherwise print "Blurred"
        if (!face_detector.blur_detection())
        {
            // Make prediction for the face
            face_detector.print_rectangle_cv(antispoofing_detector.single_prediction());
        }
        else
        {
            // Print the image with prediction (or "Blorred"), dimensions, rectangles of face detected and of face considered to make the prediction
            face_detector.print_rectangle_cv();
        } 
    }
}




int FinalPrediction::predict_images(VideoCapture cap, int n_img, string frames_path)
{
    string window_name = "Webcam";

    int i = 1;

    while (true)
    {
        // Until the decided number of frames is not reached collect frames
        if (i < n_img)
        {
            // Read a new frame from video
            bool bSuccess = cap.read(face_detector.img);

            // Breaking the while loop if the frames cannot be captured
            if (camera_disconnection(bSuccess)) return 1;

            //face_detector.rect= face_detector.detect_rectangle();

            if (face_detector.out_of_bounds())
                print_status(&face_detector.img, "Face out of bounds");
            else
            {
                // Extract only the face
                antispoofing_detector.face = face_detector.extract_rectangle();
                                            
                // Check if the face is blurred 
                if (!face_detector.blur_detection())
                {
                    // Save frame
                    imwrite(frames_path + "frame" + std::to_string(i) +".jpg", antispoofing_detector.face);
                    i++;
                } 
            }
            imshow(window_name, face_detector.img);
        }
        else
        {
            if (antispoofing_detector.pred == "Null")
                print_status(&face_detector.img, "Performing prediction...");
            else
                print_status(&face_detector.img, antispoofing_detector.pred);

            imshow(window_name, face_detector.img);

            antispoofing_detector.pred = antispoofing_detector.multiple_prediction(frames_path);
        }

        // Check when close webcam
        if (close_webcam()) return 1;
    }
    return 0;
}


int FinalPrediction::predict_realtime(VideoCapture cap)
{
    while (true)
    {
        // Read a new frame from video 
        bool bSuccess = cap.read(face_detector.img); 

        // Breaking the while loop if the frames cannot be captured
        if (camera_disconnection(bSuccess)) return 1;
            
        // Make the prediction
        FinalPrediction::predict_image();

        // Check when close webcam
        if (close_webcam()) return 1;
    }
    return 0;
}  



bool FinalPrediction::camera_disconnection(bool bSuccess)
{
    // Breaking the while loop if the frames cannot be captured
    if (bSuccess == false) 
    {
        cout << "Video camera is disconnected" << endl;
        cin.get(); //Wait for any key press
        return true;
    }
    return false;
}


bool FinalPrediction::close_webcam()
{
    if (waitKey(1) == 27)
    {
        cout << "Esc key is pressed by user. Stoppig the video" << endl;
        return true;
    }
    return false;
}




        