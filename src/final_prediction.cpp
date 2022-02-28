#include "../include/final_prediction.h"
#include "../include/antispoofing_detection.h"
#include "../include/face_detection.h"
#include "../include/utilities.h"

using namespace cv;
using namespace std;
using namespace dlib;       
        

FinalPrediction::FinalPrediction(FaceDetection *_face_detector, AntiSpoofingDetection *_antispoofing_detector):
face_detector(_face_detector), antispoofing_detector(_antispoofing_detector) {};


void FinalPrediction::predict_image()
{
    /// Makes the final prediction (real or fake) for a single image
    /** 
     * Arguments:
     *      None.
     * 
     *  Returns:
     *      None.
    */

    // If the face is detected
    if (face_detector->detect_rectangle())
    {
        // If the face detected is not out of bounds
        if (!face_detector->out_of_bounds())
        {
            // Extract only the face
            antispoofing_detector->face = face_detector->extract_rectangle();

            // If the face is not blurred, make the prediction and print them, otherwise print "Blurred"
            if (!face_detector->blur_detection())
            {
                // Make prediction for the face
                face_detector->print_rectangle_cv(antispoofing_detector->single_prediction());
            }
            else
            {
                // Print the image with prediction (or "Blorred"), dimensions, rectangles of face detected and of face considered to make the prediction
                face_detector->print_rectangle_cv();
            } 
        }
    }
    imshow("Image", face_detector->img);
}


int FinalPrediction::predict_realtime()
{
    /// Makes the final prediction (real or fake) realtime for each frame collected by the camera
    /** 
     * Arguments:
     *      None.
     * 
     *  Returns:
     *      None.
    */

    while (true)
    {
        // Read a new frame from video 
        bool bSuccess = face_detector->cap.read(face_detector->img);

        // Breaking the while loop if the frames cannot be captured
        if (camera_disconnection(bSuccess)) return 1;
            
        // Make the prediction
        FinalPrediction::predict_image();

        // Check when close webcam
        if (close_webcam()) return 1;
    }
    return 0;
} 


int FinalPrediction::predict_images(string frames_path)
{
    /// Makes the final prediction (real or fake) for multiple images
    /** 
     * Arguments:
     *      frames_path: path where the frames will be collected before computing the prediction
     * 
     *  Returns:
     *      Int that will be 0 if all works fine and 1 if camera disconnects or is closed.
    */

    string window_name = "Webcam";
    int i = 1;

    while (true)
    {
        // Until the decided number of frames is not reached collect frames
        if (i <= antispoofing_detector->n_img)
        {
            i = collect_frames(face_detector, antispoofing_detector, frames_path, i);
            if (i == antispoofing_detector->n_img + 3)
            {
                print_status(&face_detector->img, "Camera disconnected");
                imshow(window_name, face_detector->img);
                waitKey(5000);
                return 1;
            }     
        }
        else
        {
            // If there is no prediction compute it
            if (antispoofing_detector->pred == "Null")
            {
                // Compute the overall prediction
                antispoofing_detector->pred = antispoofing_detector->multiple_prediction();
            }
            else
            {
                // Print the prediction
                print_status(&face_detector->img, antispoofing_detector->pred);
                imshow(window_name, face_detector->img);
            }
        }

        // Check when close webcam
        if (close_webcam()) return 1;
    }
    
    return 0;
}       