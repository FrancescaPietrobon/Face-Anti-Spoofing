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
    string window_name = "Webcam";

    int i = 1;

    while (true)
    {
        //if (i != 1) imshow(window_name, face_detector->img);

        // Until the decided number of frames is not reached collect frames
        if (i <= antispoofing_detector->n_img)
        {
            cout << i << endl;

            // Read a new frame from video
            bool bSuccess = face_detector->cap.read(face_detector->img);

            // Breaking the while loop if the frames cannot be captured
            if (camera_disconnection(bSuccess)) return 1;

            // If the face is detected
            if (face_detector->detect_rectangle())
            {
                // If the face detected is not out of bounds
                if (!face_detector->out_of_bounds())// && !(face_detector->ROI_dim==0))
                {
                    // Extract only the face
                    antispoofing_detector->face = face_detector->extract_rectangle();
                                                        
                    // Check if the face is blurred 
                    if (!face_detector->blur_detection())
                    {
                        // Save frame
                        imwrite(frames_path + "frame" + std::to_string(i) +".jpg", antispoofing_detector->face);
                        i++;
                    } 
                }
            }
            
        }
        else
        {
            cout << "Fine while" << endl;
            // After acquisition of the images required print "Performing prediction..."
            if (i == antispoofing_detector->n_img + 1)
            {
                print_status(&face_detector->img, "Performing prediction...");
                i++;
                waitKey(500);
            }
            else
            {
                // If there is no prediction compute it
                if (antispoofing_detector->pred == "Null")
                {
                    cout << "Peroforming pred" << endl;
                    // Compute the overall prediction
                    antispoofing_detector->pred = antispoofing_detector->multiple_prediction();
                }
                else
                {
                    // Print the prediction
                    print_status(&face_detector->img, antispoofing_detector->pred);
                }
            }
        }
        
        imshow(window_name, face_detector->img);

        // Check when close webcam
        if (close_webcam()) return 1;
    }


     /*
    // Until the decided number of frames is not reached collect frames
    while (i < antispoofing_detector->n_img)
    {
        cout << i << endl;

        // Read a new frame from video
        bool bSuccess = face_detector->cap.read(face_detector->img);
        imshow(window_name, face_detector->img);

        // Breaking the while loop if the frames cannot be captured
        if (camera_disconnection(bSuccess)) return 1;

        // If the face is detected
        if (face_detector->detect_rectangle())
        {
            // If the face detected is not out of bounds
            if (!face_detector->out_of_bounds())// && !(face_detector->ROI_dim==0))
            {
                // Extract only the face
                antispoofing_detector->face = face_detector->extract_rectangle();
                                                    
                // Check if the face is blurred 
                if (!face_detector->blur_detection())
                {
                    // Save frame
                    imwrite(frames_path + "frame" + std::to_string(i) +".jpg", antispoofing_detector->face);
                    i++;
                } 
            }
        }
        imshow(window_name, face_detector->img);
    }
    cout << "Fine while" << endl;
    // While is no prediction print "Performing prediction..." otherwise print the overall prediction
    print_status(&face_detector->img, "Performing prediction...");
    imshow(window_name, face_detector->img);
    waitKey(500);

    cout << "Pre multiple pred" << endl;
    // Compute the overall prediction
    antispoofing_detector->pred = antispoofing_detector->multiple_prediction();

    // Print the prediction
    print_status(&face_detector->img, antispoofing_detector->pred);
    imshow(window_name, face_detector->img);

    // Check when close webcam
    if (close_webcam()) return 1;

    waitKey(5000);
    */
    
    return 0;
}       