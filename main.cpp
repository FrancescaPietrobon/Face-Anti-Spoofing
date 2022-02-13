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

#include "GetPot"

//#include "include/parameters.h"
#include "include/antispoofing_detection.h"
#include "include/face_detection.h"
#include "include/final_prediction.h"

using namespace std;
using namespace cv;
using namespace dlib;


/*
void predict_image(string img_path, frontal_face_detector detector, dnn::Net snn, Ptr<ml::RTrees> ml, AntiSpoofingDetection antispoofing_detector, FaceDetection face_detector)
{
    Mat img = imread(img_path, IMREAD_COLOR);

    // Extract only the face
    Mat cropedImage = face_detector.extract_rectangle(detector, img);

    // Check if the face is blurred
    bool blurred = face_detector.blur_detection(cropedImage);

    // If the face is not blurred print make the prediction and print them, otherwise print "Blurred"
    if (!blurred)
    {
        // Make prediction for the face
        string pred = antispoofing_detector.single_prediction(cropedImage, snn, ml);
        face_detector.print_rectangle_cv(detector, img, blurred, pred);
    }
    else
    {
        // Print the image with prediction (or "Blorred"), dimensions, rectangles of face detected and of face considered to make the prediction
        face_detector.print_rectangle_cv(detector, img, blurred);
    }
}
*/ 


void predict_image(string img_path, frontal_face_detector detector, dnn::Net snn, Ptr<ml::RTrees> ml)
{
    Mat img = imread(img_path, IMREAD_COLOR);
    Mat cropedImage;

    FaceDetection face_detector(detector, img, cropedImage);

    // Extract only the face
    cropedImage = face_detector.extract_rectangle();

    face_detector.cropedImage = cropedImage;

    // Check if the face is blurred
    bool blurred = face_detector.blur_detection();

    AntiSpoofingDetection antispoofing_detector(cropedImage, snn, ml);

    // If the face is not blurred print make the prediction and print them, otherwise print "Blurred"
    if (!blurred)
    {
        // Make prediction for the face
        string pred = antispoofing_detector.single_prediction();
        face_detector.print_rectangle_cv(blurred, pred);
    }
    else
    {
        // Print the image with prediction (or "Blorred"), dimensions, rectangles of face detected and of face considered to make the prediction
        face_detector.print_rectangle_cv(blurred);
    }
}



bool camera_disconnection(bool bSuccess)
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


bool close_webcam()
{
    if (waitKey(1) == 27)
    {
        cout << "Esc key is pressed by user. Stoppig the video" << endl;
        return true;
    }
    return false;
}






int main(int argc, char* argv[])
{

    //AntiSpoofingDetection antispoofing_detector;
    //FaceDetection face_detector;

    //string filename = "../src/data.dat";

    //GetPot parser(filename.c_str());
    GetPot cl(argc, argv);

    //string frames_path = parser(frames_path.c_str(), "/home/fra/Project/Frames/");
    string frames_path = "/home/fra/Project/Frames/";
    //string SNN_weights = parser(SNN_weights.c_str(), "/home/fra/PROGETTO_PACS/Face-Anti-Spoofing/models/Frozen_graph_All_final_net_5e-4.pb");
    string SNN_weights = "/home/fra/PROGETTO_PACS/Face-Anti-Spoofing/models/Frozen_graph_All_final_net_5e-4.pb";
    //string ML_weights = parser(ML_weights.c_str(), "/home/fra/PROGETTO_PACS/Face-Anti-Spoofing/models/All_RF_opencv_final_net_lr5e-4.xml");
    string ML_weights = "/home/fra/PROGETTO_PACS/Face-Anti-Spoofing/models/All_RF_opencv_final_net_lr5e-4.xml";
    //string face_detect = parser(face_detect.c_str(), "/home/fra/PROGETTO_PACS/Face-Anti-Spoofing/models/shape_predictor_68_face_landmarks.dat");
    string face_detect = "/home/fra/PROGETTO_PACS/Face-Anti-Spoofing/models/shape_predictor_68_face_landmarks.dat";
    //string example = parser(example.c_str(), "/home/fra/Scaricati/2022-02-08-163449.jpg");
    string example = "/home/fra/Scaricati/2022-02-08-163449.jpg";

    // Set webcam options
    //int deviceID = parser("deviceID", 0); // 0 = open default camera
    int deviceID = 0;       // 0 = open default camera
    int apiID = CAP_ANY;    // 0 = autodetect default API

    // Load SNN
    dnn::Net snn = cv::dnn::readNetFromTensorflow(SNN_weights);

    // Load ML Model
    Ptr<ml::RTrees> ml = Algorithm::load<ml::RTrees> (ML_weights);

    // Load face detection and pose estimation models.
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor pose_model;
    deserialize(face_detect) >> pose_model;


    

    if (cl.search(2, "-e", "--example"))
    {
        // Extract path
        string img_path = cl.next(example.c_str());

        // Make the prediction
        predict_image(img_path, detector, snn, ml);

        waitKey(5000);

    }

    
    else if (cl.search(2, "-wr", "--webcamrealtime"))
    {
        // Open the default video camera
        VideoCapture cap;

        // Open selected camera using selected API
        cap.open(deviceID, apiID);

        Mat frame;
        Mat cropedFrame;
        AntiSpoofingDetection antispoofing_detector(frame, snn, ml);
        FaceDetection face_detector(detector, frame, cropedFrame);

        while (true)
        {
            // Read a new frame from video 
            bool bSuccess = cap.read(frame); 

            face_detector.img = frame;

            // Breaking the while loop if the frames cannot be captured
            if (camera_disconnection(bSuccess)) break;

            // Extract only the face
            cropedFrame = face_detector.extract_rectangle();

            antispoofing_detector.img = cropedFrame;
            face_detector.cropedImage = cropedFrame;
                               
            // Check if the face is blurred
            bool blurred = face_detector.blur_detection();
                    
            // If the face is not blurred print make the prediction and print them, otherwise print "Blurred"
            if (!blurred)
            {
                string pred = antispoofing_detector.single_prediction();
                face_detector.print_rectangle_cv(blurred, pred);
            }
            else
                face_detector.print_rectangle_cv(blurred);

            // Check when close webcam
            if (close_webcam()) break;
        }
    }
    else
    {
        // Open the default video camera
        VideoCapture cap;

        // Open selected camera using selected API
        cap.open(deviceID, apiID);

        string window_name = "Webcam";
        string pred = "Null";

        int i = 1;

        Mat frame;
        Mat cropedFrame;
        AntiSpoofingDetection antispoofing_detector(frame, snn, ml);
        FaceDetection face_detector(detector, frame, cropedFrame);

        while (true)
        {
            // Until the decided number of frames is not reached collect frames
            if (i < 50)
            {
                // Read a new frame from video
                bool bSuccess = cap.read(frame);

                face_detector.img = frame;

                // Breaking the while loop if the frames cannot be captured
                if (camera_disconnection(bSuccess)) break;

                // Extract only the face
                cropedFrame = face_detector.extract_rectangle();

                antispoofing_detector.img = cropedFrame;
                                           
                // Check if the face is blurred
                bool blurred = face_detector.blur_detection();
                
                if (!blurred)
                {
                    // Save frame
                    imwrite(frames_path + "frame" + std::to_string(i) +".jpg", cropedFrame);
                    i++;
                } 
                imshow(window_name, frame);
            }
            else
            {
                if (pred == "Null")
                    antispoofing_detector.print_status(&frame, "Performing prediction...");

                else
                    antispoofing_detector.print_status(&frame, pred);

                imshow(window_name, frame);

                pred = antispoofing_detector.multiple_prediction(frames_path);
            }

            // Check when close webcam
            if (close_webcam()) break;
        }
    }
    

    return 0;

}





/* WRONG
bool read_and_check(VideoCapture cap, Mat frame, frontal_face_detector detector, AntiSpoofingDetection antispoofing_detector, FaceDetection face_detector)
{
    // Read a new frame from video
    bool bSuccess = cap.read(frame);

    // Breaking the while loop if the frames cannot be captured
    if (bSuccess == false) 
    {
        cout << "Video camera is disconnected" << endl;
        cin.get(); //Wait for any key press
        //break;
    }

    // Extract only the face
    Mat cropedFrame = face_detector.extract_rectangle(detector, frame);
                
                
    // Check if the face is blurred
    bool blurred = face_detector.blur_detection(cropedFrame);
    
    return blurred;
}
*/


//windowWidth=cv2.getWindowImageRect("myWindow")[2]
//windowHeight=cv2.getWindowImageRect("myWindow")[3]

//putText(temp, dim,  Point(x2, y2), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

/* To print laplacian image
    Mat abs_dst = face_detector.compute_laplacian(frame);
    const char* window_name = "Laplace Demo";
    imshow(window_name, abs_dst);
*/


/* To print rectangle and shape of the face detected
    // to extract rectangle
    std::vector<dlib::rectangle> faces = face_detector.detect_rectangle(detector, img);

    // to extract face
    //std::vector<full_object_detection> faces = face_detector.detect_shape(pose_model, detector, img);

    face_detector.print_rectangle_dlib(img, faces, output);
    //face_detector.print_rectangle_cv(detector, img);
    //face_detector.print_shape(img, faces);
*/



/* To monitor time
    auto start_SNN = chrono::high_resolution_clock::now();
    auto stop_SNN = chrono::high_resolution_clock::now();
    auto duration_SNN = chrono::duration_cast<chrono::milliseconds>(stop_SNN - start_SNN);
    cout << "Time taken to load SNN: "
         << duration_SNN.count() << " milliseconds" << endl;
*/