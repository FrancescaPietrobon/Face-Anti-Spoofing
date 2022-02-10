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

#include "my_functions.h"
#include "face_detection.h"

using namespace std;
using namespace cv;
using namespace dlib;


int main(int argc, char* argv[])
{

    // Load SNN
    string weights = "/home/fra/PROGETTO_PACS/Face-Anti-Spoofing/models/Frozen_graph_All_final_net_5e-4.pb";
    dnn::Net cvNet = cv::dnn::readNetFromTensorflow(weights);

    // Load ML Model
    Ptr<ml::RTrees> rf = Algorithm::load<ml::RTrees> ("/home/fra/PROGETTO_PACS/Face-Anti-Spoofing/models/All_RF_opencv_final_net_lr5e-4.xml");

    // Load face detection and pose estimation models.
    frontal_face_detector detector = get_frontal_face_detector();

    shape_predictor pose_model;
    deserialize("/home/fra/PROGETTO_PACS/Face-Anti-Spoofing/models/shape_predictor_68_face_landmarks.dat") >> pose_model;
    

    Mat img;
    
    //string window_name = "Image selected";
    //namedWindow(window_name); //create a window
    //imshow(window_name, img);

    if (argc > 2) // If there is a path for the image take it
    {
        std::string img_path;

        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];
            if ((arg == "-p") || (arg == "--path"))
            {
                if (i + 1 < argc) // Make sure we aren't at the end of argv!
                {
                    img_path = argv[++i]; // Increment 'i' so we don't get the argument as the next argv[i].
                    img = imread(img_path, IMREAD_COLOR);

                    string pred = make_prediction(img, cvNet, rf);

                    FaceDetection::CVprint_rectangle(detector, img, pred);
                    waitKey(5000);

                }
                else // Uh-oh, there was no argument to the destination option.
                { 
                    std::cerr << "--path option requires one argument." << std::endl;
                    return 1;
                }  
            }
            else if ((arg == "-wr") || (arg == "--webcamrealtime"))
            {
                /*
                img_path = argv[++i]; // Increment 'i' so we don't get the argument as the next argv[i].
                img = imread(img_path, IMREAD_COLOR);

                string pred = make_prediction(img, cvNet, svm);

                FaceDetection::CVprint_rectangle(detector, img, pred);
                */
            }
        }
    }
    else // If there is no path to the image open the webcam
    {
        //namedWindow(window_name);
        //Open the default video camera
        VideoCapture cap;
        int deviceID = 0;         // 0 = open default camera
        int apiID = CAP_ANY;      // 0 = autodetect default API
        // open selected camera using selected API
        cap.open(deviceID, apiID);

        int i = 0;
        while (true)
        {
            Mat frame;
            bool bSuccess = cap.read(frame); // read a new frame from video 

            // Breaking the while loop if the frames cannot be captured
            if (bSuccess == false) 
            {
                cout << "Video camera is disconnected" << endl;
                cin.get(); //Wait for any key press
                break;
            }

            // Save frame
            //imwrite("/home/fra/Project/Frames/frame" + std::to_string(i+1) +".jpg", frame);

            //imshow(window_name, frame);
            string pred = make_prediction(frame, cvNet, rf);

            FaceDetection::CVprint_rectangle(detector, frame, pred);

            i += 1;

            if (waitKey(1) == 27)
            {
                cout << "Esc key is pressed by user. Stoppig the video" << endl;
                break;
            }
  
        }
        
    }

    return 0;

}

/* To print rectangle and shape of the face detected
    // to extract rectangle
    std::vector<dlib::rectangle> faces = FaceDetection::detect_rectangle(detector, img);

    // to extract face
    //std::vector<full_object_detection> faces = FaceDetection::detect_shape(pose_model, detector, img);

    FaceDetection::print_rectangle(img, faces, output);
    //FaceDetection::CVprint_rectangle(detector, img);
    //FaceDetection::print_shape(img, faces);
*/



/* To monitor time
    auto start_SNN = chrono::high_resolution_clock::now();
    auto stop_SNN = chrono::high_resolution_clock::now();
    auto duration_SNN = chrono::duration_cast<chrono::milliseconds>(stop_SNN - start_SNN);
    cout << "Time taken to load SNN: "
         << duration_SNN.count() << " milliseconds" << endl;
*/