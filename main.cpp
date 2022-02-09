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
    Ptr<ml::RTrees> svm = Algorithm::load<ml::RTrees> ("/home/fra/PROGETTO_PACS/Face-Anti-Spoofing/models/All_RF_opencv_final_net_lr5e-4.xml");

    // Load face detection and pose estimation models.
    frontal_face_detector detector = get_frontal_face_detector();
    

    Mat img;
    
    string window_name = "Image selected";

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
                    img_path = argv[2]; // before was argv[i++]: Increment 'i' so we don't get the argument as the next argv[i].
                    img = imread(img_path, IMREAD_COLOR);
                    namedWindow(window_name); //create a window
                    imshow(window_name, img);
                    
                    image_window win;

                    dlib::cv_image<dlib::bgr_pixel> cimg;
                    cimg = FaceDetection::OpenCVMatTodlib(img);

                    std::vector<dlib::rectangle> faces;
                    faces = FaceDetection::detect_rectangle(detector, cimg);

                    win.clear_overlay();
                    win.set_image(cimg);
                    win.add_overlay(faces, rgb_pixel(255,0,0)); //to display rectangle
                    //win.add_overlay(render_face_detections(shapes)); //to display shape

                    waitKey(0);

                    string output = make_prediction(img, cvNet, svm);
                    setWindowTitle(window_name, output);

                    waitKey(0);
                }
                else // Uh-oh, there was no argument to the destination option.
                { 
                    std::cerr << "--path option requires one argument." << std::endl;
                    return 1;
                }  
            }
            // POSSO AGGIUNGERE ANCHE PATH AI MODELLI
        }
    }
    else // If there is no path to the image open the webcam
    {
        namedWindow(window_name);
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

            imshow(window_name, frame);

            //FaceDetection::detect_rectangle(detector, frame);

            string output = make_prediction(frame, cvNet, svm);
            setWindowTitle(window_name, output);

            i += 1;

            
            //wait for for 100 ms until any key is pressed.  
            //If the 'Esc' key is pressed, break the while loop.
            //If the any other key is pressed, continue the loop 
            //If any key is not pressed withing 100 ms, continue the loop 
            if (waitKey(100) == 27)
            {
                cout << "Esc key is pressed by user. Stoppig the video" << endl;
                break;
            }
  
        }
        
    }

    return 0;

}




    /* To monitor time
    auto start_SNN = chrono::high_resolution_clock::now();
    auto stop_SNN = chrono::high_resolution_clock::now();
    auto duration_SNN = chrono::duration_cast<chrono::milliseconds>(stop_SNN - start_SNN);
    cout << "Time taken to load SNN: "
         << duration_SNN.count() << " milliseconds" << endl;
    */