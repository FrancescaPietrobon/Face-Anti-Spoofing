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

#include "antispoofing_detection.h"
#include "face_detection.h"

using namespace std;
using namespace cv;
using namespace dlib;


int main(int argc, char* argv[])
{
    //windowWidth=cv2.getWindowImageRect("myWindow")[2]
    //windowHeight=cv2.getWindowImageRect("myWindow")[3]

    string frames_path = "/home/fra/Project/Frames/";

    // Load SNN
    string weights = "/home/fra/PROGETTO_PACS/Face-Anti-Spoofing/models/Frozen_graph_All_final_net_5e-4.pb";
    dnn::Net cvNet = cv::dnn::readNetFromTensorflow(weights);

    // Load ML Model
    Ptr<ml::RTrees> rf = Algorithm::load<ml::RTrees> ("/home/fra/PROGETTO_PACS/Face-Anti-Spoofing/models/All_RF_opencv_final_net_lr5e-4.xml");

    // Load face detection and pose estimation models.
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor pose_model;
    deserialize("/home/fra/PROGETTO_PACS/Face-Anti-Spoofing/models/shape_predictor_68_face_landmarks.dat") >> pose_model;

    // Set webcam options
    int deviceID = 0;         // 0 = open default camera
    int apiID = CAP_ANY;      // 0 = autodetect default API
    
    //string window_name = "Image selected";
    //namedWindow(window_name); //create a window
    //imshow(window_name, img);

    if (argc > 1)
    {
        std::string img_path;

        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];

            // If a path to saved image is given, take the prediction for that image
            if ((arg == "-p") || (arg == "--path"))
            {
                if (i + 1 < argc) // Make sure we aren't at the end of argv
                {
                    // Extract image path
                    img_path = argv[++i];

                    // Read image
                    Mat img = imread(img_path, IMREAD_COLOR);

                    // Extract only the face
                    Mat cropedImage = FaceDetection::extract_face_rectangle(detector, img);

                    // To print Cropped Image
                    //imshow( "Cropped Image", cropedImage);
                    // waitKey(5000);

                    // Check if the face is blurred
                    bool blurred = FaceDetection::blur_detection(cropedImage);

                    if (blurred)
                    {
                        // AGGIUNGI PLOT IMG CON SCRITTA BLURRED IMAGE
                        std::cerr << "Blurred image." << std::endl;
                        return 1;
                    }
                        
                    // Make prediction for the face
                    string pred = AntiSpoofingDetection::make_prediction(cropedImage, cvNet, rf);

                    // Print the image with prediction, dimensions, rectangles of face detected and of face considered to make the prediction
                    FaceDetection::cv_print_rectangle(detector, img, blurred, pred);
                    waitKey(5000);
                    
                }
                else // Uh-oh, there was no argument to the destination option.
                { 
                    std::cerr << "--path option requires one argument." << std::endl;
                    return 1;
                }  
            }
            // If a webcam realtime option is selected, make the prediction realtime for each frame 
            else if ((arg == "-wr") || (arg == "--webcamrealtime"))
            {
                // Open the default video camera
                VideoCapture cap;

                // Open selected camera using selected API
                cap.open(deviceID, apiID);

                while (true)
                {
                    Mat frame;
                    // Read a new frame from video 
                    bool bSuccess = cap.read(frame); 
                    //imshow("Webcam", frame);

                    // Breaking the while loop if the frames cannot be captured
                    if (bSuccess == false) 
                    {
                        cout << "Video camera is disconnected" << endl;
                        cin.get(); //Wait for any key press
                        break;
                    }

                    // Extract only the face
                    Mat cropedImage = FaceDetection::extract_face_rectangle(detector, frame);

                    // Check if the face is blurred
                    bool blurred = FaceDetection::blur_detection(cropedImage);
                    
                    // If the face is not blurred print make the prediction and print them, otherwise print "Blurred"
                    if (!blurred)
                    {
                        string pred = AntiSpoofingDetection::make_prediction(cropedImage, cvNet, rf);
                        FaceDetection::cv_print_rectangle(detector, frame, blurred, pred);
                    }
                    else
                        FaceDetection::cv_print_rectangle(detector, frame, blurred);

                    // Check when close webcam
                    if (waitKey(1) == 27)
                    {
                        cout << "Esc key is pressed by user. Stoppig the video" << endl;
                        break;
                    }
                }
            }
        }
    }
    // If there is no option are selected, collect the frame images and than make the prediction
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

        while (true)
        {
            // Until the decided number of frames is not reached collect frames
            if (i < 50)
            {
                // Read a new frame from video
                bool bSuccess = cap.read(frame);

                // Breaking the while loop if the frames cannot be captured
                if (bSuccess == false) 
                {
                    cout << "Video camera is disconnected" << endl;
                    cin.get(); //Wait for any key press
                    break;
                }

                // Extract only the face
                Mat cropedFrame = FaceDetection::extract_face_rectangle(detector, frame);
                
                
                // Check if the face is blurred
                bool blurred = FaceDetection::blur_detection(cropedFrame);
                
                
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
                    AntiSpoofingDetection::print_status(frame, "Performing prediction...", window_name);
                else
                    AntiSpoofingDetection::print_status(frame, pred, window_name);

                pred = AntiSpoofingDetection::multiple_prediction(frames_path, cvNet, rf);
                
            }

            // Check when close webcam
            if (waitKey(1) == 27)
            {
                cout << "Esc key is pressed by user. Stoppig the video" << endl;
                break;
            }
  
        }
        
    }

    return 0;

}


//putText(temp, dim,  Point(x2, y2), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

/* To print laplacian image
    Mat abs_dst = FaceDetection::laplacian_plot(frame);
    const char* window_name = "Laplace Demo";
    imshow(window_name, abs_dst);
*/


/* To print rectangle and shape of the face detected
    // to extract rectangle
    std::vector<dlib::rectangle> faces = FaceDetection::detect_rectangle(detector, img);

    // to extract face
    //std::vector<full_object_detection> faces = FaceDetection::detect_shape(pose_model, detector, img);

    FaceDetection::dlib_print_rectangle(img, faces, output);
    //FaceDetection::cv_print_rectangle(detector, img);
    //FaceDetection::print_shape(img, faces);
*/



/* To monitor time
    auto start_SNN = chrono::high_resolution_clock::now();
    auto stop_SNN = chrono::high_resolution_clock::now();
    auto duration_SNN = chrono::duration_cast<chrono::milliseconds>(stop_SNN - start_SNN);
    cout << "Time taken to load SNN: "
         << duration_SNN.count() << " milliseconds" << endl;
*/