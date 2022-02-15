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
#include "include/utilities.h"

using namespace std;
using namespace cv;
using namespace dlib;


int main(int argc, char* argv[])
{
    GetPot cl(argc, argv);

    string frames_path = "/home/fra/Project/Frames/";
    string SNN_weights = "/home/fra/PROGETTO_PACS/Face-Anti-Spoofing/models/Frozen_graph_All_final_net_5e-4.pb";
    string ML_weights = "/home/fra/PROGETTO_PACS/Face-Anti-Spoofing/models/All_RF_opencv_final_net_lr5e-4.xml";
    string face_detect = "/home/fra/PROGETTO_PACS/Face-Anti-Spoofing/models/shape_predictor_68_face_landmarks.dat";
    string img_path = "/home/fra/Scaricati/2022-02-08-163449.jpg";

    // Set webcam options
    int deviceID = 0;       // 0 = open default camera
    int apiID = CAP_ANY;    // 0 = autodetect default API

    int n_img = 50;

    // Load SNN
    dnn::Net snn = cv::dnn::readNetFromTensorflow(SNN_weights);

    // Load ML Model
    Ptr<ml::RTrees> ml = Algorithm::load<ml::RTrees> (ML_weights);

    // Load face detection and pose estimation models
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor pose_model;
    deserialize(face_detect) >> pose_model;

    // Open the default video camera
    VideoCapture cap;

    Mat img;
    Mat face;
    Mat cropedImage;
    bool  blurred;
    string pred = "Null";
    int ROI_dim = 200;

    FaceDetection face_detector(detector, img, cropedImage, blurred, cap, ROI_dim);
    AntiSpoofingDetection antispoofing_detector(face, snn, ml, pred);


    if (cl.search(2, "-p", "--path"))
    {
        face_detector.img = imread(img_path, IMREAD_COLOR);

        FinalPrediction final_prediction(face_detector, antispoofing_detector);
        
        // Make the prediction
        final_prediction.predict_image();

        waitKey(5000);
    }
    else
    {


        // Open selected camera using selected API
        cap.open(deviceID, apiID);

        FinalPrediction final_prediction(face_detector, antispoofing_detector);

        // Check if realtime prediction such as example or if prediction of multiple images simultaneously
        if (cl.search(2, "-e", "--example"))
            final_prediction.predict_realtime(cap);
        else
            final_prediction.predict_images(cap, n_img, frames_path);

    } 
    

    return 0;

}




//windowWidth=cv2.getWindowImageRect("myWindow")[2]
//windowHeight=cv2.getWindowImageRect("myWindow")[3]

/* To use GetPot to extract parameters
    string filename = "../src/data.dat";
    GetPot parser(filename.c_str());

    string frames_path = parser(frames_path.c_str(), "/home/fra/Project/Frames/");
    string SNN_weights = parser(SNN_weights.c_str(), "/home/fra/PROGETTO_PACS/Face-Anti-Spoofing/models/Frozen_graph_All_final_net_5e-4.pb");
    string ML_weights = parser(ML_weights.c_str(), "/home/fra/PROGETTO_PACS/Face-Anti-Spoofing/models/All_RF_opencv_final_net_lr5e-4.xml");
    string face_detect = parser(face_detect.c_str(), "/home/fra/PROGETTO_PACS/Face-Anti-Spoofing/models/shape_predictor_68_face_landmarks.dat");
    string example_path = parser(example.c_str(), "/home/fra/Scaricati/2022-02-08-163449.jpg");

    // Set webcam options
    int deviceID = parser("deviceID", 0); // 0 = open default camera

    // Extract path
    string img_path = cl.next(img_path.c_str());
*/


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