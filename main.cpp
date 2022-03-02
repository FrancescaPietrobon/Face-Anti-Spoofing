#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <iostream>
#include <string>
#include <fstream>

#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

#include <mpi/mpi.h>

#include <json/json.h>

#include "GetPot"

#include "include/antispoofing_detection.h"
#include "include/face_detection.h"
#include "include/final_prediction.h"
#include "include/utilities.h"

using namespace std;
using namespace cv;
using namespace dlib;


int main(int argc, char* argv[])
{  
    MPI_Init (NULL, NULL);
    int world_rank, world_size;
    MPI_Comm_size (MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);

    GetPot cl(argc, argv);
    
    // Open json file with parameters
    std::fstream config_doc;
    config_doc.open ("../src/data.json", std::ios::in);
    Json::Value root;
    config_doc >> root;

    // Load parameters
    string frames_path = root["frames_path"].asString();
    string SNN_weights = root["SNN_weights"].asString();
    string ML_weights = root["ML_weights"].asString();
    string face_detect = root["face_detect"].asString();
    string example_path = root["example_path"].asString();
    int ROI_dim = root["ROI_dim"].asInt(); 
    int n_img = root["n_img"].asInt();

    // Set webcam options
    int deviceID = root["deviceID"].asInt();      // 0 = open default camera
    int apiID = CAP_ANY;                          // detect APIs

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
    if (world_rank == 0)
        cap.open(deviceID, apiID);

    // Construct classes
    FaceDetection face_detector(detector, cap, ROI_dim);
    AntiSpoofingDetection antispoofing_detector(snn, ml, n_img, frames_path);
    FinalPrediction final_prediction(&face_detector, &antispoofing_detector);
           
    // To see the prediction of a single image
    if (cl.search(2, "-p", "--path") && world_rank == 0)
    {
        // If at runtime a path is given use that image otherwise use the provided image
        string img_selected = cl.next(example_path.c_str());

        // Read the image
        face_detector.img = imread(img_selected, IMREAD_COLOR);
            
        // Perform the prediction
        final_prediction.predict_image();

        // Let see the prediction before close the image
        waitKey(5000);
    }
    else
    {
        // To see a realtime prediction 
        if (cl.search(2, "-e", "--example")  && world_rank == 0)
            final_prediction.predict_realtime();
        // To collect multiple frames and then make the final prediction
        else 
            final_prediction.predict_multiple_frames(frames_path, world_rank, world_size);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}