#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <cassert>

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

    cout << world_rank << endl;
    GetPot cl(argc, argv);

    // Open json file with parameters
    std::ifstream config_doc("../src/data.json");
    assert(config_doc);

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

    VideoCapture cap;
    if (world_rank == 0)
    {
        // Open the default video camera
        cap.open(deviceID, apiID);
    }

    // Construct classes
    FaceDetection face_detector(detector, cap, ROI_dim);
    AntiSpoofingDetection antispoofing_detector(snn, ml, n_img, frames_path, world_rank, world_size);
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

        waitKey(5000);
    }
    else
    {
        // To see a realtime prediction 
        if (cl.search(2, "-e", "--example")  && world_rank == 0)
            final_prediction.predict_realtime();
        // To collect multiple frames and then make the final prediction
        else 
        {
            // If only one processor
            if (world_size == 1)
                final_prediction.predict_images(frames_path);
            else
            {
                string window_name = "Webcam";
                namedWindow( window_name, WINDOW_AUTOSIZE );
                int tot_real = 0;
                int i = 1;

                while (true)
                {
                    cout << "in while " + to_string(world_rank) << endl;
                    // Until the decided number of frames is not reached collect frames
                    if (world_rank == 0 && i <= n_img)
                    {
                        cout << "Collecte image " + to_string(i) << endl;
                        i = collect_frames(&face_detector, &antispoofing_detector, frames_path, i);
                        if (i == n_img + 3)
                        {
                            print_status(&face_detector.img, "Camera disconnected");
                            imshow(window_name, face_detector.img);
                            waitKey(5000);
                            return 1;
                        }  
                    }
                    else
                    { 
                        int count_real = 0;
                        for(int j = world_rank; j < n_img ; j+=world_size)
                        {
                            cout << "Predicting image " + to_string(j+1) + "by processor " + to_string(world_rank) << endl;
                            // Extract the images saved
                            antispoofing_detector.face = imread(frames_path + "frame" + std::to_string(j+1) +".jpg", IMREAD_COLOR);

                            // Check if the prediction for the image is 0: Real or 1: Fake
                            if (antispoofing_detector.value_prediction() == 0)
                                count_real += 1;
                        }

                        // Gather all partial averages down to the root process
                        int *sum_real = NULL;
                        if (world_rank == 0)
                            sum_real = new int[world_size];
                            
                        MPI_Gather(&count_real, 1, MPI_INT, sum_real, 1, MPI_INT, 0, MPI_COMM_WORLD);

                        // Compute the total average of all numbers.
                        if (world_rank == 0)
                        {
                            tot_real = antispoofing_detector.compute_sum_real(sum_real, world_size);

                            // Take the one with higher number of occurences
                            if (tot_real > n_img/2)
                                antispoofing_detector.pred = "Real";     
                            else
                                antispoofing_detector.pred = "Fake";

                            print_status(&face_detector.img, antispoofing_detector.pred);
                            imshow(window_name, face_detector.img);
                            waitKey(5000);
                        }

                        // Clean up
                        if (world_rank == 0)
                            delete(sum_real);
                                        
                        break;
                    }
                    if (close_webcam()) break; 
                }
                
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}


/*
    else
    {
        string window_name = "Webcam";
        int i = 1;

        while (true)
        {
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

    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;

}
*/
    /*
    // Open the default video camera
    VideoCapture cap;
    if (world_rank == 0 || (world_rank == world_size))
        cap.open(deviceID, apiID);

    int ROI_dim = 350;
    int n_img = 50;

    FaceDetection face_detector(detector, cap, ROI_dim);
    AntiSpoofingDetection antispoofing_detector(snn, ml, n_img, frames_path, world_rank, world_size);
    FinalPrediction final_prediction(&face_detector, &antispoofing_detector);

    cout << "Pre choice" << endl;

    if (cl.search(2, "-p", "--path"))
    {
        if (world_rank == 0)
        {
            face_detector.img = imread(im//#include <cstdio>

            // Make the prediction
            final_prediction.predict_image();

            waitKey(5000);
        }
    }
    else
    { cout << "Pre realtime or multiple" << endl;
        // Check if realtime prediction such as example or if prediction of multiple images simultaneously
        if (cl.search(2, "-e", "--example"))
        {
            if (world_rank == 0)
                final_prediction.predict_realtime();
            cout << "Example" << endl;
        }
        else
        {
            cout << "Pre predicting images" << endl;
            final_prediction.predict_images(frames_path);
        }

    } 
    */

    




/*
//MPI_Init (&argc, &argv);
    MPI_Init (NULL, NULL);
    int world_rank, world_size;
    MPI_Comm_size (MPI_COMM_WORLD, &world_rank);
    MPI_Comm_rank (MPI_COMM_WORLD, &world_size);
    //printf ("Hello form process %d of %d\n", world_rank, world_size);
    MPI_Finalize();
*/


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