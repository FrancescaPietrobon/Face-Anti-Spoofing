#include "../include/final_prediction.h"
#include "../include/antispoofing_detection.h"
#include "../include/face_detection.h"

using namespace cv;
using namespace std;
using namespace dlib;       
        
 
void FinalPrediction::predict_image(string img_path, frontal_face_detector detector, dnn::Net snn, Ptr<ml::RTrees> ml)
{

    Mat img = imread(img_path, IMREAD_COLOR);

    // Extract only the face
    Mat cropedImage = FinalPrediction::extract_rectangle(detector, img);

    // Check if the face is blurred
    bool blurred = FinalPrediction::blur_detection(cropedImage);

    // If the face is not blurred print make the prediction and print them, otherwise print "Blurred"
    if (!blurred)
    {
        // Make prediction for the face
        string pred = FinalPrediction::single_prediction(cropedImage, snn, ml);
        FinalPrediction::print_rectangle_cv(detector, img, blurred, pred);
    }
    else
    {
        // Print the image with prediction (or "Blorred"), dimensions, rectangles of face detected and of face considered to make the prediction
        FinalPrediction::print_rectangle_cv(detector, img, blurred);
    }
} 
        