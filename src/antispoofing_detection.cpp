#include "../include/antispoofing_detection.h"

using namespace cv;
using namespace std;
using namespace dlib;


AntiSpoofingDetection::AntiSpoofingDetection(dnn::Net _snn, Ptr<ml::RTrees> _ml):
snn(_snn), ml(_ml) {};


string AntiSpoofingDetection::single_prediction()
{
    float prediction = AntiSpoofingDetection::value_prediction();

    string output;

    if (prediction == 0)
        output = "Real";
    else
        output = "Fake";
    
    return output;
}


float AntiSpoofingDetection::value_prediction()
{
    // SNN prediction
    Mat blob = dnn::blobFromImage(face, 1, Size(256, 256), Scalar(0,0,0), true, false, CV_32F);
    snn.setInput(blob);
    Mat features = snn.forward();

    // ML Model prediction
    float prediction = ml->predict(features);

    return prediction;
}


string AntiSpoofingDetection::multiple_prediction(string frames_path)
{
    int real = 0;
    int fake = 0;

    string frame;

    for (int i=1; i<50; i++)
    {
        //Extract the images saved
        frame = frames_path + "frame" + std::to_string(i) +".jpg";
        Mat face = imread(frame, IMREAD_COLOR);

        // Make the prediction for every image
        float prediction = AntiSpoofingDetection::value_prediction();
        if (prediction == 0)
            real += 1;
        else
            fake += 1;
    }
    // Take the one with higher number of occurences
    if (real > fake)
        return "Real";
    else
        return "Fake";
}