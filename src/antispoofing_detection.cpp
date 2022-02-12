#include "../include/antispoofing_detection.h"

using namespace cv;
using namespace std;
using namespace dlib;



string AntiSpoofingDetection::single_prediction(Mat img, dnn::Net snn, Ptr<ml::RTrees> ml)
{
    float prediction = AntiSpoofingDetection::value_prediction(img, snn, ml);

    string output;

    if (prediction == 0)
        output = "Real";
    else
        output = "Fake";
    
    return output;
}

float AntiSpoofingDetection::value_prediction(Mat img, dnn::Net snn, Ptr<ml::RTrees> ml)
{
    // SNN prediction
    //image, output size, mean values to subtract from channels, swap first and last channels, cropped after resize or not, depth of output blob
    Mat blob = dnn::blobFromImage(img, 1, Size(256, 256), Scalar(0,0,0), true, false, CV_32F);
    snn.setInput(blob);
    Mat features = snn.forward();

    // ML Model prediction
    float prediction = ml->predict(features);

    return prediction;
}

string AntiSpoofingDetection::multiple_prediction(string frames_path, dnn::Net snn, Ptr<ml::RTrees> ml)
{
    int real = 0;
    int fake = 0;

    for (int i=1; i<50; i++)
    {
        string frame = frames_path + "frame" + std::to_string(i) +".jpg";
        Mat img = imread(frame, IMREAD_COLOR);
        float prediction = AntiSpoofingDetection::value_prediction(img, snn, ml);
        if (prediction == 0)
            real += 1;
        else
            fake += 1;
    }

    if (real > fake)
        return "Real";
    else
        return "Fake";
}


void AntiSpoofingDetection::print_status(Mat *frame, string message)
{
    Mat black = Mat::zeros(Size(frame->cols,frame->rows),CV_8UC1);
    putText(black, message, Point(200,200), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255,255), 2, LINE_AA);
    black.copyTo(*frame);
}
