#include "../include/antispoofing_detection.h"

using namespace cv;
using namespace std;
using namespace dlib;


AntiSpoofingDetection::AntiSpoofingDetection(Mat _face, dnn::Net _snn, Ptr<ml::RTrees> _ml, string _pred): //CONTROLLA _pred
face(_face), snn(_snn), ml(_ml), pred("Null") {};


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
    //image, output size, mean values to subtract from channels, swap first and last channels, cropped after resize or not, depth of output blob
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

    for (int i=1; i<50; i++)
    {
        string frame = frames_path + "frame" + std::to_string(i) +".jpg";
        Mat face = imread(frame, IMREAD_COLOR);
        float prediction = AntiSpoofingDetection::value_prediction();
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
