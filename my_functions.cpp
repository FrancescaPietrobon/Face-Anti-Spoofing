#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <iostream>
#include <string>

#include "my_functions.h"

using namespace cv;
using namespace std;


string make_prediction(Mat img, dnn::Net cvNet, Ptr<ml::RTrees> svm)
{
        // SNN prediction      

        //image, output size, mean values to subtract from channels, swap first and last channels, cropped after resize or not, depth of output blob
        Mat blob = dnn::blobFromImage(img, 1, Size(256, 256), Scalar(0,0,0), true, false, CV_32F);
        cvNet.setInput(blob);
        Mat features = cvNet.forward();
        //cout << "features = " << endl << " "  << features << endl << endl;

        // ML Model prediction
        float prediction = svm->predict(features);

        string output;
        if (prediction == 0)
            output = "Real";
        else
            output = "Fake";
        
        return output;
}