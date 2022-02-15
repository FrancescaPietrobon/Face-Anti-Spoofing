#include "../include/utilities.h"

using namespace cv;
using namespace std;
using namespace dlib;       
        

void print_status(Mat *frame, string message, bool black)
{
    if (black)
    {
        Mat black = Mat::zeros(Size(frame->cols,frame->rows),CV_8UC1);
        putText(black, message, Point(200,200), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255,255), 2, LINE_AA);
        black.copyTo(*frame);
    }
    else
        putText(*frame, message, Point(200,200), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255,255), 2, LINE_AA);
    
}


bool camera_disconnection(bool bSuccess)
{
    // Breaking the while loop if the frames cannot be captured
    if (bSuccess == false) 
    {
        cout << "Video camera is disconnected" << endl;
        cin.get(); //Wait for any key press
        return true;
    }
    return false;
}


bool close_webcam()
{
    if (waitKey(1) == 27)
    {
        cout << "Esc key is pressed by user. Stoppig the video" << endl;
        return true;
    }
    return false;
}
