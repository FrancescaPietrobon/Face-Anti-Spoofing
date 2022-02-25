#include "../include/utilities.h"

using namespace cv;
using namespace std;
using namespace dlib;       
        

void print_status(Mat *img, string message, bool black)
{
    // https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga3d2abfcb995fd2db908c8288199dba82

    int fontFace = FONT_HERSHEY_SIMPLEX;
    double fontScale = 1;
    int thickness = 2;

    // Get boundary of the text
    int baseline=0;
    Size textSize = getTextSize(message, fontFace, fontScale, thickness, &baseline);
    baseline += thickness;

    // center the text
    Point textOrg((img->cols - textSize.width)/2, (img->rows + textSize.height)/2);

    if (black)
    {
        Mat black = Mat::zeros(Size(img->cols,img->rows),CV_8UC1);
        putText(black, message, textOrg, fontFace, fontScale, Scalar(255,255,255), thickness, LINE_AA);
        black.copyTo(*img);
    }
    else
        putText(*img, message, textOrg, fontFace, fontScale, Scalar(255,255,255), thickness, LINE_AA);
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


int collect_frames(FaceDetection *face_detector, AntiSpoofingDetection *antispoofing_detector, string frames_path, int i)
{
    namedWindow( "Webcam", WINDOW_AUTOSIZE );
    // Until the decided number of frames is not reached collect frames
    if (i <= antispoofing_detector->n_img)
    {
        // Read a new frame from video
        bool bSuccess = face_detector->cap.read(face_detector->img);

        // Stop collecting frames if the frames cannot be captured
        if (camera_disconnection(bSuccess)) return (antispoofing_detector->n_img + 3);

        // If the face is detected
        if (face_detector->detect_rectangle())
        {
            // If the face detected is not out of bounds
            if (!face_detector->out_of_bounds())
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
    // After acquisition of the images required print "Performing prediction..."
    if (i == antispoofing_detector->n_img + 1)
    {
        print_status(&face_detector->img, "Performing prediction...");
        i++;
        waitKey(100);
    }

    imshow("Webcam", face_detector->img);
    return i;
}




/*
Mat print_image(VideoCapture cap, string message)
{
    // https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga3d2abfcb995fd2db908c8288199dba82

    int fontFace = FONT_HERSHEY_SIMPLEX;
    double fontScale = 1;
    int thickness = 2;

    // Get boundary of the text
    int baseline=0;
    Size textSize = getTextSize(message, fontFace, fontScale, thickness, &baseline);
    baseline += thickness;

    // center the text
    Point textOrg((cap.width - textSize.width)/2, (cap.height + textSize.height)/2);

    Mat black = Mat::zeros(Size(cap.width,cap.height),CV_8UC1);
    putText(black, message, textOrg, fontFace, fontScale, Scalar(255,255,255), thickness, LINE_AA);
    
    return black;
}
*/