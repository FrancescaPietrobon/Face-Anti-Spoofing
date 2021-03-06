#include "../include/utilities.h"

using namespace cv;
using namespace std;
using namespace dlib;       
        

void print_status(Mat *img, string message, bool black)
{
    /// Prints the image pointed with a given message
    /** 
     * Arguments:
     *      img: the pointer to the image in which the message would be printed.
     *      message: the string containing the message to print.
     *      black: bool to choose if a black background it's nedded.
     * 
     *  Returns:
     *      None.
    */

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
    /// Checks if the camera is disconnected
    /** 
     * Arguments:
     *      bSuccess: bool telling if a frame it's read.
     * 
     *  Returns:
     *      None.
    */

    // Checks if the camera is disconnected
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
    /// Checks if Esc is pressed so the camera will be closed
    /** 
     * Arguments:
     *      None.
     * 
     *  Returns:
     *      None.
    */

    if (waitKey(1) == 27)
    {
        cout << "Esc key is pressed by user. Stoppig the video" << endl;
        return true;
    }
    return false;
}


int collect_frames(FaceDetection *face_detector, AntiSpoofingDetection *antispoofing_detector, string frames_path)
{
    /// Collects the required amouth of images and save them in the given folder
    /** 
     * Arguments:
     *      face_detector: pointer to a class object FaceDetection that collect all the data and functions required to
     *                     detect the face.
     *      antispoofing_detector: pointer to a class object AntiSpoofingDetection that collect all the data and functions required to
     *                     solve the Anti-Spoofing task.
     *      frame_path: string with the path of the folder where the frames must be saved
     * 
     *  Returns:
     *      Int with zero if all works fine, 1 if the camera was closed or disconnected.
    */

    int i = 1;
    // Until the decided number of frames is not reached collect frames
    while (i <= antispoofing_detector->n_img)
    {
        // Read a new frame from video
        bool bSuccess = face_detector->cap.read(face_detector->img);
        //imshow("Webcam", face_detector->img);

        // Stop collecting frames if the frames cannot be captured
        if (camera_disconnection(bSuccess)) return 1;

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
        imshow("Webcam", face_detector->img);
        waitKey(1);

        if (close_webcam()) return 1;
    }
    // After acquisition of the images required print "Performing prediction..."
    print_status(&face_detector->img, "Performing prediction...");
    imshow("Webcam", face_detector->img);
    waitKey(10);

    return 0;
}