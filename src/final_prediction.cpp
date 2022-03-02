#include "../include/final_prediction.h"
#include "../include/antispoofing_detection.h"
#include "../include/face_detection.h"
#include "../include/utilities.h"
#include <mpi/mpi.h>

using namespace cv;
using namespace std;
using namespace dlib;       
        

FinalPrediction::FinalPrediction(FaceDetection *_face_detector, AntiSpoofingDetection *_antispoofing_detector):
face_detector(_face_detector), antispoofing_detector(_antispoofing_detector) {};


void FinalPrediction::predict_image()
{
    /// Makes the final prediction (real or fake) for a single image
    /** 
     * Arguments:
     *      None.
     * 
     *  Returns:
     *      None.
    */

    // If the face is detected
    if (face_detector->detect_rectangle())
    {
        // If the face detected is not out of bounds
        if (!face_detector->out_of_bounds())
        {
            // Extract only the face
            antispoofing_detector->face = face_detector->extract_rectangle();

            // If the face is not blurred, make the prediction and print them, otherwise print "Blurred"
            if (!face_detector->blur_detection())
            {
                // Make prediction for the face
                face_detector->print_rectangle_cv(antispoofing_detector->single_prediction());
            }
            else
            {
                // Print the image with prediction (or "Blorred"), dimensions, rectangles of face detected and of face considered to make the prediction
                face_detector->print_rectangle_cv();
            } 
        }
    }
    imshow("Image", face_detector->img);
}


int FinalPrediction::predict_realtime()
{
    /// Makes the final prediction (real or fake) realtime for each frame collected by the camera
    /** 
     * Arguments:
     *      None.
     * 
     *  Returns:
     *      None.
    */

    while (true)
    {
        // Read a new frame from video 
        bool bSuccess = face_detector->cap.read(face_detector->img);

        // Breaking the while loop if the frames cannot be captured
        if (camera_disconnection(bSuccess)) return 1;
            
        // Make the prediction
        FinalPrediction::predict_image();

        // Check when close webcam
        if (close_webcam()) return 1;
    }
    return 0;
} 


int FinalPrediction::predict_multiple_frames(string frames_path, int world_rank, int world_size)
{
    /// Makes the final prediction (real or fake) for a fixed number of frames collected by 
    /// the camera and saved in the given folder
    /** 
     * Arguments:
     *      frames_path: path where the frames will be collected before computing the prediction.
     *      world_rank: int of the current number of processor
     *      world_size: int of the number of processors available.
     * 
     *  Returns:
     *      Int that will be 0 if all works fine and 1 if camera disconnects or is closed.
    */

    // If only one processor use the non parallel version otherwise the parallel ones
    if (world_size == 1)
        FinalPrediction::predict_images(frames_path);          
    else
        FinalPrediction::predict_images_par(frames_path, world_rank, world_size);

    return 0;
}


int FinalPrediction::predict_images(string frames_path)
{
    /// Makes the final prediction (real or fake) for multiple images
    /** 
     * Arguments:
     *      frames_path: path where the frames will be collected before computing the prediction
     * 
     *  Returns:
     *      Int that will be 0 if all works fine and 1 if camera disconnects or is closed.
    */

    // Collect the given number of frames
    int collected_images = collect_frames(face_detector, antispoofing_detector, frames_path);
    if (collected_images == 1)
        return 1;

    // Compute the overall prediction
    antispoofing_detector->pred = antispoofing_detector->multiple_prediction();

    // Print the prediction
    print_status(&face_detector->img, antispoofing_detector->pred);
    imshow("Webcam", face_detector->img);
    waitKey(5000);
    
    return 0;
}


int FinalPrediction::predict_images_par(string frames_path, int world_rank, int world_size)
{
    /// Makes the final prediction (real or fake) for multiple images in parallel
    /** 
     * Arguments:
     *      frames_path: path where the frames will be collected before computing the prediction.
     *      world_rank: int of the current number of processor
     *      world_size: int of the number of processors available.
     * 
     *  Returns:
     *      Int that will be 0 if all works fine and 1 if camera disconnects or is closed.
    */

    int collected_images;
    int tot_real = 0;
                
    // If in rank 0 frames are colected
    if (world_rank == 0)
    {
        // Collect all the images
        collected_images = collect_frames(face_detector, antispoofing_detector, frames_path);

        // If collected_images returns one the camera was closed or disconnected
        if (collected_images == 1)
            return 1;
    }
                
    // Broadcast the value of collected_images to all the processors
    MPI_Bcast(&collected_images, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // If the images are collected all the processors can make the prediction
    if (collected_images == 0)
    { 
        // Compute the prediction for the images saved
        int count_real =  antispoofing_detector->multiple_prediction_par(world_rank, world_size);

        // Gather all partial averages down to the root process
        int *sum_real = NULL;
        if (world_rank == 0)
            sum_real = new int[world_size];
                                
        MPI_Gather(&count_real, 1, MPI_INT, sum_real, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Compute the total average of all numbers.
        if (world_rank == 0)
        {
            tot_real = antispoofing_detector->compute_sum_real(sum_real, world_size);

            // Take the one with higher number of occurences
            if (tot_real > antispoofing_detector->n_img/2)
                antispoofing_detector->pred = "Real";     
            else
                antispoofing_detector->pred = "Fake";

            // Print the the status in the image
            print_status(&face_detector->img, antispoofing_detector->pred);
            imshow("Webcam", face_detector->img);
            waitKey(5000);
        }

        // Clean up
        if (world_rank == 0)
           delete(sum_real);
                                            
        if (close_webcam()) return 1; 
    }
    return 0;
}