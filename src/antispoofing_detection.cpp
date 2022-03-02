#include "../include/antispoofing_detection.h"
#include <mpi/mpi.h>


using namespace cv;
using namespace std;
using namespace dlib;


AntiSpoofingDetection::AntiSpoofingDetection(dnn::Net _snn, Ptr<ml::RTrees> _ml, int _n_img, string _frames_path):
snn(_snn), ml(_ml), n_img(_n_img), frames_path(_frames_path) {};


string AntiSpoofingDetection::single_prediction()
{
    /// After detecting if the prediction of the image is 0 (Real) or 1 (Fake),
    /// converts it in a string prediction.
    /** 
     * Arguments:
     *      None.
     * 
     *  Returns:
     *      String with the final prediction: "Real" or "Fake".
    */

    int prediction = AntiSpoofingDetection::value_prediction();

    string output;

    if (prediction == 0)
        output = "Real";
    else
        output = "Fake";
    
    return output;
}


int AntiSpoofingDetection::value_prediction()
{
    /// Uses the Siamese Neural Network saved to compute the features,
    /// then makes the prediction with the Machine Learning model saved.
    /** 
     * Arguments:
     *      None.
     * 
     *  Returns:
     *      Int with 0 if the image is predicted as real and 1 if it is predicted as fake.
    */

    // SNN prediction
    Mat blob = dnn::blobFromImage(face, 1, Size(256, 256), Scalar(0,0,0), true, false, CV_32F);
    snn.setInput(blob);

    // Uncomment to see the time needed to load the SNN
    //auto start_SNN = chrono::high_resolution_clock::now();

    Mat features = snn.forward();

    //auto stop_SNN = chrono::high_resolution_clock::now();
    //auto duration_SNN = chrono::duration_cast<chrono::milliseconds>(stop_SNN - start_SNN);
    //cout << "Time taken to load the SNN: " << duration_SNN.count() << " milliseconds" << endl;

    // ML Model prediction
    int prediction = ml->predict(features);

    return prediction;
}


string AntiSpoofingDetection::multiple_prediction()
{
    /// Extracts all the saved images, makes the prediction for each one and takes as final prediction
    /// the one with majority of the occurrences.
    /** 
     * Arguments:
     *      None.
     * 
     *  Returns:
     *      String with the prediction so "Real" or "Fake".
    */

    int count_real = 0;

    // For each image take the prediction and update the count of real images.
    for (int i=1; i<n_img; i++)
        count_real = AntiSpoofingDetection::one_pred(i, count_real);

    // Take the one with higher number of occurences
    if (count_real > n_img/2)
        return "Real";
    else
        return "Fake";
}



int AntiSpoofingDetection::multiple_prediction_par(int world_rank, int world_size)
{
    /// Extracts all the saved images, makes the prediction for each one and count the real images
    /** 
     * Arguments:
     *      world_rank: int of the current number of processor
     *      world_size: int of the number of processors available.
     * 
     *  Returns:
     *      Int of the number of real images detected.
    */

   int count_real = 0;

    // For each image take the prediction and update the count of real images.
   for(int j = world_rank; j < n_img ; j+=world_size)
        count_real = AntiSpoofingDetection::one_pred(j+1, count_real);

    return count_real;
}


int AntiSpoofingDetection::one_pred(int i, int count_real)
{
    /// Computes the prediction of a single image
    /** 
     * Arguments:
     *      i: index of the image that need to be predicted.
     *      count_real: int of the number of real images detected.
     * 
     *  Returns:
     *      Int of the current total number of real images.
    */

    // Extract the images saved
    AntiSpoofingDetection::face = imread(frames_path + "frame" + std::to_string(i) +".jpg", IMREAD_COLOR);
                        
    // Check if the prediction for the image is 0: Real or 1: Fake
    if (AntiSpoofingDetection::value_prediction() == 0)
            count_real += 1;

    return count_real;
}


int AntiSpoofingDetection::compute_sum_real(int *sum_real, int world_size)
{
    /// Computes the sum of the number of real images detected by all the processors.
    /** 
     * Arguments:
     *      sum_real: pointer to the number of real images.
     *      world_size: int of the number of processors available.
     * 
     *  Returns:
     *      Int of the total number of real images in all the processors.
    */

    int tot_real = 0;

    // Sum all the real detected by all the processors
    for (int i=0; i<world_size; i++)
        tot_real = tot_real + sum_real[i];
    
    return tot_real;
}



// UNUSED FUNCTIONS

/*
int* AntiSpoofingDetection::create_indexes(int elements_per_proc, int world_size)
{
    /// Creates a matrix with indexes of the saved images. It is used to split
    /// the images between the processors. 
    * 
     * Arguments:
     *      elements_per_proc: int of the number of elements that every processor
     *                         have to analyze.
     *      world_size: int of the number of processors available.
     * 
     *  Returns:
     *      Pointer to the matrix of indexes splitted.
    

    int* img_indexes = new int[elements_per_proc * world_size];
    int count = 0;

    // Fill the matrix with the indexes for every processor
    for (int i = 0; i < elements_per_proc; i++)
        for (int j = 0; j < world_size; j++)
            *(img_indexes + i * world_size + j) = ++count;

    return img_indexes;
}


int AntiSpoofingDetection::compute_real(int *sub_indexes, int elements_per_proc)
{
    /// Extracts all the saved images for the given partial indexes, makes the prediction for each one
    /// and collects the number of occurences for real images.
    * 
     * Arguments:
     *      sub_indexes: pointer of the partial indexes of the images used by one processor.
     *      elements_per_proc: int of the number of element used by one processor.
     * 
     *  Returns:
     *      Int of the number of real images in the inspected processor.
    

    int real = 0;
    string frame;

    for (int i=0; i<elements_per_proc; i++)
    {
        // Extract the images saved
        frame = frames_path + "frame" + std::to_string(sub_indexes[i]) +".jpg";
        face = imread(frame, IMREAD_COLOR);

        // Check if the prediction for the image is 0: Real or 1: Fake
        if (AntiSpoofingDetection::value_prediction() == 0)
            real += 1;
    }
    
    return real;
}
*/

/*
string AntiSpoofingDetection::MPI_multiple_prediction()
{
    int elements_per_proc = int(n_img / world_size);

    int tot_real = 0;

    cout << world_rank << endl;
    // Create the vector of image indexes. Its total size will be the number of elements
    // per process times the number of processes
    int *img_index = NULL;
    if (world_rank == 0) {
        img_index = create_indexes(elements_per_proc, world_size);
    }

    // Create a buffer that will hold a subset of indexes
    int *sub_indexes = new int[elements_per_proc];

    cout << "Pre scatter" << endl;
    // Scatter the indexes to all processes
    MPI_Scatter(img_index, elements_per_proc, MPI_INT, sub_indexes,
                elements_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute the number of real images of the subset
    int count_real = AntiSpoofingDetection::compute_real(sub_indexes, elements_per_proc);

    // Gather all partial averages down to the root process
    int *sum_real = NULL;
    if (world_rank == 0)
        sum_real = new int[world_size];
        
    cout << "Pre Gather" << endl;
    MPI_Gather(&count_real, 1, MPI_INT, sum_real, 1, MPI_INT, 0, MPI_COMM_WORLD);

    cout << "Pre compute_sum_real" << endl;
    // Compute the total average of all numbers.
    if (world_rank == 0) 
        tot_real = AntiSpoofingDetection::compute_sum_real(sum_real, world_size);

    cout << "Pre deallocation" << endl;
    // Clean up
    if (world_rank == 0) {
        delete(img_index);
        delete(sum_real);
    }
    delete(sub_indexes);
    
    cout << "After deallocation" << endl;

    // Take the one with higher number of occurences
    if (tot_real > n_img/2)
        return "Real";
    else
        return "Fake";
}
*/