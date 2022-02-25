#include "../include/antispoofing_detection.h"
#include <mpi/mpi.h>


using namespace cv;
using namespace std;
using namespace dlib;


AntiSpoofingDetection::AntiSpoofingDetection(dnn::Net _snn, Ptr<ml::RTrees> _ml, int _n_img, string _frames_path, int _world_rank, int _world_size):
snn(_snn), ml(_ml), n_img(_n_img), frames_path(_frames_path), world_rank(_world_rank), world_size(_world_size) {};


string AntiSpoofingDetection::single_prediction()
{
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
    // SNN prediction
    Mat blob = dnn::blobFromImage(face, 1, Size(256, 256), Scalar(0,0,0), true, false, CV_32F);
    snn.setInput(blob);
    Mat features = snn.forward();

    // ML Model prediction
    int prediction = ml->predict(features);

    return prediction;
}


string AntiSpoofingDetection::multiple_prediction()
{
    int real = 0;
    string frame;

    for (int i=1; i<n_img; i++)
    {
        //Extract the images saved
        frame = frames_path + "frame" + std::to_string(i) +".jpg";
        face = imread(frame, IMREAD_COLOR);

        // Check if the prediction for the image is 0: Real or 1: Fake
        if (AntiSpoofingDetection::value_prediction() == 0)
            real += 1;
    }

    // Take the one with higher number of occurences
    if (real > n_img/2)
        return "Real";
    else
        return "Fake";
}


int* AntiSpoofingDetection::create_indexes(int elements_per_proc, int world_size)
{
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
    cout << "Number of real images = " + to_string(real) << endl;
    return real;
}


int AntiSpoofingDetection::compute_sum_real(int *sum_real, int world_size)
{
    int tot_real = 0;

    // Sum all the real detected by all the processors
    for (int i=0; i<world_size; i++)
        tot_real = tot_real + sum_real[i];
    
    return tot_real;
}




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