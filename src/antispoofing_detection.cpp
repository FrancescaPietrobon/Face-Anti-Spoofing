#include "../include/antispoofing_detection.h"
#include <mpi/mpi.h>


using namespace cv;
using namespace std;
using namespace dlib;


AntiSpoofingDetection::AntiSpoofingDetection(dnn::Net _snn, Ptr<ml::RTrees> _ml, int _n_img, string _frames_path, int _world_rank, int _world_size):
snn(_snn), ml(_ml), n_img(_n_img), frames_path(_frames_path), world_rank(_world_rank), world_size(_world_size) {};


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


string AntiSpoofingDetection::base_multiple_prediction()
{
    int real = 0;
    int fake = 0;

    string frame;

    for (int i=1; i<n_img; i++)
    {
        //Extract the images saved
        frame = frames_path + "frame" + std::to_string(i) +".jpg";
        Mat face = imread(frame, IMREAD_COLOR);

        // Make the prediction for every image
        float prediction = AntiSpoofingDetection::value_prediction(); //TO PARALLELIZE
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


string AntiSpoofingDetection::multiple_prediction() // to remove after cleaning parallelization
{
    int elements_per_proc = int(n_img / world_size);

    int tot_real = 0;

    cout << world_rank << endl;
    // Create the vector of image indexes. Its total size will be the number of elements
    // per process times the number of processes
    int *img_index = NULL;
    if (world_rank == 0) {
        img_index = create_indexes(elements_per_proc * world_size);
    }

    // Create a buffer that will hold a subset of indexes
    int *sub_indexes = (int *)malloc(sizeof(int) * elements_per_proc);

    cout << "Pre scatter" << endl;
    // Scatter the indexes to all processes
    MPI_Scatter(img_index, elements_per_proc, MPI_INT, sub_indexes,
                elements_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute the number of real images of the subset
    int count_real = AntiSpoofingDetection::compute_real(sub_indexes, elements_per_proc);

    // Gather all partial averages down to the root process
    int *sum_real = NULL;
    if (world_rank == 0)
        sum_real = (int *)malloc(sizeof(int) * world_size);
        
    cout << "Pre Gather" << endl;
    MPI_Gather(&count_real, 1, MPI_INT, sum_real, 1, MPI_INT, 0, MPI_COMM_WORLD);

    cout << "Pre compute_sum_real" << endl;
    // Compute the total average of all numbers.
    if (world_rank == 0) 
        tot_real = AntiSpoofingDetection::compute_sum_real(sum_real, world_size);

    cout << "Pre deallocation" << endl;
    // Clean up
    if (world_rank == 0) {
        free(img_index);
        free(sum_real);
    }
    free(sub_indexes);
    
    cout << "After deallocation" << endl;

    // Take the one with higher number of occurences
    if (tot_real > n_img/2)
        return "Real";
    else
        return "Fake";
}


int* AntiSpoofingDetection::create_indexes(int num_elements)
{
    int *img_index = (int *)malloc(sizeof(int) * num_elements);
    
    for (int i = 0; i < num_elements; i++) {
        img_index[i] = i+1;
    }

    return img_index;
}



int AntiSpoofingDetection::compute_real(int *sub_indexes, int elements_per_proc)
{
    int real = 0;
    string frame;
    Mat face;
    Mat blob;
    Mat features;
    float prediction;

    for (int i=0; i<elements_per_proc; i++)
    {
        //Extract the images saved
        frame = frames_path + "frame" + std::to_string(sub_indexes[i]) +".jpg";
        face = imread(frame, IMREAD_COLOR);

        // Make the prediction for every image
        //float prediction = AntiSpoofingDetection::value_prediction();

        // SNN prediction
        blob = dnn::blobFromImage(face, 1, Size(256, 256), Scalar(0,0,0), true, false, CV_32F);
        snn.setInput(blob);
        features = snn.forward();

        // ML Model prediction
        prediction = ml->predict(features);

        if (prediction == 0)
            real += 1;
    }
    cout << "Number of real images = " + to_string(real) << endl;
    return real;
}

int AntiSpoofingDetection::compute_sum_real(int *sum_real, int world_size)
{
    int tot_real = 0;

    for (int i=0; i<world_size; i++)
        tot_real = tot_real + sum_real[i];
    
    return tot_real;
}