# Face-Anti-Spoofing

This C++ application is used to detect Face Anti-Spoofing based on some models developed in *Python* using the libraries *Tensorflow*, *Keras*, and *OpenCV*.
Once the weights of each implemented model are extracted and the required libraries are installed, the user has to modify some paths in [data.json](https://github.com/FrancescaPietrobon/Face-Anti-Spoofing/blob/main/src/data.json) as follows in order to run the algorithm.
 - *"frames_path"*: path of the folder in which the frames will be saved
 - *"SNN_weights"*: path of the frozen graph of a network used as feature extractor
 - *"ML_weights"*: path of the Machine Learning model trained and saved with *OpenCV* that it's used to compute the prediction
 - *"face_detector"*: path of the *Dlib* network that detects faces
 - *"example_path"*: path of an image that will be used as example

The following C++ libraries have to be downloaded:
- *OpenCV 4.4.0*
- *Dlib*
- *jsoncpp*
- *MPI*

The *GetPot* library is also exploited even though the user is not required to have it installed in his machine, since the useful files are already included in the repository.

Before running the code, all the CMake files must be created through the following lines:
```
mkdir build
cd build
cmake --make ..
make
```

The following options are available:
1. Predicting a single image.
    ```
    ./FaceAntiSpoofing -p <img_path>
    ```
    Where ```<img_path>``` is the path of the image to predict. If the path is not given it takes the image with path *"example_path"* in [data.json](https://github.com/FrancescaPietrobon/Face-Anti-Spoofing/blob/main/src/data.json).
  
2. Predicting real-time each frame looking at the rectangle considered as face and as ROI to take the prediction.
    ```
    ./FaceAntiSpoofing -e
    ```
3. Acquiring some frames of the video and then make the final prediction.
    ```
    ./FaceAntiSpoofing
    ```
4. Acquiring some frames of the video and then make the final prediction in parallel.
    ```
    mpirun -n <num_cores> ./FaceAntiSpoofing
    ```
    Where ```<num_cores>``` is the number of cores you want to use.
   
5. Predicting all the images in a folder and computing the confusion matrix using the entire image as ROI
    ```
    ./FaceAntiSpoofing -cm_all_image
    ```
6. Predicting all the images in a folder and computing the confusion matrix using only a portion of the image as ROI
    ```
    ./FaceAntiSpoofing -cm
    ```
