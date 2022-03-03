# Face-Anti-Spoofing

This C++ application is used to detect Face Anti-Spoofing based on some models developed in *Python* using the libraries *Tensorflow*, *Keras*, and *OpenCV*.
Once extracted the weights of the models and installed the required libraries, to run the code it is necessary only to modify some path in [data.json](https://github.com/FrancescaPietrobon/Face-Anti-Spoofing/blob/main/src/data.json), setting:
 - *"frames_path"*: path of the folder in which the frames will be saved
 - *"SNN_weights"*: path of the frozen graph of a network used as feature extractor
 - *"ML_weights"*: path of the Machine Learning model trained and saved with *OpenCV* that it's used to compute the prediction
 - *"face_detector"*: path of the *Dlib* network that detects faces
 - *"example_path"*: path of an image that will be used as example

The libraries that need to be installed to run the code are:
- *OpenCV 4.4.0*
- *Dlib*
- *jsoncpp*
- *MPI*

The *GetPot* library is also used but the needed files are already included in the repository.

Before running the code all the CMake files must be created and this can be done through:
```
mkdir build
cd build
cmake --make ..
make
```

To run the code different options are available:
1. Predict a single image.
    ```
    ./FaceAntiSpoofing -p <img_path>
    ```
    Where ```<img_path>``` is the path of the image to predict. If the path is not given it takes the image with path *"example_path"* in [data.json](https://github.com/FrancescaPietrobon/Face-Anti-Spoofing/blob/main/src/data.json).
  
2. Predict real-time each frame looking at the rectangle considered as face and as ROI to take the prediction.
    ```
    ./FaceAntiSpoofing -e
    ```
3. Acquired some frames of the video and then make the final prediction.
    ```
    ./FaceAntiSpoofing
    ```
4. Acquired some frames of the video and then make the final prediction in parallel.
    ```
    mpirun -n <num_cores> ./FaceAntiSpoofing
    ```
    Where ```<num_cores>``` is the number of cores you want to use.
