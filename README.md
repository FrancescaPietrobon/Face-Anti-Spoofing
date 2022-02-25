# Face-Anti-Spoofing

To run the code in [main.cpp](https://github.com/FrancescaPietrobon/Face-Anti-Spoofing/blob/main/main.cpp) it is necessary to modify some paths and run the code with:

```
./FaceAntiSpoofing -p <img_path>
```
To make the prediction for the image in <img_path>. If the path is not given it take the image <example_path> in [data.json](https://github.com/FrancescaPietrobon/Face-Anti-Spoofing/blob/main/data.json).

```
./FaceAntiSpoofing -e
```
To make the prediction realtime for each frame looking at the rectangle considered as face and as ROI to take the prediction.

```
./FaceAntiSpoofing
```
To acquired some frames of the video and then make the final prediction.

```
mpirun -n <num_cores> ./FaceAntiSpoofing
```
To acquired some frames of the video and then make the final prediction using <num_cores> number of cores in parallel.
