# Face-Anti-Spoofing

To run the code in [main.cpp](https://github.com/FrancescaPietrobon/Face-Anti-Spoofing/blob/main/main.cpp) it is necessary to modify some paths and run the code with:

```
./FaceAntiSpoofing
```
To acquired some frames of the video and then make the final prediction.

```
./FaceAntiSpoofing -p
```
To make the prediction for the image in <img_path> setted in the main.

```
./FaceAntiSpoofing -e
```
To make the prediction realtime for each frame looking at the rectangle considered as face and as ROI to take the prediction.
