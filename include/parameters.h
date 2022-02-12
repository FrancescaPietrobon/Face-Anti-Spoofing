#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#include <iostream>
#include <string>

using namespace std;

class PathsParameters
{
    public:
        PathsParameters(const std::string &filename);

        string frames_path;
        string SNN_weights;
        string ML_weights;
        string face_detect;
};

#endif