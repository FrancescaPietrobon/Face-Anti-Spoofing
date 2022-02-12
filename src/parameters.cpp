#include "../GetPot"
#include "../include/parameters.h"


PathsParameters::PathsParameters(const std::string &filename)
{
  GetPot file(filename.c_str());

  frames_path = file(frames_path.c_str(), "/home/fra/Project/Frames/");
  SNN_weights = file(SNN_weights.c_str(), "/home/fra/PROGETTO_PACS/Face-Anti-Spoofing/models/Frozen_graph_All_final_net_5e-4.pb");
  ML_weights = file(ML_weights.c_str(), "/home/fra/PROGETTO_PACS/Face-Anti-Spoofing/models/All_RF_opencv_final_net_lr5e-4.xml");
  face_detect = file(face_detect.c_str(), "/home/fra/PROGETTO_PACS/Face-Anti-Spoofing/models/shape_predictor_68_face_landmarks.dat");
}
