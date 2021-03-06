cmake_minimum_required(VERSION 3.9)

project(FaceAntiSpoofing VERSION 1.0)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


# control where the static and shared libraries are built so that on windows
# we don't need to tinker with the path to run the executable
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_library(AntiSpoofingDetection src/antispoofing_detection.cpp)
add_library(FaceDetection src/face_detection.cpp)
add_library(Prediction src/final_prediction.cpp)
add_library(Utilities src/utilities.cpp)

find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)
find_package(MPI REQUIRED)
find_package(jsoncpp REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${dlib_INCLUDE_DIRS})
include_directories("/usr/include/jsoncpp")
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicxx)

add_executable(FaceAntiSpoofing main.cpp)

target_link_libraries(FaceAntiSpoofing ${OpenCV_LIBS} )
target_link_libraries(FaceAntiSpoofing ${dlib_LIBS})
target_link_libraries(FaceAntiSpoofing jsoncpp)
target_link_libraries(FaceAntiSpoofing ${MPI_CXX_LIBRARIES})
target_link_libraries(FaceAntiSpoofing Utilities)
target_link_libraries(FaceAntiSpoofing AntiSpoofingDetection)
target_link_libraries(FaceAntiSpoofing FaceDetection)
target_link_libraries(FaceAntiSpoofing Prediction)
target_link_libraries(FaceAntiSpoofing Utilities)

