cmake_minimum_required(VERSION 3.1)

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

add_library(MyFunctions STATIC my_functions.cpp)
add_library(FaceDetection STATIC face_detection.cpp)

find_package(dlib REQUIRED)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${dlib_INCLUDE_DIRS})

add_executable(FaceAntiSpoofing main.cpp)# my_functions.cpp my_functions.h)
target_link_libraries(FaceAntiSpoofing ${OpenCV_LIBS} )
target_link_libraries(FaceAntiSpoofing ${dlib_LIBS})
target_link_libraries(FaceAntiSpoofing MyFunctions)
target_link_libraries(FaceAntiSpoofing FaceDetection)
