cmake_minimum_required(VERSION 2.8)
project( laser_visualizer )
#set(OpenCV_DIR "/home/jeroen/opencv/build")
#find_package( OpenCV REQUIRED) # or just put find_package PATHS and NO_DEFAULT_PATH
find_package( OpenCV REQUIRED PATHS "/home/jeroen/opencv/build" NO_DEFAULT_PATH)
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( laser_vis src/laser_vis.cpp )
target_link_libraries( laser_vis ${OpenCV_LIBS} )
