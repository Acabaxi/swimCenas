cmake_minimum_required(VERSION 2.8)
project( seg_opencv )

# packages                                                                                                                                                                                                 
find_package( OpenCV REQUIRED )
include_directories(${OPENCV_INCLUDE_DIRS})


#Include the FindCUDA script
#include(FindCUDA)


add_executable( seg segTest.cpp)
target_link_libraries( seg ${OpenCV_LIBS})

