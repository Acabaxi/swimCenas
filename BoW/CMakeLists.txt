cmake_minimum_required(VERSION 2.8)
project( bof_opencv )

# packages                                                                                                                                                                                                 
find_package( OpenCV REQUIRED )
include_directories(${OPENCV_INCLUDE_DIRS})

set (CMAKE_CXX_FLAGS -fopenmp)
#Include the FindCUDA script
#include(FindCUDA)


add_executable( bow bagofwords.cpp)
target_link_libraries( bow ${OpenCV_LIBS})

