cmake_minimum_required(VERSION 2.8)
project( bof_opencv )

# packages                                                                                                                                                                                                 
find_package( OpenCV REQUIRED )
include_directories(${OPENCV_INCLUDE_DIRS})

set (CMAKE_CXX_FLAGS -fopenmp)
#Include the FindCUDA script
#include(FindCUDA)


add_executable( SVM bagofwordsSVM.cpp)
target_link_libraries( SVM ${OpenCV_LIBS})

add_executable( KNN bagofwordsKNN.cpp)
target_link_libraries( KNN ${OpenCV_LIBS})

add_executable( ANN bagofwordsANN.cpp)
target_link_libraries( ANN ${OpenCV_LIBS})

add_executable( sal saliencyTest.cpp)
target_link_libraries( sal ${OpenCV_LIBS})

