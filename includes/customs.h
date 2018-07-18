#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <iomanip>
#include <math.h>

using namespace cv;
using namespace std;

void showImage(String windowName,Mat img);
Mat saturationSegment(Mat img, int value);
Mat SIFT2Root(Mat SIFTDesc);


Mat SIFT2Root(Mat SIFTDesc){

    Mat RootSIFTDesc(SIFTDesc.rows,SIFTDesc.cols,CV_32F);
    
    for(int yIdx = 0; yIdx < SIFTDesc.rows; yIdx++){
        float elementSum = 0;
        //RootSIFTDesc.create(SIFTDesc.size(),SIFTDesc.type());
        //cout << RootSIFTDesc.size()  << endl;
        for(int idx = 0; idx < SIFTDesc.size().width; idx++){
            //cout << idx << "     " << (float)SIFTDesc.at<float>(0,idx)<< endl;
            elementSum += (float)SIFTDesc.at<float>(yIdx,idx);
        }
        //cout << (float)elementSum << endl;
        //SIFTDesc.copyTo(RootSIFTDesc);

        for(int idx = 0; idx < SIFTDesc.size().width; idx++){
            
            RootSIFTDesc.at<float>(yIdx,idx) = (float)sqrt((float)SIFTDesc.at<float>(yIdx,idx) / (float)elementSum);
        }
        //cout << RootSIFTDesc << endl;
        

    }
    
    return RootSIFTDesc;
}


void showImage(String windowName,Mat img){

    namedWindow(windowName,WINDOW_NORMAL);
    imshow(windowName,img);
    waitKey(0);
    return;
}

Mat saturationSegment(Mat img, int value){

    Mat imgHSV;
    vector<Mat> imgSplit;
    Mat imgOut;

    cvtColor(img,imgHSV,CV_BGR2HSV);
    split(imgHSV,imgSplit);

    //Lookup Table
    Mat look(1,256,CV_8U);
    uchar* p = look.ptr();
    for( int i = 0; i < 256; ++i)
        if(i < value)
            p[i] = 255;
        else
            p[i] = 0;

    LUT(imgSplit[1],look,imgOut);

    return imgOut;
    // //Row pointer
    // uchar* p;
    // for(int i = 0; i < img.rows; ++i){
    //     p = img.ptr<uchar>(i);
    //     for(int j = 0; j < img.cols; ++j){
    //         if( (p[j] < value))
    //             p[j] = 255;
    //         else
    //             p[j] = 0;
    //     }
    // }    
}
