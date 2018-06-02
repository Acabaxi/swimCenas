#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;

void showImage(String windowName,Mat img){

    namedWindow(windowName,WINDOW_NORMAL);
    imshow(windowName,img);
    waitKey(0);
    return;
}

void saturationSegment(Mat img, int value){

    //Lookup Table
    Mat look(1,256,CV_8U);
    uchar* p = look.ptr();
    for( int i = 0; i < 256; ++i)
        if(i < value)
            p[i] = 255;
        else
            p[i] = 0;

    LUT(img,look,img);

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

/**
 * @function main
 */
int main(int argn, char **argv)
{
    Mat input;
    Mat imgOut;
    
    vector<Mat> imgSplit;
    for(int i = 3; i < 7; i++){

        char *  fich = new char[20];
        sprintf(fich, "%d", i);
        string path = "/home/pedro/stuff/imagens/ancora/NP_" + string(fich) + string(".jpg");
        input = imread(path);


        showImage("Input Img",input);

        cvtColor(input,imgOut,CV_BGR2HSV);
        split(imgOut,imgSplit);

        showImage("Hue",imgSplit[0]);
        showImage("Saturation",imgSplit[1]);
        showImage("Value",imgSplit[2]);

        Mat imgMask;
        imgSplit[1].copyTo(imgMask);

        saturationSegment(imgMask,60);
        
        showImage("Mask",imgMask);
    
    
    }
    return 1;

    
}
