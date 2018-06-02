#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <iostream>

#include <list>


using namespace cv;
using namespace std;

void showImage(String windowName,Mat img){

    namedWindow(windowName,WINDOW_NORMAL);
    imshow(windowName,img);
    waitKey(0);
    return;
}

void saturationSegment(Mat img, int value,int valmin){

    //Lookup Table
    Mat look(1,256,CV_8U);
    uchar* p = look.ptr();
    for( int i = 0; i < 256; ++i)
        if((i > valmin) && (i < value))
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
    for(int i = 1; i < 10; i++){

        char *  fich = new char[20];
        sprintf(fich, "%d", i);
        string path = "/home/pedro/stuff/imagens/ancora/test/NP_" + string(fich) + string(".jpg");
        input = imread(path);


        // showImage("eletro",input);seg

        cvtColor(input,imgOut,CV_BGR2HSV);
        split(imgOut,imgSplit);

        // showImage("Hue",imgSplit[0]);
        // showImage("Saturation",imgSplit[1]);
        // showImage("Value",imgSplit[2]);

        Mat imgThresholded;
        threshold(imgSplit[1],imgThresholded,127,255,THRESH_BINARY);
        // showImage("Threshold",imgThresholded);
        // Mat imgSobel;
        // Mat imgLap;

        // int dx = 1, dy = 1;
        // for(dx = 1; dx < 10 ; dx=dx+2){
        //     Sobel(imgSplit[1],imgSobel,-1,1,1,dx);
        //     showImage("Sobel",imgSobel);

        //     Laplacian(imgSplit[1],imgLap,-1,dx);
        //     showImage("Laplace",imgLap);
        // }

        int width = input.cols;
        int height = input.rows;

        Mat m1 = Mat::zeros(height,width,CV_8UC1);
        cout << "Height "<<height << "Width " <<width << endl;
        showImage("welele",m1);
        showImage("asd",input);

        Mat imgBlurMask;
        Mat imgBlur;
            
        Mat masked;
        Mat gray;
        cvtColor(input,gray,CV_BGR2GRAY);

        Mat saturationLayer;
        
        imgSplit[1].copyTo(saturationLayer);

        int xRoot = 0; int xIter = 0;
        int yRoot = 0; int yIter = 0;
        int xCropPixels = width/6;
        int yCropPixels = height/6;

        for(yRoot = 0; yRoot+yCropPixels < height; yRoot += height/5,yIter++ ){
           
            for(xRoot = 0; xRoot+xCropPixels < width; xRoot += width/5,xIter++){
             
                cout << "xRoot = " << xRoot<< "yRoot = " << yRoot << endl;

                cv::Rect ROI(xRoot,yRoot,xCropPixels,yCropPixels);

                cv::Mat croppedMat(saturationLayer,ROI);

                //showImage("crop",croppedMat);

                for(int i = 0; i < xCropPixels; i++){
                    for(int j = 0; j <yCropPixels; j++){
                        m1.at<uchar>(yRoot+j,xRoot+i) = croppedMat.at<uchar>(j,i);
                    }
                
                }
                
            }
        }

            showImage("eletro",m1);

    }
    return 1;   
}
