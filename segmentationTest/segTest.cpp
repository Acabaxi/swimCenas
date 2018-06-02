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
    for(int i = 1; i < 10   ; i++){

        char *  fich = new char[20];
        sprintf(fich, "%d", i);
        string path = "/home/pedro/stuff/imagens/ancora/test/NP_" + string(fich) + string(".jpg");
        input = imread(path);


        //  showImage("Input Img",input);

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

        Mat imgBlurMask;
        Mat imgBlur;
            
        Mat masked;
        Mat gray;
        cvtColor(input,gray,CV_BGR2GRAY);

        Mat saturationLayer;
        imgSplit[1].copyTo(saturationLayer);

        
        for(int zise = 7; zise < 9; zise+=2){
            // cout << "Blur" << zise << endl;
            GaussianBlur(saturationLayer,imgBlurMask,Size(21,21),zise,zise);
            // showImage("Blur",imgBlurMask);

            imgBlurMask.copyTo(imgBlur);

            // threshold(imgBlur,imgBlur,127,255,THRESH_BINARY);
            saturationSegment(imgBlur,130,90);
            // threshold(imgBlur,imgBlur,0,255,THRESH_OTSU);
            // bitwise_not(imgBlur,imgBlur);
            // showImage("Mask",imgBlur);
            // showImage("Mask2",imgBlurMask);
        }
        
       
        
        
     
        // Mat cross = getStructuringElement(MORPH_RECT, Size(5,5));
        // dilate(imgBlur,imgBlur,cross);
        cv::Ptr<cv::Feature2D> f2d              = cv::xfeatures2d::SIFT::create();
        std::vector<cv::KeyPoint> keypoints; 

        for(int zise = 3; zise < 7 ; zise++){
            // cout  << "Morph open size: "<<zise << endl;
            Mat imgMid;
            Mat imgErod;
            Mat cross = getStructuringElement(MORPH_CROSS, Size(zise,zise));
  
            Mat bigcross = getStructuringElement(MORPH_RECT,  Size(zise+2,zise+2));
            
            // morphologyEx(imgBlur,imgBlur,MORPH_CLOSE,bigcross);
            morphologyEx(imgBlur,imgBlur,MORPH_OPEN,bigcross);
            // dilate(imgBlur,imgBlur,bigcross);
            // imgMid.copyTo(imgMask);
           

    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);
            
            bitwise_and(gray,imgBlur,imgMid);
            // showImage("ImgBlur",imgBlur);
            // showImage("maksks",imgMid);
            imwrite("segmented.png",imgMid,compression_params);
            // Mat drawer;
            // f2d->detect(imgMid,keypoints,noArray());
            // drawKeypoints(imgMid,keypoints,drawer,Scalar(0,0,255));
            // showImage("keypoints",drawer );
        }     


        
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;

        Mat dst = Mat::zeros(imgBlur.rows,imgBlur.cols,CV_8UC1);
        findContours(imgBlur,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
        
        // int idx = 0;
        // for(;idx >= 0; idx = hierarchy[idx][0]){
        //     drawContours(dst,contours,idx,Scalar(255,0,255),5,8);
        //     showImage("Contours",dst);
        // }

        vector<int> bigContours;
    
        vector<vector <Point> > approxArray; 
        for(int countour = 0; countour < contours.size(); countour++){
            
            std::vector<cv::Point> approx;
            cv::approxPolyDP(cv::Mat(contours[countour]), approx, cv::arcLength(cv::Mat(contours[countour]), true)*0.02, true);

            int area = cv::contourArea(contours[countour]);
            // cout << area << endl;
            if(area  > 10000){
                bigContours.push_back(countour);
                // cout << "pushing Contour " << countour << " AREA " << area << endl;
            }
            
            
        }
        // cout << bigContours.size();
        Mat maskContour = Mat::zeros(imgBlur.rows,imgBlur.cols,CV_8UC1); 
        for(int contourIter = 0; contourIter < bigContours.size();contourIter++){
            // cout << "bigcontour " <<bigContours[contourIter] << endl;
            drawContours(maskContour,contours,bigContours[contourIter],Scalar(255,0,255),-1,8);
            // showImage("maskContour",maskContour);
        }

        Mat bigcross = getStructuringElement(MORPH_RECT,  Size(50,50));
        Mat matCopy;

        dilate(maskContour,maskContour,bigcross);
        input.copyTo(matCopy,maskContour);
        // showImage("dilated",matCopy);

        // showImage("maskContour",maskContour);
        // Mat imgMask;
        // imgSplit[1].copyTo(imgMask);
        // saturationSegment(imgMask,130,90);
        // showImage("Mask",imgMask);


        // Mat decolored = Mat(input.size(),CV_8UC1);
        // Mat dummy = Mat(input.size(),CV_8UC3);
        // decolor(input,decolored,dummy);
        // showImage("decolored",decolored);


        // Mat satLayer, data;
        // imgSplit[1].convertTo(satLayer,CV_32F);
        // data = satLayer.reshape(1,1);
        // // satLayer.convertTo(satLayer,CV_32FC1);
        // int k = 8;
        // Mat labels;
        // Mat centers;
        // TermCriteria tc(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001);
        // std::cout << "Image type and size " << satLayer.size() << "    " << data.size() << endl;

        // kmeans(data,k,labels,tc,5,KMEANS_PP_CENTERS,centers);
        // cout << "done kmenas label size" << labels.size() << " centers size " << centers.size() <<  endl;

        // Mat newImg(imgSplit[1].size(),imgSplit[1].type());

        
        // for( int y = 0; y < satLayer.rows; y++ ){
        //     for( int x = 0; x < satLayer.cols; x++ )
        //     { 
        //     // cout << " y x " << y << "   " << x << endl;
        //     int cluster_idx = labels.at<int>(y + x*satLayer.rows);
            
        //     // cout << "Cetner value " << centers.at<float>(cluster_idx) << " Label " << cluster_idx <<endl;
        //     newImg.at<uchar>(y,x) = centers.at<float>(cluster_idx, 0);
        //     // cout << "new iamge value " << (newImg.at<int>(y,x)) << endl;
        //     }
        // }

        // showImage("Porterized",newImg);


        vector<int> compression_params;
        compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
        compression_params.push_back(100);

        char *  outFile = new char[20];
        sprintf(outFile, "%d", i);
        string pathOut = "/home/pedro/stuff/imagens/ancora/test_mask/NP_" + string(fich) + string(".jpg");
        imwrite(pathOut,maskContour,compression_params);
    }

    
    return 1;

    
}
