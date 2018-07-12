#include "../includes/customs.h"
#include "../includes/evaluation.h"
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <map>
#include "opencv2/ml.hpp"
#include "opencv2/opencv_modules.hpp"
#include <ctime>
#include <cstdlib>
#include <opencv2/saliency.hpp>

using namespace std;
using namespace cv;
using namespace saliency;

int main(){

    Mat inputIMG = imread("/home/pedro/stuff/imagens/ancora/test/1.jpg");
    Mat salMap;

    showImage("1",inputIMG);
    vector<Mat> chan;
    split(inputIMG,chan);

    equalizeHist(chan[0],chan[0]);
    equalizeHist(chan[1],chan[2]);
    equalizeHist(chan[2],chan[2]);
    
    Mat asd;
    merge(chan,asd);
    showImage("asd",asd);


    cvtColor(asd,asd,CV_BGR2GRAY);
    cvtColor(inputIMG,inputIMG,CV_BGR2GRAY);

    //equalizeHist(inputIMG,inputIMG);
    showImage("asdasd",asd);
    showImage("inputIMG",inputIMG);
    Ptr<Saliency> salie;
    
    salie = StaticSaliencySpectralResidual::create();

    if(salie->computeSaliency(inputIMG,salMap))
        cout << "elo" << endl;

    StaticSaliencySpectralResidual ele;
    ele.computeBinaryMap(salMap,salMap);
    //cvtColor(inputIMG,inputIMG,CV_BGR2GRAY);
    addWeighted(asd,0.5,salMap,0.5,0,salMap);
    cout << salMap.size() << "            " << inputIMG.size() << endl;
    showImage("salMap",salMap);
    //banana->computeSaliency(inputIMG,salMap);
    

    return 1;

}