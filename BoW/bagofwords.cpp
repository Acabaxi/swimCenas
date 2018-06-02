#include "../includes/customs.h"
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <map>
#include "opencv2/ml.hpp"
#include "opencv2/opencv_modules.hpp"
#include <ctime>
#include <cstdlib>

using namespace std;
using namespace cv;

#define TRAIN_RATIO 1
#define DICTIONARY_BUILD 0

Mat extractDescriptors(Mat imgsrc,Ptr<Feature2D> f2d);
Mat segmentMask(Mat imgsrc, int value, int isDebug);
Mat descriptorsFromKeypointFile(string className, int imageNumber, string keypointType, BOWImgDescriptorExtractor bowDE);


Mat descriptorsFromKeypointFile(string className, int imageNumber, string keypointType, BOWImgDescriptorExtractor bowDE){

    string projectHome = "/home/pedro/stuff/imagens/";
    
    Mat input;
    vector<KeyPoint> keypoints;
    Mat imgGray;
    Mat bowDescriptors;

    char* fileNumber = new char[20];
    sprintf(fileNumber, "%d", imageNumber);
    string filename = projectHome + className + "/" + string(fileNumber) + ".jpg";
    
    input = imread(filename);

    string path = projectHome + "BoW/" + className + "_ymls/" + keypointType + string(fileNumber) + ".yml"; 
    
    FileStorage fs(path, FileStorage::READ);

    if(fs.isOpened()){
        fs[keypointType] >> keypoints;
        fs.release();
    }
    
    if(keypointType.compare("maskKeypoints") == 0){
        input = segmentMask(input,100,0);
    }
    else
        cvtColor(input,input,CV_BGR2GRAY);

   
    bowDE.compute(input, keypoints, bowDescriptors);

    return bowDescriptors;    
}
Mat segmentMask(Mat imgsrc, int value, int isDebug){
    /** Gets mask and return Grayscale and Masked image
     * 
    */   
    Mat imgMask;
    Mat imgGray;

    imgMask = saturationSegment(imgsrc, value);

    cvtColor(imgsrc,imgGray,CV_BGR2GRAY);
    
    Mat bigcross = getStructuringElement(MORPH_RECT,  Size(50,50));
    dilate(imgMask,imgMask,bigcross); 

    bitwise_and(imgGray,imgMask,imgGray);

    if(isDebug){
        showImage("Mask",imgMask);
        showImage("Masked",imgGray);
    }

    return imgGray;
}
Mat extractDescriptors(Mat imgsrc,Ptr<Feature2D> f2d){

    vector<KeyPoint> keypoints;
    Mat descriptors;

    f2d->detect(imgsrc, keypoints);

    if(keypoints.size() > 0 ){
        Mat imgDrawn;   
        f2d->compute(imgsrc,keypoints,descriptors);
        // drawKeypoints(imgsrc,keypoints,imgDrawn,Scalar(0,0,255));
        // showImage("Keypoints",imgDrawn);
    }
    else{
        cout << "No keypoints found! " << endl;
    }

    return descriptors;
}

int main(){

    srand ( unsigned ( time(0) ) );
    Mat input;
    Mat imgMasked;
    
    Ptr<Feature2D> f2d =  xfeatures2d::SIFT::create();
    
    Mat descriptors;
    Mat featuresUnclustered;


    Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);

    // Define parameters
    Ptr<Feature2D> detector = xfeatures2d::SIFT::create();
    vector<KeyPoint> keypoints;
    Mat bowDescriptors;

    BOWImgDescriptorExtractor bowDE(detector, matcher);


    int dictionarySamples = 10;

    vector<int> class1numberList;
    int class1imageNumber = 100;
    int resolutionRoot = 103; //1 - High res, 101 - Low res, 202 - Medium res


    vector<int> class2numberList;
    int class2imageNumber = 100;


    for(int i = resolutionRoot; i < resolutionRoot + class1imageNumber; i++) class1numberList.push_back(i);
        random_shuffle(class1numberList.begin(),class1numberList.end());
    int asd=0;
    for(int i = 1; i < class2imageNumber; i++) class2numberList.push_back(i);
        random_shuffle(class2numberList.begin(),class2numberList.end());


#if DICTIONARY_BUILD == 2 //Keypoints

    vector<vector<KeyPoint> > grayKeypoints;
    vector<vector<KeyPoint> > maskKeypoints;

    vector<KeyPoint> keypointsMask;
    vector<KeyPoint> keypointsGray;

    Mat imgGray;
    Mat imgMask;
    
    // #pragma omp parallel for ordered schedule(dynamic,3)
    for(int i = 1; i < 304; i++){
        cout << "Detecting image: " << i << endl;
        
        char* fileNumber = new char[20];
        sprintf(fileNumber, "%d", i);
        string filename = "/home/pedro/stuff/imagens/ancora/" + string(fileNumber) + ".jpg";

        input = imread(filename);
        // showImage("input",input);

        imgMask = segmentMask(input,100,0);

        cvtColor(input,imgGray,CV_BGR2GRAY);

        detector->detect(imgGray, keypointsGray);
        detector->detect(imgMask, keypointsMask); 
        
        // #pragma omp ordered
        // {
        grayKeypoints.push_back(keypointsGray);
        maskKeypoints.push_back(keypointsMask);
        // }

        string outfile1 = "./ymls/maskKeypoints" + string(fileNumber) + ".yml";
        string outfile2 = "./ymls/grayKeypoints" + string(fileNumber) + ".yml";

        FileStorage fs2(outfile1, FileStorage::WRITE);
        FileStorage fs(outfile2, FileStorage::WRITE);

        if(fs.isOpened()){
            fs << "grayKeypoints" << keypointsGray;
            fs.release();
        }
        if(fs2.isOpened()){
            fs2 << "maskKeypoints" << keypointsMask;
            fs.release();
        }
    }


    // FileStorage fs("./ymls/grayKeypoints.yml", FileStorage::WRITE);
    // FileStorage fs2("./ymls/maskKeypoints.yml", FileStorage::WRITE);

    // if(fs.isOpened()){
    //     fs << "grayKeypoints" << grayKeypoints;
    //     fs.release();
    // }
    // if(fs2.isOpened()){
    //     fs2 << "maskKeypoints" << maskKeypoints;
    //     fs.release();
    // }

#elif DICTIONARY_BUILD == 1
    cout << "Dictionary ancoras" << endl;
    for(vector<int>::iterator it=class1numberList.begin(); it!=class1numberList.end();++it){
        
        cout << ++asd << endl;
        if(asd == dictionarySamples)
            break;

        char* fileNumber = new char[20];
        sprintf(fileNumber, "%d", *it);
        string filename = "/home/pedro/stuff/imagens/ancora/dictionary/" + string(fileNumber) + ".jpg";

        input = imread(filename);
        // showImage("input",input);

        imgMasked = segmentMask(input,100,0);
      
        //Feature extraction
        
        descriptors = extractDescriptors(imgMasked,f2d);
        featuresUnclustered.push_back(descriptors);
    }


    cout << "Dictionary fundo" << endl;
    asd = 0;
    for(vector<int>::iterator it=class2numberList.begin(); it!=class2numberList.end();++it){
        
       
        cout << ++asd << endl;
        if(asd == dictionarySamples)
            break;
        char* fileNumber = new char[20];
        sprintf(fileNumber, "%d", *it);
        string filename = "/home/pedro/stuff/imagens/fundo/dictionary/" + string(fileNumber) + ".jpg";
        cout << filename << endl;
        input = imread(filename);
        // showImage("input",input);

        cvtColor(input,imgMasked,CV_BGR2GRAY);

        //Feature extraction
        
        descriptors = extractDescriptors(imgMasked,f2d);
        featuresUnclustered.push_back(descriptors);
    }




    //BOWKMeansTrainer
    int dictionarySize = 1000;
    TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);
    int retries = 2;
    int flags   = cv::KMEANS_PP_CENTERS;

    cout << "Clustering features" << endl;
    BOWKMeansTrainer bowTrainer(dictionarySize,tc,retries,flags);
    Mat dictionary = bowTrainer.cluster(featuresUnclustered); 


    char* dicNumber = new char[20];
    sprintf(dicNumber, "%d", dictionarySamples);
    string dictionaryFile = "dictionaryLowRes_" + string(dicNumber) + ".yml";
    
    FileStorage fs(dictionaryFile, FileStorage::WRITE);
    if(fs.isOpened()){
        fs << "vocabulary" << dictionary;
        fs.release();
    }
    
    

#else


    Mat dictionary;

    FileStorage fs("dictionaryLowRes_100.yml", cv::FileStorage::READ);
    if(fs.isOpened())
    {
        cout<<"Loading dictionary" <<endl;
        fs["vocabulary"] >> dictionary;
        fs.release();
    }

    bowDE.setVocabulary(dictionary);

    cout << "Training SVM" << endl;
    //Extract descriptors to train SVM
    map<string, Mat > mapTrainingData;
    mapTrainingData.clear();    

    int train1Counter;
    int train2Counter;
    cout << "Ancoras train descriptors" << endl;
    
    
    
    int listSize = class1numberList.size() * TRAIN_RATIO;
    // #pragma omp parallel for ordered schedule(dynamic,3)
    for(train1Counter = 1; train1Counter < listSize;train1Counter++){
        
        Mat imgGray;

        char* fileNumber = new char[20];
        sprintf(fileNumber, "%d", class1numberList[train1Counter]);
        string filename = "/home/pedro/stuff/imagens/ancora/dictionary/" + string(fileNumber) + ".jpg";
        cout << fileNumber << endl;
        input = imread(filename);

        imgGray = segmentMask(input,100,0);
        
        Mat descriptors;
        
        detector->detect(imgGray,keypoints); 
        
        if(keypoints.size() > 0){

            bowDE.compute(imgGray, keypoints, bowDescriptors);

            if(mapTrainingData.count("ancora") == 0){
                mapTrainingData["ancora"].create(0, bowDescriptors.cols, bowDescriptors.type());
            }
            mapTrainingData["ancora"].push_back(bowDescriptors);
        }
        cout << "done" << endl;
        cout << train1Counter <<endl;
        
        // descriptors = descriptorsFromKeypointFile("ancora",class1numberList[train1Counter],"maskKeypoints",bowDE);
        
        // if(mapTrainingData.count("ancora") == 0){
        //     mapTrainingData["ancora"].create(0, bowDescriptors.cols, bowDescriptors.type());
        // }
        // mapTrainingData["ancora"].push_back(bowDescriptors);
        

    }

    cout << "Fundo Train descriptors" << endl;
    for(train2Counter = 1; train2Counter < 50;train2Counter++){
    // for(train2Counte"r = 1; train2Counter < TRAIN_RATIO * class2numberList.size();train2Counter++){
        
        Mat imgGray;

        char* fileNumber = new char[20];
        sprintf(fileNumber, "%d", class2numberList[train2Counter]);
        string filename = "/home/pedro/stuff/imagens/fundo/dictionary/" + string(fileNumber) + ".jpg";
        cout << fileNumber << endl;
        input = imread(filename);

        cvtColor(input,imgGray,CV_BGR2GRAY);

        detector->detect(imgGray,keypoints); 
        if(keypoints.size() > 0){ 
           // cout << "oi" << endl;
            bowDE.compute(imgGray, keypoints, bowDescriptors);

            if(mapTrainingData.count("fundo") == 0){
                mapTrainingData["fundo"].create(0, bowDescriptors.cols, bowDescriptors.type());
            }
            mapTrainingData["fundo"].push_back(bowDescriptors);
        }

    }

    //SVM and prediction
    map<string,Ptr<ml::SVM> > oneToAllSVM;

    
    // Build one-to-All classifier for each of the training data classes
    float RBF_C[] = { 1.4,1.43,1.46,1.49,10,15,20,25,100,150,200,250};
    float RBF_gamma[] = { 0.01, 0.1, 1};
    
        for(int gamma = 0;gamma < 3; gamma++){
        for(int cenas = 0; cenas <= 12; cenas++)
    {   
        cout << "_-------------------------------------" << endl;
        cout << "_-------------------------------------" << endl;
        cout << "_-------------------------------------" << endl;
        cout << "_-------------------------------------" << endl;
        cout << "_-------------------------------------" << endl;
        cout << "C: " << RBF_C[cenas] << "  Gamma: "<< RBF_gamma[gamma] << endl;
        cout << "_-------------------------------------" << endl;
        cout << "_-------------------------------------" << endl;
        cout << "_-------------------------------------" << endl;
        cout << "_-------------------------------------" << endl;
        cout << "_-------------------------------------" << endl;
        cout << "_-------------------------------------" << endl;
        cout << "_-------------------------------------" << endl;

        for( map<string, Mat>::iterator it = mapTrainingData.begin(); it != mapTrainingData.end(); ++it){

        string class_name = (*it).first;
        cout << class_name << " Classifier" << endl;
        Mat samples(0,bowDescriptors.cols,CV_32FC1);
        Mat labels(0,1,CV_32S);

        samples.push_back(mapTrainingData[class_name]);
        Mat class_label = Mat::ones(mapTrainingData[class_name].rows, 1, CV_32S);
        labels.push_back(class_label);

        for (map<string, Mat>::iterator it1 = mapTrainingData.begin(); it1 != mapTrainingData .end(); ++it1) {
            string not_class_ = (*it1).first;
            if(not_class_[0] == class_name[0]) continue;
            samples.push_back(mapTrainingData[not_class_]);
            class_label = Mat::zeros(mapTrainingData[not_class_].rows, 1, CV_32S);
            labels.push_back(class_label);
        }

        Mat samples_32f;
        samples.convertTo(samples_32f, CV_32FC1);

        oneToAllSVM[class_name] = ml::SVM::create();
        oneToAllSVM[class_name]->setKernel(ml::SVM::RBF);
        oneToAllSVM[class_name]->setC(RBF_C[cenas]);
        oneToAllSVM[class_name]->setGamma(RBF_gamma[gamma]);

        Ptr<ml::TrainData> dataclassTrain = ml::TrainData::create(samples_32f, ml::ROW_SAMPLE, labels);
        
        oneToAllSVM[class_name]->train(dataclassTrain);

    }

    cout << "Testing SVMs" << endl;
    Mat groundTruth(0, 1, CV_32FC1);
    Mat evalData(0, dictionary.rows, CV_32FC1);   //descriptors of the vocab
    Mat resultsClass(0, 1, CV_32FC1);
    
    Mat evalResult(0,1,CV_32FC1);

    int ancoraNumber = 19; 
    for(int testNumber = 1;testNumber < ancoraNumber ;testNumber++){    

        Mat imgGray;
        Mat imgMask;
        Mat imgMid;
        Mat outimg;
        Mat imgOut;
      
        char* fileNumber = new char[20];
        sprintf(fileNumber, "%d", testNumber);
        
        string filename = "/home/pedro/stuff/imagens/ancora/test/NP_" + string(fileNumber) + ".jpg";
        cout << filename << endl;
        string maskFilename = "/home/pedro/stuff/imagens/ancora/test_mask/NP_" + string(fileNumber) + ".jpg";

        input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
        
        // cvtColor(input,imgOut,CV_BGR2HSV);

        // vector<cv::Mat> arrHSV;
        // split(imgOut,arrHSV);
        // input = arrHSV[1];
        // showImage("Saturation",input);  

        
        // cout << "Read input" << endl;
       // imgMask = imread(maskFilename,CV_LOAD_IMAGE_GRAYSCALE);

        // cout <<  "Read Mask image" << endl;
        
        // showImage("Mask",input);
        
        input.copyTo(imgMid);
        // showImage("Detect",imgMid);

        detector->detect(imgMid,keypoints);
        // cout << train1Counter << " Keypoints" << endl;
        // Mat outimg;
        // drawKeypoints(input,keypoints,outimg,Scalar(0,0,255));
        // showImage("Keypoints",outimg);
        if(keypoints.size() > 0){

            bowDE.compute(imgMid, keypoints, bowDescriptors);
            // cout << "keypoints size" << keypoints.size() << endl;
            // cout << "Descriptors size" << bowDescriptors.size() << endl;
            
            hash<int> results;

            for(map<string,Ptr<ml::SVM> >::iterator it = oneToAllSVM.begin(); it != oneToAllSVM.end();++it){
                float res = (*it).second->predict(bowDescriptors);
                cout << "Classifier: " << (*it).first << "Prediction: " << res << endl;

    
            }
        }  }
        // descriptors = descriptorsFromKeypointFile("ancora",class1numberList[train1Counter],"grayKeypoints",bowDE);
        // for(map<string,Ptr<ml::SVM> >::iterator it = oneToAllSVM.begin(); it != oneToAllSVM.end();++it){
        //     float res = (*it).second->predict(bowDescriptors);
        //     cout << "Classifier: " << (*it).first << "Prediction: " << res << endl;
        // }

    }}

    for(int i = 0;i < 30;i++){

        Mat imgGray;

        char* fileNumber = new char[20];
        sprintf(fileNumber, "%d", i);
        
        string filename = "/home/pedro/stuff/imagens/fundo/test/" + string(fileNumber) + ".jpg";

        input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
        
        detector->detect(input,keypoints);
        cout << train1Counter << " Keypoints" << endl;
        Mat outimg;
        // drawKeypoints(input,keypoints,outimg,Scalar(0,0,255));
        // showImage("Keypoints",outimg);
        if(keypoints.size() > 0){

            bowDE.compute(input, keypoints, bowDescriptors);
            cout << "keypoints size " << keypoints.size() << endl;
            cout << "Descriptors size " << bowDescriptors.size() << endl;
            
            for(map<string,Ptr<ml::SVM> >::iterator it = oneToAllSVM.begin(); it != oneToAllSVM.end();++it){
                float res = (*it).second->predict(bowDescriptors);
                cout << "Classifier: " << (*it).first << "Prediction: " << res << endl;
            }
        }  
    }
#endif

    return 1;

}