#include "../includes/customs.h"
#include "../includes/evaluation.h"
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <map>
#include "opencv2/ml.hpp"
#include "opencv2/opencv_modules.hpp"
#include <ctime>
#include <cstdlib>


using namespace std;
using namespace cv;

#define TRAIN_RATIO 0.9     
#define DICTIONARY_BUILD 0

#define NAMEFILE_CONFIG_PARAM "param_config.xml"

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
    
    Mat bigcross = getStructuringElement(MORPH_RECT,  Size(30,30));

    dilate(imgMask,imgMask,bigcross); 

    bitwise_and(imgGray,imgMask,imgGray);

    if(isDebug){
        //showImage("Mask",imgMask);
        //showImage("Masked",imgGray);
        
        return imgMask;
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

struct config_parameters

{
    std::string corePath;
    std::string maskPath;
    std::string dictionaryPath;
    std::string testPath;

    std::string nameFiles;
    std::vector<std::string> filepaths_objs;
    int number_objects;
    int number_samplesPerTrialObject;
    int number_TrialsPerObject;  //! tag "number_TrialsPerObject". [1..number_TrialsPerObject]

};

void load_config(config_parameters & pconfig, char * filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if(fs.isOpened())
    {
        fs["corePath"]                >> pconfig.corePath;
        fs["maskPath"]                >> pconfig.maskPath;
        fs["dictionaryPath"]          >> pconfig.dictionaryPath;
        fs["testPath"]                >> pconfig.testPath;
        fs["nameFiles"]               >> pconfig.nameFiles;
        fs["number_objects"]          >> pconfig.number_objects;
        fs["number_samplesPerTrialObject"]   >> pconfig.number_samplesPerTrialObject;
        fs["number_TrialsPerObject"]  >> pconfig.number_TrialsPerObject;


        cv::FileNode features   = fs["filepaths_objs"];
        cv::FileNodeIterator it = features.begin(), it_end = features.end();
        for( ; it != it_end; ++it )
            pconfig.filepaths_objs.push_back((*it));
        fs.release();
    }
    return;
}

void save_config()
{
    std::string corePath        = "/home/pedro/stuff/imagens";
    std::vector<std::string> filepaths_objs;
    filepaths_objs.push_back("fundo");
    filepaths_objs.push_back("ancora");
    
    int number_objects          = filepaths_objs.size();
    std::string maskPath        = "/mask/";
    std::string testPath        = "/test/";
    std::string dictionaryPath  = "/dictionary/";
    
    int number_samplesPerTrialObject = 70;
    
    std::string nameFiles       = "/";
    int number_TrialsPerObject     = 4;


    cv::FileStorage fs_conf(NAMEFILE_CONFIG_PARAM, cv::FileStorage::WRITE);
    if(fs_conf.isOpened())
    {
        fs_conf << "corePath"                << corePath;
        fs_conf << "maskPath"                << maskPath;
        fs_conf << "testPath"                << testPath;
        fs_conf << "dictionaryPath"          << dictionaryPath;
    
        fs_conf << "nameFiles"               << nameFiles;
        fs_conf << "number_objects"          << number_objects;
        fs_conf << "number_samplesPerTrialObject"   << number_samplesPerTrialObject;
        fs_conf << "filepaths_objs"          << filepaths_objs;
        fs_conf << "number_TrialsPerObject"  << number_TrialsPerObject;
        fs_conf.release();
    }
    return;
}

void print_config(config_parameters pconfig)
{
        std::cout<<"corePath: "                 << pconfig.corePath<<std::endl;
        std::cout<<"maskPath: "                 << pconfig.maskPath<<std::endl;
        std::cout<<"testPath: "                 << pconfig.testPath<<std::endl;
        std::cout<<"dictionaryPath: "           << pconfig.dictionaryPath<<std::endl;
        
        std::cout<<"nameFiles: "                << pconfig.nameFiles<<std::endl;
        std::cout<<"number_objects: "           << pconfig.number_objects<<std::endl;
        std::cout<<"number_TrialsPerObject: "   << pconfig.number_TrialsPerObject<<std::endl;
        std::cout<<"number_samplesPerTrialObject: "    << pconfig.number_samplesPerTrialObject<<std::endl;
        for( int i = 0; i < pconfig.filepaths_objs.size(); i++ )
            std::cout<<" ->"<<pconfig.filepaths_objs[i]<<std::endl;

    return;
}
int main(){

    config_parameters pconfig;
    load_config(pconfig,NAMEFILE_CONFIG_PARAM);
    // save_config();

    print_config(pconfig);

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
    int resolutionRoot = 1; //1 - High res, 101 - Low res, 202 - Medium res


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
    //FileStorage fs("dictionary2Class_1000.yml", cv::FileStorage::READ);
    
    
    if(fs.isOpened())
    {
        cout<<"Loading dictionary" <<endl;
        fs["vocabulary"] >> dictionary;
        fs.release();
    }

    bowDE.setVocabulary(dictionary);

    cout << dictionary.rows << endl;
    cout << "Training ANN" << endl;
    //Extract descriptors to train SVM
    map<string, Mat > mapTrainingData;
    mapTrainingData.clear();    
    
    int obj_idx = 0;

    //#pragma omp parallel for ordered schedule(dynamic,3)
    for(obj_idx = 0; obj_idx < pconfig.number_objects; obj_idx++){
         
        /*Escolher imagens aleatórias para treino*/

        vector<int> numberList;
        for(int i = 1; i < 101; i++) 
            numberList.push_back(i);
        
        random_shuffle(numberList.begin(),numberList.end());

        int listSize;

        cout << "--------------- Training " << pconfig.filepaths_objs[obj_idx] << endl;
        if(pconfig.filepaths_objs[obj_idx].compare("fundo") == 0){
            listSize = 50;
        }
        else
            listSize = pconfig.number_samplesPerTrialObject * TRAIN_RATIO;

        #pragma omp parallel for ordered schedule(dynamic,3)
        for(int counter = 0; counter < listSize; counter++){

            KeyPointsFilter kpFilter;

            Mat inputIMG, inputMask;
            Mat grayIMG;
            vector<KeyPoint> keypointsIMG;
            Mat bowDescriptorsIMG;

            char* fileNumber = new char[20];
            
            sprintf(fileNumber, "%d", numberList[counter]);
            cout << fileNumber << endl;
            string filenameMask = pconfig.corePath + "/" + pconfig.filepaths_objs[obj_idx] 
                        + pconfig.maskPath + string(fileNumber) + ".jpg";

            string filenameIMG = pconfig.corePath + "/" + pconfig.filepaths_objs[obj_idx] 
                        + pconfig.dictionaryPath + string(fileNumber) + ".jpg";

            /*Read IMGs*/            
            inputIMG = imread(filenameIMG);       
            inputMask = imread(filenameMask,CV_LOAD_IMAGE_GRAYSCALE); //Grayscale because masks in copyTo need to have 1 single channel

            
            /*Color conversion in input image*/
            cvtColor(inputIMG, grayIMG, CV_BGR2GRAY);

            /*Apply Mask*/
            Mat maskedGray;
            grayIMG.copyTo(maskedGray, inputMask);

            //showImage("asd", maskedGray);
            
            detector->detect(maskedGray,keypointsIMG); 

            /*filter keypoints*/
            //kpFilter.retainBest(keypointsIMG, 1000);
            //cout << "keypoint number " << keypointsIMG.size() << endl;

            if(keypointsIMG.size() > 0){

                bowDE.compute(maskedGray, keypointsIMG, bowDescriptorsIMG);

                if(mapTrainingData.count(pconfig.filepaths_objs[obj_idx]) == 0){
                    mapTrainingData[pconfig.filepaths_objs[obj_idx]].create(0, bowDescriptorsIMG.cols, bowDescriptorsIMG.type());
                }
                mapTrainingData[pconfig.filepaths_objs[obj_idx]].push_back(bowDescriptorsIMG);

            }
            //showImage("masked",inputIMG);
        }        
        cout << pconfig.filepaths_objs[obj_idx] << " done \n";
    }
    
    /*knn*/
    /*Declaration*/
    Ptr<ml::ANN_MLP> neuralNet(ml::ANN_MLP::create() ); 
    
    /*Parameters*/
    
    vector<int> layers = {dictionary.rows, 1500, 300, 2};
    
    neuralNet->setLayerSizes(layers);
    neuralNet->setTrainMethod(ml::ANN_MLP::RPROP,0.0001);
    neuralNet->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM,1,1);
    neuralNet->setBackpropMomentumScale(0.6);
    neuralNet->setBackpropWeightScale(0.1);
    


    /*Training Data Setup*/

    cout << "Layer sizes " << neuralNet->getLayerSizes() << endl;
    int classNumber = 0;
    Mat samples(0,0,CV_32FC1);
    Mat labels(0,1,CV_32S);

    for(map<string, Mat>::iterator it = mapTrainingData.begin(); it != mapTrainingData.end(); ++it,classNumber++){

        string class_name = (*it).first;
        cout << class_name << " Classifier" << endl;
        

        samples.push_back(mapTrainingData[class_name]);
        cout << "Samples size " << samples.size() << endl;
        
        Mat oneHot = Mat::zeros(mapTrainingData[class_name].rows,pconfig.number_objects,CV_32FC1);

        for(int onePass = 0; onePass < mapTrainingData[class_name].rows; onePass++){
            oneHot.at<float>(onePass,classNumber) = 1.f;
        }
        //Mat class_label = Mat::ones(mapTrainingData[class_name].rows, 1, CV_32S);
        cout << "SUper Hot" << oneHot << endl;
        labels.push_back(oneHot);


        //oneToAllSVM[class_name] = ml::SVM::create();
        //oneToAllSVM[class_name]->setKernel(ml::SVM::RBF);
        //oneToAllSVM[class_name]->setType(ml::SVM::C_SVC);
        //oneToAllSVM[class_name]->setC(CVal);
        //oneToAllSVM[class_name]->setGamma(GammaVal);
        //Ptr<ml::TrainData> dataclassTrain = ml::TrainData::create(samples_32f, ml::ROW_SAMPLE, labels);
        
        //oneToAllSVM[class_name]->train(dataclassTrain);
    }
    //cout << " Labels " << samples << endl;

    Mat samples_32f;
    samples.convertTo(samples_32f, CV_32FC1);

    Ptr<ml::TrainData> dataclassTrain = ml::TrainData::create(samples_32f, ml::ROW_SAMPLE, labels);
    neuralNet->train(dataclassTrain);
    //kNearestNeighbors->setAlgorithmType(ml::KNearest::KDTREE);
    //neuralNet->setIsClassifier(true);

    cout << "Testing SVMs" << endl;
    Mat groundTruth(0, 1, CV_32FC1);
    Mat evalData(0, dictionary.rows, CV_32FC1);   //descriptors of the vocab
    Mat resultsClass(0, 1, CV_32FC1);
    
    Mat evalResult(0,1,CV_32FC1);


    /*Testing SVMs*/

    for(obj_idx = 0; obj_idx < pconfig.number_objects; obj_idx++){
        //if(pconfig.filepaths_objs[obj_idx].compare("fundo") == 0 ){
        //    continue;
        //}

        int testSize = 3;

        //#pragma omp parallel for ordered schedule(dynamic,3)
        for(int counter = 1; counter <= testSize; counter++){

            Mat inputIMG, inputMask;
            Mat grayIMG;
            vector<KeyPoint> keypointsIMG;
            Mat bowDescriptorsIMG;

            char* fileNumber = new char[20];
            
            sprintf(fileNumber, "%d", counter);

            cout << fileNumber << endl;

            string filenameIMG = pconfig.corePath + "/" + pconfig.filepaths_objs[obj_idx] 
                        + pconfig.testPath + string(fileNumber) + ".jpg";

            /*Read IMGs*/            
            inputIMG = imread(filenameIMG);       
            
            /*Color conversion in input image*/
            cvtColor(inputIMG, grayIMG, CV_BGR2GRAY);

            // showImage("Detect",imgMid);

            int width = grayIMG.cols;
            int height = grayIMG.rows;
            
            Mat m1 = Mat::zeros(height,width,CV_8UC1);
            
            map<string, Mat> outputImage;
            for(int objIdx = 0; objIdx < pconfig.number_objects; objIdx++){
                
                outputImage[pconfig.filepaths_objs[objIdx]].create(height, width, CV_8UC1);
                outputImage[pconfig.filepaths_objs[objIdx]] = Mat::zeros(height,width,CV_8UC1);

            }
           
            int xRoot = 0; int xIter = 0;
            int yRoot = 0; int yIter = 0;
            int xCropPixels = width/4;
            int yCropPixels = height/4;

            //#pragma omp parallel for ordered schedule(dynamic,3)
            for(yRoot = 0; yRoot+yCropPixels < height; yRoot += height/8,yIter++ ){          
                for(xRoot = 0; xRoot+xCropPixels < width; xRoot += width/8,xIter++){
                
                    vector<KeyPoint> keypointsTest;
                    //cout << "xRoot = " << xRoot<< "yRoot = " << yRoot << endl;
                    cv::Rect ROI(xRoot,yRoot,xCropPixels,yCropPixels);
                    cv::Mat croppedMat(grayIMG,ROI);
                    //showImage("crop",croppedMat);

                    detector->detect(croppedMat,keypointsTest);
                    
                    //cout << "keypoints" << keypointsTest.size();
                    if(keypointsTest.size() > 0){
                        bowDE.compute(croppedMat, keypointsTest, bowDescriptors);
                        // cout << "keypoints size" << keypoints.size() << endl;
                        // cout << "Descriptors size" << bowDescriptors.size() << endl;
                        
                        Mat neighbors;
                        float res = neuralNet->predict(bowDescriptors);
                        cout <<"Neural pred " << res << endl;
                        
                         // cout << "Classifier: " << (*it).first << "Prediction: " << res << endl;
                        
                        for(int i = 0; i < xCropPixels; i++){
                            for(int j = 0; j <yCropPixels; j++){
                                outputImage[pconfig.filepaths_objs[(int)res]].at<uchar>(yRoot+j,xRoot+i) = croppedMat.at<uchar>(j,i);
                            }                  
                        }  
                                                   
                    }
                }
            }

            //string truthPath = pconfig.corePath + "/" + pconfig.filepaths_objs[obj_idx] 
            //            + "/truth/" + string(fileNumber) + ".jpg";
//
//
            //Mat truthIMG = imread(truthPath);
            //
            //evaluatePredictionSingle(outputImage[pconfig.filepaths_objs[obj_idx]],truthIMG);

            for(int numObjs = 0; numObjs < pconfig.number_objects;numObjs++){
                showImage(pconfig.filepaths_objs[numObjs],outputImage[pconfig.filepaths_objs[numObjs]]);
                //outputImage[(*it).first] = Mat::zeros(height,width,CV_8UC1);
            }
            //showImage("masked",inputIMG);
        }        


        cout << pconfig.filepaths_objs[obj_idx] << " done \n";
        

    }

    
#endif

    return 1;

}