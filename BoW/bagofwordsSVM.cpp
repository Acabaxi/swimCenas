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
#define DICTIONARY_BUILD 1

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
    cout << "keypoints " << keypoints.size() << endl;
    if(keypoints.size() > 0 ){
        Mat imgDrawn;   
        f2d->compute(imgsrc,keypoints,descriptors);
        cout << descriptors.size() << endl;
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

    cout << "ello"<< endl;

#elif DICTIONARY_BUILD == 1

    //#pragma omp parallel for ordered schedule(dynamic,3)
    for(int obj_idx = 0; obj_idx < pconfig.number_objects; obj_idx++){
        cout << "Dictionary " << pconfig.filepaths_objs[obj_idx] << endl;
        
        for(int i = 1; i < 100; i++){
            
            Mat inputIMG;
            Mat inputMask;
            
            char* fileNumber = new char[20];
            sprintf(fileNumber, "%d", i);
            cout << fileNumber << endl;

            string filenameMask = pconfig.corePath + "/" + pconfig.filepaths_objs[obj_idx] 
                        + pconfig.maskPath + string(fileNumber) + ".jpg";

            string filenameIMG = pconfig.corePath + "/" + pconfig.filepaths_objs[obj_idx] 
                        + pconfig.dictionaryPath + string(fileNumber) + ".jpg";

            /*Read IMGs*/            
            inputIMG = imread(filenameIMG,CV_LOAD_IMAGE_GRAYSCALE);       
            inputMask = imread(filenameMask,CV_LOAD_IMAGE_GRAYSCALE); //Grayscale because masks in copyTo need to have 1 single channel

            /*Color conversion in input image*/
            //cvtColor(inputIMG, grayIMG, CV_BGR2GRAY);

            /*Apply Mask*/
            Mat maskedGray;
            inputIMG.copyTo(maskedGray, inputMask);

            //showImage("asd", maskedGray);
        
            //Feature extraction
            Mat dicDescriptors;
            Mat rootDescriptors;
            dicDescriptors = extractDescriptors(maskedGray,f2d);

            rootDescriptors = SIFT2Root(dicDescriptors);
            featuresUnclustered.push_back(rootDescriptors);
            cout << featuresUnclustered.size() << endl;
        }
    
    }



    //BOWKMeansTrainer
    int dicSize[] = {700,1000};
    for(int dicTrav = 0; dicTrav < 6; dicTrav++){
        int dictionarySize = dicSize[dicTrav];

        TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);
        int retries = 2;
        int flags   = cv::KMEANS_PP_CENTERS;

        cout << "Clustering features " << dictionarySize << endl;
        BOWKMeansTrainer bowTrainer(dictionarySize,tc,retries,flags);

        Mat dictionary = bowTrainer.cluster(featuresUnclustered); 

        char* dicNumber = new char[20];
        sprintf(dicNumber, "%d", dictionarySize);
        string dictionaryFile = "dictionaryRootSIFT_" + string(dicNumber) + ".yml";
        
        FileStorage fs(dictionaryFile, FileStorage::WRITE);
        if(fs.isOpened()){
            fs << "vocabulary" << dictionary;
            fs.release();
        }
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
    else
        cout << "tamal" << endl;

    bowDE.setVocabulary(dictionary);

    cout << "Training SVM" << endl;
    //Extract descriptors to train SVM
    map<string, Mat > mapTrainingData;
    mapTrainingData.clear();    

    int train1Counter;
    int train2Counter;
    cout << "Ancoras train descriptors" << endl;
    
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
    



    //SVM and prediction
    map<string,Ptr<ml::SVM> > oneToAllSVM;

    
    // Build one-to-All classifier for each of the training data classes
    float RBF_C[] = { 1.4,1.43,1.46,1.49,10,15,20,25,100,150,200,250};
    float RBF_gamma[] = { 0.01, 0.1, 1};
    
    float CVal = 1.42;
    float GammaVal = 1;

        cout << "_-------------------------------------" << endl;
        cout << "_-------------------------------------" << endl;
        cout << "_-------------------------------------" << endl;
        cout << "_-------------------------------------" << endl;
        cout << "_-------------------------------------" << endl;
        //cout << "C: " << RBF_C[cenas] << "  Gamma: "<< RBF_gamma[gamma] << endl;
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
        oneToAllSVM[class_name]->setType(ml::SVM::C_SVC);
        oneToAllSVM[class_name]->setC(CVal);
        oneToAllSVM[class_name]->setGamma(GammaVal);

        Ptr<ml::TrainData> dataclassTrain = ml::TrainData::create(samples_32f, ml::ROW_SAMPLE, labels);
        
        oneToAllSVM[class_name]->train(dataclassTrain);

    }

    cout << "Testing SVMs" << endl;
    Mat groundTruth(0, 1, CV_32FC1);
    Mat evalData(0, dictionary.rows, CV_32FC1);   //descriptors of the vocab
    Mat resultsClass(0, 1, CV_32FC1);
    
    Mat evalResult(0,1,CV_32FC1);


    /*Testing SVMs*/

    for(obj_idx = 0; obj_idx < pconfig.number_objects; obj_idx++){
        if(pconfig.filepaths_objs[obj_idx].compare("fundo") == 0 ){
            continue;
        }

        int testSize = 1;

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

                        for(map<string,Ptr<ml::SVM> >::iterator it = oneToAllSVM.begin(); it !=  oneToAllSVM.end();++it){
                            float res = (*it).second->predict(bowDescriptors);
                           // cout << "Classifier: " << (*it).first << "Prediction: " << res << endl;

                            if(res){
                                for(int i = 0; i < xCropPixels; i++){
                                    for(int j = 0; j <yCropPixels; j++){
                                        outputImage[(*it).first].at<uchar>(yRoot+j,xRoot+i) = croppedMat.at<uchar>(j,i);
                                    }                  
                                }  
                            } 
                        }   
                    }
                }
            }

            string truthPath = pconfig.corePath + "/" + pconfig.filepaths_objs[obj_idx] 
                        + "/truth/" + string(fileNumber) + ".jpg";


            Mat truthIMG = imread(truthPath);
            
            evaluatePredictionSingle(outputImage[pconfig.filepaths_objs[obj_idx]],truthIMG);

            for(map<string,Ptr<ml::SVM> >::iterator it = oneToAllSVM.begin(); it !=  oneToAllSVM.end();++it){
                showImage((*it).first,outputImage[(*it).first]);
                outputImage[(*it).first] = Mat::zeros(height,width,CV_8UC1);
            }
            //showImage("masked",inputIMG);
        }        


        cout << pconfig.filepaths_objs[obj_idx] << " done \n";
        

    }
#endif

    return 1;

}