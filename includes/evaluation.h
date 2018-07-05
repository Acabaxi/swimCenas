#include "opencv2/imgproc.hpp"
#include <iomanip>
#include <iostream>

using namespace cv;
using namespace std;

vector <double> evaluatePredictionSingle(Mat detection, Mat groundTruth);
void printPredictions(vector<double> stats);

vector<double> evaluatePredictionSingle(Mat detection, Mat groundTruth){
    
    if(groundTruth.channels() > 1){
        cvtColor(groundTruth,groundTruth,CV_BGR2GRAY);
    }

    /*Make sure mask doesn't have any stray color*/
    /*Maybe replace thresholding the whole image to a soft threshold in the comparison*/
    threshold( groundTruth, groundTruth, 250, 255,CV_THRESH_BINARY );
    showImage("gnd",groundTruth);

    vector <double> stats;

    int truePositivePx = 0, falsePositivePx = 0;
    int trueNegativePx = 0, falseNegativePx = 0;
    
    int numPixelsPositive = 0;
    int numPixelsNegative = 0;

    /*Check dimensions*/
    if(groundTruth.cols != detection.cols){
        throw "Different number of COLUMNS";
    }
    if(groundTruth.rows != detection.rows){
        throw "Different number of ROWS";
    }

    /*Determine negative and positive pixels*/
    for(int i = 0; i < groundTruth.cols * groundTruth.rows; i++){
        
        if(groundTruth.data[i] > 0){
            cout << (int)groundTruth.data[i] << endl;
            numPixelsPositive++;
            
            if(detection.data[i] > 0)
                truePositivePx++;
            else if(detection.data[i] == 0)
                falseNegativePx++;
        }
        else if(groundTruth.data[i] == 0){
            numPixelsNegative++; 

            if(detection.data[i] > 0)
                falsePositivePx++;
            else if(detection.data[i] == 0)
                trueNegativePx++;
        }

    }

    //cout << "True positive " << truePositivePx << " False Negative " << falseNegativePx << endl;
    /*Calculate stuff*/
    double truePositiveR = (double)truePositivePx / (double)(truePositivePx + falseNegativePx);
    double trueNegativeR = (double)trueNegativePx / (double)(trueNegativePx + falsePositivePx);
    double falsePositiveR = (double)falsePositivePx / (double)(falsePositivePx + trueNegativePx);
    double falseNegativeR = (double)falseNegativePx / (double)(truePositivePx + falseNegativePx);

    double accuracy = (double)(truePositivePx + trueNegativePx) / (double)(truePositivePx + trueNegativePx + falsePositivePx + falseNegativePx);

    
    stats.push_back(accuracy);
    stats.push_back(truePositiveR);
    stats.push_back(trueNegativeR);
    stats.push_back(falsePositiveR);
    stats.push_back(falseNegativeR);

    printPredictions(stats);
    return stats;
}

void printPredictions(vector<double> stats){

    cout.setf(ios::fixed);

    cout << "Accuracy " << std::setprecision(3) << stats[0] * 100<< endl;
    cout << "TPR " << stats[1] * 100 << endl;
    cout << "TNR " << stats[2] * 100 << endl;
    cout << "FPR " << stats[3] * 100 << endl;
    cout << "FNR " << stats[4] * 100 << endl;

}
