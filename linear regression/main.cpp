//
//  main.cpp
//  linear regression
//
//  Created by Abdoul Oudouss Diakite on 27/08/2022.
//
//#include <Eigen/Dense>

#include <iostream>
#include <vector>
#include "LM.h"
//using namespace Eigen;
using namespace std;
using namespace LM;
int main(int argc, const char * argv[]) {
    // insert code here...
    DataFrame df;
    df = readCSV(/*path*/"/Users/abdouloudoussdiakite/Desktop/Teranga Advisory/Application/iris.csv",
                 /*separator*/',',
                 /*header*/true);
    vector<double> index_X{1,2,3};
    // fitting linear regression model
    model LR =  lm(/*DataFrame*/ df, /*Index of column y*/0, /*vector index of Xi column*/index_X);
    // Summary model
    cout<<LR;
    return 0;
}
