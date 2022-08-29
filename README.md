---
title: "Linear Regression with C++"
author: "ABDOUL OUDOUSS DIAKITE"
date: "2022-08-29"
---

## Introduction

This repository contains a set of functions contained in an LM namespace. The idea is to facilitate the task of linear regression.\
The namespace contains two particular types, the matrix and the dataframe of numeric values only.


```cpp
#include <iostream>
#include <vector>
#include "LM.h"
//using namespace Eigen;
using namespace std;
using namespace LM;
int main(int argc, const char * argv[]) {
    // insert code here...
    DataFrame df;
    df = readCSV(/*path*/"iris.csv",
                 /*separator*/',',
                 /*header*/true);
    vector<double> index_X{1,2,3};
    // fitting linear regression model
    model LR =  lm(/*DataFrame*/ df, /*Index of column y*/0, /*vector index of Xi column*/index_X);
    // Summary model
    cout<<LR;
    return 0;
}
```



## Matrix


```cpp
// declare empty matrix
matrix m;
// Creat 4x4 matrix zero
m = Matrix(4,4);
// Multiply matrix by numeric value
m = 4 * m;
// Dot operation for matrix
m = m * m
// Transpose matrix
m= transpose(m);
// inverse matrix
m = inverse(m);
// deterimnant of matrix
double det = determinant(m, m.size());
// adjugate matrix
matrix adj = adjoint(m);
// Display matrix
cout<<m;
```

## DataFrame


```cpp
// Declare a dataframe
Dataframe df;
// reading csv file 
// readCSV(string path,char sep, bool header)
df = readCSV("iris.csv",',',true);
// Access colnames
df.colnames;
// Access to data
df.data; // is matrix
// display 
cout<<df;
```


