//
//  LM.h
//  linear regression
//
//  Created by Abdoul Oudouss Diakite on 27/08/2022.
//
#include <fstream>
#include <sstream>
#include <cmath>
#ifndef LM_h
#define LM_h

namespace LM {
// Define
typedef std::vector< std::vector<double> > matrix;
typedef std::vector<double> vect;
typedef struct{
    std::vector<std::string> colnames;
    matrix data;
} DataFrame;
typedef struct{
    matrix beta;
    std::vector<std::string> colnames;
    double RSquare;
} model;
// initialize  matrix
matrix Matrix(unsigned long  nrow,unsigned long  ncol, double initial_value = 0){
    matrix new_matrix(nrow, std::vector<double>(ncol,initial_value));
    return new_matrix;
}
// number of row
unsigned long nrow(matrix const& m ){
    return m.size();
}
// number of column
unsigned long ncol(matrix const& m ){
    return m[0].size();
}

// unsigned in
DataFrame readCSV(std::string path, char separator, bool header = false)
{
    DataFrame df;
    std::ifstream  data(path);
    std::string line;

        while(std::getline(data,line))
        {
            std::vector<std::string> colnames;
            std::stringstream lineStream(line);
            std::string cell;
            std::vector<double> rows;
            if (header) {
                
                while(std::getline(lineStream,cell,separator))
                {
                    df.colnames.push_back(cell);
                }
                header =false;
            }
            else{
                while(std::getline(lineStream,cell,separator))
                {
                    rows.push_back(std::stod(cell));
                }

                df.data.push_back(rows);
            }
            }

    if (df.colnames.empty()) {
        df.colnames.resize(ncol(df.data));
        for (int i = 0; i< ncol(df.data); i++) {
            df.colnames[i] = "Column" +std::to_string(i);
        }
    }
    return df;
}



// cout Operator for matrix
std::ostream &operator<<(std::ostream &out, matrix const& m ){
    unsigned long n_row = nrow(m), n_col = ncol(m);
    for (int i = 0; i<n_row; i++) {
        for (int j = 0; j<n_col; j++)
            out<<m[i][j]<<"   ";
        out<<std::endl;
    }
    return out;
}

// cout Operator for Dataframe
std::ostream &operator<<(std::ostream &out, DataFrame const& df ){
    unsigned long n_row = nrow(df.data), n_col = ncol(df.data);
    for (int i = 0; i<df.colnames.size(); i++) {
        std::cout<<df.colnames[i]<<";";
    }
    std::cout<<"\n";
    for (int i = 0; i<n_row; i++) {
        for (int j = 0; j<n_col; j++)
            out<<df.data[i][j]<<";";
        out<<std::endl;
    }
    return out;
}

// cout Operator for model
std::ostream &operator<<(std::ostream &out, model const& lm ){
   out<<"------------------------------------------------------- \n";
   out<<"Coefficients : \n";
    for (unsigned int i = 0; i<lm.colnames.size(); i++) {
        out<<lm.colnames[i]<< " : "<<lm.beta[0][i]<<" | ";
    }
    out<<"\n";
   out<<"------------------------------------------------------- \n";
   out<<"R-Square : ";
   out<<lm.RSquare<<"\n";
   out<<"\n";
   return out;
}



// transpse
matrix transpose(matrix const& m){
    matrix t = Matrix(ncol(m), nrow(m));
    for (int i = 0; i<nrow(m); i++) {
        for (int j = 0; j<ncol(m); j++)
            t[j][i] = m[i][j];
        
    }
  return t;
}

// matrix multiplication
matrix operator*(matrix const&  m1,matrix const& m2){
    matrix result;
    try{
        if(ncol(m1) == nrow(m2)){
            result = Matrix(nrow(m1), ncol(m2),0);
            for(int i =0; i<nrow(m1);i++)
                for(int j=0;j<ncol(m2); j++)
                {
                    for(int k=0; k<nrow(m2); k++)
                    {
                        result[i][j] += m1[i][k]*m2[k][j];
                    }
                }
                    
            return result;
        }
        else throw 505;
        
    }
    catch(int err){
        std::cout<<"The number of columns of the first matrix is ​​not equal to the number of rows of the second matrix";
        exit(-1);
    }
}

// multiplication matrix by a numeric value
matrix operator*(double value,matrix   m){
    for(int i = 0; i< nrow(m); ++i)
        for(int j=0; j<ncol(m); j++)
            m[i][j] *= value;
    return m;
}
matrix operator*(matrix   m,double value){
  
    return value*m;
}

// vector multiplication by numeric
vect operator*(double numeric, vect v){
    for(int i=0; i<v.size(); i++)
        v[i] *= numeric;
    return v;
}
// vector absolute value
vect vabs(vect v){
    for(int i =0; i<v.size();i++)
        v[i] = fabs(v[i]);
    return v;
}
// sum vector
vect operator+(vect v,vect const&v2){
        for(int i =0; i<v.size();i++)
            v[i] += v2[i];
        return v;

}

// inverse matrix using Gauss-Jordan
matrix getCofactor(matrix A, matrix temp, unsigned long p, unsigned long q,unsigned long n)
{
    int i = 0, j = 0;
    // Looping for each element of the matrix
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            //  Copying into temporary matrix only those
            //  element which are not in given row and
            //  column
            if (row != p && col != q) {
                temp[i][j++] = A[row][col];
 
                // Row is filled, so increase row index and
                // reset col index
                if (j == n - 1) {
                    j = 0;
                    i++;
                }
            }
        }
    }
    return temp;
}

double determinant(matrix A, unsigned long n)
{
    double D = 0; // Initialize result
 
    //   if matrix contains single element
    if (n == 1)
        return A[0][0];
 
    matrix temp = Matrix(n, n); // To store cofactors
 
    int sign = 1; // To store sign multiplier
 
    // Iterate for each element of first row
    for (unsigned long f = 0; f < n; f++) {
        // Getting Cofactor of A[0][f]
        temp = getCofactor(A, temp, 0, f, n);
        D += sign * A[0][f] * determinant(temp, n - 1);
 
        // terms are to be added with alternate sign
        sign = -sign;
    }
 
    return D;
}
// Function to get adjoint of A[N][N].
matrix adjoint(matrix A)
{
    unsigned long N = A.size();
    matrix adj = Matrix(N,N);
    if (N == 1) {
        adj[0][0] = 1;
        exit(-1);
    }
 
    // temp is used to store cofactors of A[][]
    int sign = 1;
    matrix temp = Matrix(N,N);
 
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // Get cofactor of A[i][j]
            temp = getCofactor(A, temp, i, j, N);
 
            // sign of adj[j][i] positive if sum of row
            // and column indexes is even.
            sign = ((i + j) % 2 == 0) ? 1 : -1;
 
            // Interchanging rows and columns to get the
            // transpose of the cofactor matrix
            adj[j][i] = (sign) * (determinant(temp, N - 1));
        }
    }
    return adj;
}

// Function to calculate and store inverse, returns false if
// matrix is singular
matrix inverse(matrix A)
{
    unsigned long N = A.size();
    matrix inverse = Matrix(N, N);
    // Find determinant of A[][]
    int det = determinant(A, N);
    if (det == 0) {
        std::cout << "Singular matrix, can't find its inverse";
        exit(-1);
    }
 
    // Find adjoint
    matrix adj = Matrix(N, N);
    adj = adjoint(A);
 
    // Find Inverse using formula "inverse(A) =
    // adj(A)/det(A)"
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            inverse[i][j] = adj[i][j] / float(det);
 
    return inverse;
}


model lm(DataFrame df,unsigned long index_y, vect index_X)
{
    model m_model;
    if(!index_X.empty() && index_y >= 0)
    {
        matrix y, X, tXX, beta, temp,m;
        std::vector<std::string> col_names;
        col_names.push_back("Intercept");
        m = transpose(df.data); // transpose matrix
        y.push_back(m[index_y]); // creat y a matrix 1xn
        X.push_back(std::vector<double>(nrow(df.data),1)); // insert vector 1 in fist row
        for (double v: index_X) {
            col_names.push_back(df.colnames[v]);
            X.push_back(m[v]); // fill X rows
        }
        X = transpose(X); // Get X
        y = transpose(y); // Get vertical y
        tXX = transpose(X)*X; // tX*X
        tXX = inverse(tXX); // invers
        temp = transpose(X)*y; // tX * y
        beta = tXX*temp; // Beta vector
        m_model.beta = transpose(beta);
        matrix y_hat, y_bar = transpose(y);
        y_hat = X * beta;
        double average_y = 0, sum1 = 0 , sum2 = 0;
        for(double v:y_bar[0]){
            average_y += v;
        }
        average_y /= y_bar[0].size(); // mean y
        for (unsigned long i = 0 ; i< y_bar[0].size() ; i++) {
            sum1 += (y[i][0] -y_hat[i][0])*(y[i][0] -y_hat[i][0]);
            sum2 += (y_hat[i][0] -average_y)*(y_hat[i][0] -average_y);
        }
        m_model.RSquare = 1-sum1/sum2;
        m_model.colnames= col_names;
        return m_model;
    }
    exit(-1);
}


}
#endif /* LM_h */
