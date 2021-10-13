//
// Created by Dáša Pawlasová on 12.10.2021.
// Matrix class implementation
//

#include "matrix.h"
#include <iostream>
using namespace std;

Matrix::Matrix(int rows, int cols){
    this->rows = rows;
    this->cols = cols;
    this->matrix = vector<vector<ELEMENT_TYPE>> ( rows , vector<ELEMENT_TYPE> (cols, MATRIX_INITIAL));
}

void Matrix::PrintMatrix(){
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            cout << fixed << setprecision(DECIMAL_PLACES_IN_PRINT) << this->matrix[i][j] << " ";
        }
        cout << endl;
    }
}
