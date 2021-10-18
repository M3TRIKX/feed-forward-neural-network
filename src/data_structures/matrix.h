//
// Created by Dáša Pawlasová on 12.10.2021.
//

#include <iomanip>
#include <iostream>
#include <vector>

#ifndef FEEDFORWARDNEURALNET_MATRIX_H
#define FEEDFORWARDNEURALNET_MATRIX_H

class MatrixSizeException: std::exception {};

/**
 * Class representing a matrix
 */
template<typename ELEMENT_TYPE>
class Matrix {
    std::vector <std::vector<ELEMENT_TYPE>> matrix;
    int numRows;
    int numCols;

    static const int DECIMAL_PLACES_IN_PRINT = 2;
    static const int MATRIX_INITIAL = 1;

public:

    /**
     * Matrix class constructor, initiates the matrix with zeros
     * @param rows - amount of rows in the matrix
     * @param cols - amount of columns in the matrix
     */
    Matrix(int rows, int cols):
        numRows(rows), numCols(cols), matrix(rows, std::vector<ELEMENT_TYPE>(cols, 1)) {}

    /**
     * Prints matrix to standard output
     */
    void printMatrix() {
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                std::cout << std::fixed << std::setprecision(DECIMAL_PLACES_IN_PRINT) << matrix[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    /**
     * Get vectors of the matrix
     * @return Vectors of the matrix
     */
    auto &getMatrix() {
        return matrix;
    }

    /**
     * Gets amount of rows in matrix.
     * @return amount of rows
     */
    int getNumRows() {
        return numRows;
    }

    /**
     * Gets amount of columns in matrix.
     * @return amount of columns
     */
    int getNumCols() {
        return numCols;
    }

    auto operator+ (Matrix<ELEMENT_TYPE> & second){
        if (getNumRows() != second.getNumRows() || getNumCols() != second.getNumCols()){
            throw MatrixSizeException();
        }
        Matrix result = Matrix<ELEMENT_TYPE>(getNumRows(), getNumCols());
        for (int i = 0; i < getNumRows(); i++){
            for (int j = 0; j < getNumCols(); j++) {
                result.matrix[i][j] = matrix[i][j] + second.matrix[i][j];
            }
        }
        return result;
    }

    void operator += (Matrix<ELEMENT_TYPE> other){
        if (getNumRows() != other.getNumRows() || getNumCols() != other.getNumCols()){
            throw MatrixSizeException();
        }
        for (int i = 0; i < getNumRows(); i++){
            for (int j = 0; j < getNumCols(); j++) {
                matrix[i][j] += other.matrix[i][j];
            }
        }
    }

    auto operator- (Matrix<ELEMENT_TYPE> & second){
        if (getNumRows() != second.getNumRows() || getNumCols() != second.getNumCols()){
            throw MatrixSizeException();
        }
        Matrix result = Matrix<ELEMENT_TYPE>(getNumRows(), getNumCols());
        for (int i = 0; i < getNumRows(); i++){
            for (int j = 0; j < getNumCols(); j++) {
                result.matrix[i][j] = matrix[i][j] - second.matrix[i][j];
            }
        }
        return result;
    }

    void operator -= (Matrix<ELEMENT_TYPE> other){
        if (getNumRows() != other.getNumRows() || getNumCols() != other.getNumCols()){
            throw MatrixSizeException();
        }
        for (int i = 0; i < getNumRows(); i++){
            for (int j = 0; j < getNumCols(); j++) {
                matrix[i][j] -= other.matrix[i][j];
            }
        }
    }

    auto operator* (Matrix<ELEMENT_TYPE> & second){
        if (getNumRows() != second.getNumRows() || getNumCols() != second.getNumCols()){
            throw MatrixSizeException();
        }
        Matrix result = Matrix<ELEMENT_TYPE>(getNumRows(), getNumCols());
        for (int i = 0; i < getNumRows(); i++){
            for (int j = 0; j < getNumCols(); j++) {
                result.matrix[i][j] = matrix[i][j] * second.matrix[i][j];
            }
        }
        return result;
    }

    void operator *= (Matrix<ELEMENT_TYPE> other){
        if (getNumRows() != other.getNumRows() || getNumCols() != other.getNumCols()){
            throw MatrixSizeException();
        }
        for (int i = 0; i < getNumRows(); i++){
            for (int j = 0; j < getNumCols(); j++) {
                matrix[i][j] *= other.matrix[i][j];
            }
        }
    }
};

#endif //FEEDFORWARDNEURALNET_MATRIX_H
