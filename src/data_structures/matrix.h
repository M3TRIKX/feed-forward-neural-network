//
// Created by Dáša Pawlasová on 12.10.2021.
//

#include <vector>
#include <iomanip>

#ifndef FEEDFORWARDNEURALNET_MATRIX_H
#define FEEDFORWARDNEURALNET_MATRIX_H
#define ELEMENT_TYPE double
#define MATRIX_INITIAL 0
#define DECIMAL_PLACES_IN_PRINT 2

using namespace std;

/**
 * Class representing a matrix
 */
class Matrix {
private:
    vector <vector<ELEMENT_TYPE>> matrix;
    int rows;
    int cols;
public:

    /**
     * Matrix class constructor, initiates the matrix with zeros
     * @param rows - amount of rows in the matrix
     * @param cols - amount of columns in the matrix
     */
    Matrix(int rows, int cols);

    /**
     * Prints matrix to standard output
     */
    void PrintMatrix();

    /**
     * Get vectors of the matrix
     * @return Vectors of the matrix
     */
    vector <vector<ELEMENT_TYPE>> GetMatrix(){
        return matrix;
    }

    /**
     * Gets amount of rows in matrix.
     * @return amount of rows
     */
    int GetRows(){
        return rows;
    }

    /**
     * Gets amount of columns in matrix.
     * @return amount of columns
     */
    int GetCols(){
        return cols;
    }
};

#endif //FEEDFORWARDNEURALNET_MATRIX_H
