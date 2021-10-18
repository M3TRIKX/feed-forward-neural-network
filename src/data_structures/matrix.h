//
// Created by Dáša Pawlasová on 12.10.2021.
//

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#ifndef FEEDFORWARDNEURALNET_MATRIX_H
#define FEEDFORWARDNEURALNET_MATRIX_H

/**
 * Class representing a matrix
 */
template<typename ELEMENT_TYPE>
class Matrix {
    std::vector <std::vector<ELEMENT_TYPE>> matrix;
    int numRows;
    int numCols;

    static const int DECIMAL_PLACES_IN_PRINT = 2;

public:

    /**
     * Matrix class constructor, initiates the matrix with zeros
     * @param rows - amount of rows in the matrix
     * @param cols - amount of columns in the matrix
     */
    Matrix(int rows, int cols):
        numRows(rows), numCols(cols), matrix(rows, std::vector<ELEMENT_TYPE>(cols, 0)) {}

    Matrix(int rows, int cols, ELEMENT_TYPE defaultValue):
        numRows(rows), numCols(cols), matrix(rows, std::vector<ELEMENT_TYPE>(cols, defaultValue)) {}

    Matrix(std::vector<std::vector<ELEMENT_TYPE>> &&m) {
        if (m.size() == 0) {
            throw std::exception();
        }

        size_t rowSize = m[0].size();
        for (auto const &row : m) {
            if (row.size() != rowSize) {
                throw std::exception();
            }
        }

        matrix = std::move(m);
        numRows = matrix.size();
        numCols = static_cast<int>(rowSize);
    }

    static Matrix<ELEMENT_TYPE> generateRandomMatrix(int cols, int rows, ELEMENT_TYPE min, ELEMENT_TYPE max) {
        Matrix res(cols, rows);

        for (unsigned i = 0; i < rows; ++i) {
            for (unsigned j = 0; j < cols; ++j) {
                res.matrix[i][j] = generateRandomDecimal(min, max);
            }
        }

        return res;
    }

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

    Matrix matmul(const Matrix &m2) {
        return slowMatmul(m2);
    }

    // Equal operators
    friend bool operator==(const Matrix<ELEMENT_TYPE> &m1, const Matrix<ELEMENT_TYPE> &m2) {
        return m1.matrix == m2.matrix;
    }

    friend bool operator!=(const Matrix<ELEMENT_TYPE> &m1, const Matrix<ELEMENT_TYPE> &m2) {
        return !(m1 == m2);
    }

private:
    static ELEMENT_TYPE generateRandomDecimal(ELEMENT_TYPE min, ELEMENT_TYPE max) {
        return  (max - min) * ((((ELEMENT_TYPE) rand()) / (ELEMENT_TYPE) RAND_MAX)) + min;
    }

    Matrix slowMatmul(const Matrix &m2) {
        if (numCols != m2.numRows) {
            throw std::exception();
        }

        Matrix res(numRows, m2.numCols, 0);

        for (unsigned i = 0; i < numRows; ++i) {
            for (unsigned j = 0; j < m2.numCols; ++j) {
                for (unsigned k = 0; k < numCols; ++k) {
                    res.matrix[i][j] += matrix[i][k] * m2.matrix[k][j];
                }
            }
        }

        return res;
    }
};

#endif //FEEDFORWARDNEURALNET_MATRIX_H
