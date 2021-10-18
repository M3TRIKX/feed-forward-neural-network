//
// Created by Dáša Pawlasová on 12.10.2021.
//

#include <cstdlib>
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
    unsigned int numRows;
    unsigned int numCols;
    std::vector<std::vector<ELEMENT_TYPE>> matrix;

    static const int DECIMAL_PLACES_IN_PRINT = 2;

public:

    Matrix() = default;

    /**
     * Matrix class constructor, initiates the matrix with zeros
     * @param rows - amount of rows in the matrix
     * @param cols - amount of columns in the matrix
     */
    Matrix(unsigned int rows, unsigned int cols) :
            numRows(rows), numCols(cols), matrix(rows, std::vector<ELEMENT_TYPE>(cols, 0)) {}

    /**
     * Matrix class constructor, initiates the matrix with defaultValue
     * @param rows - amount of rows in the matrix
     * @param cols - amount of columns in the matrix
     * @pram defaultValue - a value the matrix will be initialized with
     */
    Matrix(unsigned int rows, unsigned int cols, ELEMENT_TYPE defaultValue) :
            numRows(rows), numCols(cols), matrix(rows, std::vector<ELEMENT_TYPE>(cols, defaultValue)) {}

    /**
     * Matrix class constructor, initiates the matrix from a vector
     * @param matrix - 2D array we want to create the matrix from
     */
    Matrix(std::vector<std::vector<ELEMENT_TYPE>> &&vecMatrix) {
        if (vecMatrix.size() == 0) {
            throw std::exception();
        }

        size_t rowSize = vecMatrix[0].size();
        for (auto const &row : vecMatrix) {
            if (row.size() != rowSize) {
                throw std::exception();
            }
        }

        matrix = std::move(vecMatrix);
        numRows = matrix.size();
        numCols = static_cast<int>(rowSize);
    }

    /**
     * Generates a randomly initialized matrix
     * @param cols - num of matrix rows
     * @param rows - num of matrix cols
     * @param min - random min bound
     * @param max - random max bound
     * @return randomly initialized matrix
     */
    static Matrix<ELEMENT_TYPE> generateRandomMatrix(unsigned int rows, unsigned int cols, ELEMENT_TYPE min, ELEMENT_TYPE max) {
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

    const auto &getMatrix() const {
        return matrix;
    }

    auto getItem(int row, int col) const {
        return matrix[row][col];
    }

    void setItem(int row, int col) {
        return matrix[row][col];
    }

    /**
     * Gets amount of rows in matrix.
     * @return amount of rows
     */
    unsigned int getNumRows() const {
        return numRows;
    }

    /**
     * Gets amount of columns in matrix.
     * @return amount of columns
     */
    unsigned int getNumCols() const {
        return numCols;
    }

    Matrix matmul(const Matrix &m2) {
        return slowMatmul(m2);
    }

    /**
     * Applies given function to matrix and returns a result matrix.
     * @tparam F - Function type
     * @param f - unary function to apply
     * @return resulting matrix
     */
    template<typename F>
    auto applyFunction(F f) {
        auto result = Matrix<ELEMENT_TYPE>(getNumRows(), getNumCols());

        for (int i = 0; i < getNumRows(); i++) {
            for (int j = 0; j < getNumCols(); j++) {
                result.matrix[i][j] = f(matrix[i][j]);
            }
        }

        return result;
    }

    // Arithmetic operators

    /**
     * Addition of matrix to original matrix
     * @param rhs - matrix to add
     */
    auto &operator+=(const Matrix<ELEMENT_TYPE> &rhs) {
        if (getNumRows() != rhs.getNumRows() || getNumCols() != rhs.getNumCols()) {
            throw MatrixSizeException();
        }

        for (int i = 0; i < getNumRows(); i++) {
            for (int j = 0; j < getNumCols(); j++) {
                matrix[i][j] += rhs.matrix[i][j];
            }
        }

        return *this;
    }

    /**
     * Subtraction of matrix to original matrix
     * @param rhs - matrix to subtract
     */
    auto &operator-=(const Matrix<ELEMENT_TYPE> &rhs) {
        if (getNumRows() != rhs.getNumRows() || getNumCols() != rhs.getNumCols()) {
            throw MatrixSizeException();
        }

        for (int i = 0; i < getNumRows(); i++) {
            for (int j = 0; j < getNumCols(); j++) {
                matrix[i][j] -= rhs.matrix[i][j];
            }
        }

        return *this;
    }

    /**
     * Multiplication of matrix
     * @param rhs - matrix to multiply by
     */
    auto &operator*=(const Matrix<ELEMENT_TYPE> &rhs) {
        if (getNumRows() != rhs.getNumRows() || getNumCols() != rhs.getNumCols()) {
            throw MatrixSizeException();
        }

        for (int i = 0; i < getNumRows(); i++) {
            for (int j = 0; j < getNumCols(); j++) {
                matrix[i][j] *= rhs.matrix[i][j];
            }
        }

        return *this;
    }

    /**
     * Addition of matrices
     * @param lhs - matrix which is copied and returned
     * @param rhs - matrix 2 which is added to lhs
     * @return result matrix
     */
    friend auto operator+(Matrix<ELEMENT_TYPE> lhs, const Matrix<ELEMENT_TYPE> &rhs) {
        lhs += rhs;
        return lhs;
    }

    /**
     * Addition of matrices
     * @param lhs - matrix which is copied and returned
     * @param rhs - matrix 2 which is subtracted to lhs
     * @return result matrix
     */
    friend auto operator-(Matrix<ELEMENT_TYPE> lhs, const Matrix<ELEMENT_TYPE> &rhs) {
        lhs -= rhs;
        return lhs;
    }

    /**
      * Addition of matrices
      * @param lhs - matrix which is copied and returned
      * @param rhs - matrix 2 which is
      * @return result matrix
      */
    friend auto operator*(Matrix<ELEMENT_TYPE> lhs, const Matrix<ELEMENT_TYPE> &rhs) {
        lhs *= rhs;
        return lhs;
    }

    // Equal operators
    friend bool operator==(const Matrix<ELEMENT_TYPE> &lhs, const Matrix<ELEMENT_TYPE> &rhs) {
        return lhs.matrix == rhs.matrix;
    }

    friend bool operator!=(const Matrix<ELEMENT_TYPE> &lhs, const Matrix<ELEMENT_TYPE> &rhs) {
        return !(lhs == rhs);
    }

private:
    static ELEMENT_TYPE generateRandomDecimal(ELEMENT_TYPE min, ELEMENT_TYPE max) {
        return (max - min) * ((((ELEMENT_TYPE) rand()) / (ELEMENT_TYPE) RAND_MAX)) + min;
    }

    /**
     * Computes matrix multiplication in a dummy way
     * @param rhs - a matrix we are multiplying with
     * @return *this "matmul" rhs
     */
    Matrix slowMatmul(const Matrix &rhs) {
        if (numCols != rhs.numRows) {
            throw MatrixSizeException();
        }

        Matrix res(numRows, rhs.numCols, 0);

        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < rhs.numCols; ++j) {
                for (int k = 0; k < numCols; ++k) {
                    res.matrix[i][j] += matrix[i][k] * rhs.matrix[k][j];
                }
            }
        }

        return res;
    }
};

#endif //FEEDFORWARDNEURALNET_MATRIX_H
