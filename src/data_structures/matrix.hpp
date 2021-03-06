//
// Created by Dáša Pawlasová on 12.10.2021.
//

#ifndef FEEDFORWARDNEURALNET_MATRIX_H
#define FEEDFORWARDNEURALNET_MATRIX_H

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <omp.h>
#include <cstring>
#include <algorithm>

class MatrixSizeException : std::exception {};

/**
 * Class representing a matrix
 */
template<typename ELEMENT_TYPE>
class Matrix {
    size_t numRows;
    size_t numCols;
    std::vector<ELEMENT_TYPE> matrix;

    static const int DECIMAL_PLACES_IN_PRINT = 4;

public:
    Matrix() : numRows(0), numCols(0) {}

    /**
     * Matrix class constructor, initiates the matrix with zeros
     * @param rows - amount of rows in the matrix
     * @param cols - amount of columns in the matrix
     */
    Matrix(size_t rows, size_t cols) :
            numRows(rows), numCols(cols), matrix(rows * cols, 0) {
    }

    /**
     * Matrix class constructor, initiates the matrix with defaultValue
     * @param rows - amount of rows in the matrix
     * @param cols - amount of columns in the matrix
     * @pram defaultValue - a value the matrix will be initialized with
     */
    Matrix(size_t rows, size_t cols, ELEMENT_TYPE defaultValue) :
            numRows(rows), numCols(cols), matrix(rows * cols, defaultValue) {
    }

    /**
     * Matrix class constructor, initiates the matrix from a vector
     * @param matrix - 2D array we want to create the matrix from
     */
    Matrix(std::vector<std::vector<ELEMENT_TYPE>> &&vecMatrix) {
        if (vecMatrix.empty()) {
            throw std::exception();
        }

        size_t rowSize = vecMatrix[0].size();
        for (auto const &row: vecMatrix) {
            if (row.size() != rowSize) {
                throw std::exception();
            }
        }

        numRows = vecMatrix.size();
        numCols = rowSize;

        matrix.reserve(numRows * numCols);
        for (const auto &row: vecMatrix) {
            std::copy(row.begin(), row.end(), std::back_inserter(matrix));
        }
    }

    /**
     * Generates matrix with random values from uniform distribution
     * @param rows - amount of rows
     * @param cols - amount of columns
     * @param min - lower bound
     * @param max - upper bound
     * @return generated matrix
     */
    static Matrix<ELEMENT_TYPE>
    generateRandomUniformMatrix(size_t rows, size_t cols, ELEMENT_TYPE min, ELEMENT_TYPE max) {
        Matrix res(rows, cols);
        std::random_device randomDevice;
        std::mt19937 generator(randomDevice());
        std::uniform_real_distribution<> distribution(min, max);

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                res.setItem(i, j, distribution(generator));
            }
        }

        return res;
    }

    /**
     * Prints matrix to standard output
     */
    void printMatrix() {
        for (size_t i = 0; i < numRows; ++i) {
            for (size_t j = 0; j < numCols; ++j) {
                std::cout << std::fixed
                          << std::setprecision(DECIMAL_PLACES_IN_PRINT)
                          << getItem(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }

    /**
     * Get column of a matrix
     * @param col - column index
     * @return column of a matrix
     */
    auto getMatrixCol(size_t col) const {
        std::vector<ELEMENT_TYPE> res(numRows);

        for (size_t i = 0; i < numRows; ++i) {
            res[i] = getItem(i, col);
        }

        return res;
    }

    /**
     * Get item of a matrix
     * @param row - row index
     * @param col - column index
     * @return item
     */
    auto getItem(size_t row, size_t col) const {
        return matrix[numCols * row + col];
    }

    /**
     * Set item
     * @param row - row index
     * @param col - column index
     * @param val - value to set
     */
    void setItem(size_t row, size_t col, ELEMENT_TYPE val) {
        matrix[numCols * row + col] = val;
    }

    auto getMaxRowElement(size_t row) {
        auto startIt = matrix.begin() + row * numCols;
        auto endIt = startIt + numCols;
        return *(std::max_element(startIt, endIt));
    }

    /**
     * Gets amount of rows in matrix.
     * @return amount of rows
     */
    size_t getNumRows() const {
        return numRows;
    }

    /**
     * Gets amount of columns in matrix.
     * @return amount of columns
     */
    size_t getNumCols() const {
        return numCols;
    }

    /**
     * Matrix multiplication with additional feature of multiplying a part of the first matrix
     * with the whole second matrix rhs.
     * @param rhs - Matrix we are multiplying *this with
     * @param numRowsToMultiply - Number of rows from *this matrix we want to use for multiplication
     * @return multiplied matrices
     */
    Matrix matmul(const Matrix &rhs, int numRowsToMultiply = -1) const {
        if (numCols != rhs.numRows) {
            throw MatrixSizeException();
        }

        numRowsToMultiply = numRowsToMultiply == -1 ? getNumRows() : numRowsToMultiply;
        Matrix res(numRowsToMultiply, rhs.numCols, 0);

        for (size_t i = 0; i < numRowsToMultiply; ++i) {
            for (size_t k = 0; k < numCols; ++k) {
                float x = getItem(i, k);
#pragma omp simd
                for (size_t j = 0; j < rhs.numCols; ++j) {
                    res.matrix[i * res.numCols + j] += x * rhs.getItem(k, j);
                }
            }
        }

        return res;
    }

    /**
     * Transposes matrix in place
     * @param result matrix to transpose
     */
    void transpose(Matrix &result) {
        for (size_t i = 0; i < numRows; ++i) {
#pragma omp simd
            for (size_t j = 0; j < numCols; ++j) {
                result.setItem(j, i, getItem(i, j));
            }
        }
    }

    /**
     * Transposes a matrix
     * @return transposed matrix
     */
    Matrix transpose() {
        Matrix res(numCols, numRows);
        transpose(res);
        return res;
    }

    void reset() {
        std::fill(matrix.begin(), matrix.end(), 0);
    }

    /**
     * Applies given function to matrix and returns a result matrix.
     * @tparam F - Function type
     * @param f - unary function to apply
     */
    template<typename F>
    void applyFunction(F f) {
        for (size_t i = 0; i < getNumRows(); i++) {
#pragma omp simd
            for (size_t j = 0; j < getNumCols(); j++) {
                setItem(i, j, f(getItem(i, j)));
            }
        }
    }

    /**
     * Addition of matrix to original matrix
     * @param rhs - matrix to add
     */
    auto &operator+=(const Matrix<ELEMENT_TYPE> &rhs) {
        if (getNumRows() != rhs.getNumRows() || getNumCols() != rhs.getNumCols()) {
            throw MatrixSizeException();
        }

        for (size_t i = 0; i < getNumRows(); i++) {
#pragma omp simd
            for (size_t j = 0; j < getNumCols(); j++) {
                matrix[i * numCols + j] += rhs.getItem(i, j);
            }
        }

        return *this;
    }

    /**
     * Addition of vector to each row in the matrix
     * @param rhs - vector to add to each row in the matrix
     * @return this
     */
    auto &operator+=(const std::vector<ELEMENT_TYPE> &rhs) {
        if (getNumCols() != rhs.size()) {
            throw MatrixSizeException();
        }

        for (size_t i = 0; i < getNumRows(); ++i) {
#pragma omp simd
            for (size_t j = 0; j < getNumCols(); ++j) {
                matrix[i * numCols + j] += rhs[j];
            }
        }

        return *this;
    }

    /**
     * Addition of value to each element in the matrix
     * @param x - value to add to each row in the matrix
     * @return this
     */
    auto &operator+=(ELEMENT_TYPE x) {
        for (size_t i = 0; i < getNumRows(); ++i) {
            for (size_t j = 0; j < getNumCols(); ++j) {
                matrix[i * numCols + j] += x;
            }
        }

        return *this;
    }

    /**
     * Subtraction of matrix to original matrix
     * @param rhs - matrix to subtract
     * @return this
     */
    auto &operator-=(const Matrix<ELEMENT_TYPE> &rhs) {
        if (getNumRows() != rhs.getNumRows() || getNumCols() != rhs.getNumCols()) {
            throw MatrixSizeException();
        }

        for (size_t i = 0; i < getNumRows(); i++) {
#pragma omp simd
            for (size_t j = 0; j < getNumCols(); j++) {
                matrix[i * numCols + j] -= rhs.getItem(i, j);
            }
        }

        return *this;
    }

    /**
     * Multiplication of matrix
     * @param rhs - matrix to multiply by
     * @return this
     */
    auto &operator*=(const Matrix<ELEMENT_TYPE> &rhs) {
        if (getNumRows() != rhs.getNumRows() || getNumCols() != rhs.getNumCols()) {
            throw MatrixSizeException();
        }

        for (size_t i = 0; i < getNumRows(); i++) {
#pragma omp simd
            for (size_t j = 0; j < getNumCols(); j++) {
                matrix[i * numCols + j] *= rhs.getItem(i, j);
            }
        }

        return *this;
    }

    /**
     * Multiplication of matrix by value
     * @param x - value to multiply by
     * @return this
     */
    auto &operator*=(ELEMENT_TYPE x) {
        for (size_t i = 0; i < numRows; ++i) {
#pragma omp simd
            for (size_t j = 0; j < numCols; ++j) {
                matrix[i * numCols + j] *= x;
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
     * Adds rhs vector to each row in the matrix lhs
     * @param lhs - matrix which is copied and returned
     * @param rhs - vector we want to add to each row
     * @return result matrix
     */
    friend auto operator+(Matrix<ELEMENT_TYPE> lhs, const std::vector<ELEMENT_TYPE> &rhs) {
        lhs += rhs;
        return lhs;
    }

    /**
     * Adds value to each row in the matrix lhs
     * @param lhs - matrix which is copied and returned
     * @param x - value to add
     * @return result matrix
     */
    friend auto operator+(Matrix<ELEMENT_TYPE> lhs, ELEMENT_TYPE x) {
        lhs += x;
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
      * Multiplication of matrices
      * @param lhs - matrix which is copied and returned
      * @param rhs - matrix 2 which is
      * @return result matrix
      */
    friend auto operator*(Matrix<ELEMENT_TYPE> lhs, const Matrix<ELEMENT_TYPE> &rhs) {
        lhs *= rhs;
        return lhs;
    }

    /**
     * Multiplication of matrix by a value
     * @param lhs - matrix to multiply
     * @param x - value to multiply by
     * @return multiplied lhs matrix
     */
    friend auto operator*(Matrix<ELEMENT_TYPE> lhs, ELEMENT_TYPE x) {
        lhs *= x;
        return lhs;
    }

    /**
     * Divide matrix by a value
     * @param lhs - matrix to divide
     * @param x - value to divide by
     * @return divided lhs matrix
     */
    friend auto operator/(Matrix<ELEMENT_TYPE> lhs, ELEMENT_TYPE x) {
        lhs *= 1 / x;
        return lhs;
    }

    /**
     * Division of matrices
     * @param lhs - matrix which is copied and returned
     * @param rhs - matrix 2
     * @return result matrix
     */
    friend auto operator/(Matrix<ELEMENT_TYPE> lhs, const Matrix<ELEMENT_TYPE> &rhs) {
        for (size_t i = 0; i < lhs.numRows; ++i) {
#pragma omp simd
            for (size_t j = 0; j < lhs.numCols; ++j) {
                lhs.setItem(i, j, lhs.getItem(i, j) / rhs.getItem(i, j));
            }
        }
        return lhs;
    }

    friend class DataManager;
};

#endif //FEEDFORWARDNEURALNET_MATRIX_H
