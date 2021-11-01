//
// Created by Dáša Pawlasová on 12.10.2021.
//

#ifndef FEEDFORWARDNEURALNET_MATRIX_H
#define FEEDFORWARDNEURALNET_MATRIX_H

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

class MatrixSizeException: std::exception {};

/**
 * Class representing a matrix
 */
template<typename ELEMENT_TYPE>
class Matrix {
    size_t numRows;
    size_t numCols;
    std::vector<std::vector<ELEMENT_TYPE>> matrix;

    static const int DECIMAL_PLACES_IN_PRINT = 4;

public:

    Matrix() = default;

    /**
     * Matrix class constructor, initiates the matrix with zeros
     * @param rows - amount of rows in the matrix
     * @param cols - amount of columns in the matrix
     */
    Matrix(size_t rows, size_t cols) :
            numRows(rows), numCols(cols), matrix(rows, std::vector<ELEMENT_TYPE>(cols, 0)) {}

    /**
     * Matrix class constructor, initiates the matrix with defaultValue
     * @param rows - amount of rows in the matrix
     * @param cols - amount of columns in the matrix
     * @pram defaultValue - a value the matrix will be initialized with
     */
    Matrix(size_t rows, size_t cols, ELEMENT_TYPE defaultValue) :
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
    static Matrix<ELEMENT_TYPE> generateRandomMatrix(size_t rows, size_t cols, ELEMENT_TYPE min, ELEMENT_TYPE max) {
        Matrix res(rows, cols);

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                res.matrix[i][j] = generateRandomDecimal(min, max);
            }
        }

        return res;
    }

    static auto generateBatches(const Matrix<ELEMENT_TYPE> &mat, size_t batchSize) {
        auto currentIt = mat.matrix.begin();
        size_t alreadyProcessed = 0;
        size_t matRows = mat.numRows;

        std::vector<Matrix<ELEMENT_TYPE>> res;

        while (alreadyProcessed < mat.getNumRows()) {
            auto minSize = std::min(matRows - alreadyProcessed, batchSize);
            res.emplace_back(std::vector<std::vector<ELEMENT_TYPE>>(currentIt, currentIt + minSize));
            alreadyProcessed += batchSize;
            std::advance(currentIt, batchSize);
        }

        return res;
    }

    /**
     * Prints matrix to standard output
     */
    void printMatrix() {
        for (size_t i = 0; i < numRows; i++) {
            for (size_t j = 0; j < numCols; j++) {
                std::cout << std::fixed << std::setprecision(DECIMAL_PLACES_IN_PRINT) << getItem(i,j) << " ";
            }

            std::cout << std::endl;
        }
    }

    const auto &getMatrix() const {
        return matrix;
    }

    const auto &getMatrixRow(size_t row) const {
        return matrix[row];
    }

    const auto getMatrixCol(size_t col) const {
        std::vector<ELEMENT_TYPE> res(numRows, 0);

        for (size_t i = 0; i < numRows; ++i) {
            res[i] = matrix[i][col];
        }

        return res;
    }

    auto getItem(size_t row, size_t col) const {
        return matrix[row][col];
    }

    void setItem(size_t row, size_t col, ELEMENT_TYPE val) {
        matrix[row][col] = val;
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
     * with the whole second matrix m2.
     * @param m2 Matrix we are multiplying *this with
     * @param numRows Number of rows from *this matrix we want to use for multiplication
     * @return multiplied matrices
     */
    Matrix matmul(const Matrix &m2, int numRowsToMultiply = -1) const {
        if (numRowsToMultiply == -1) {
            return slowMatmul(m2, getNumRows());
        }
        return slowMatmul(m2, numRowsToMultiply);
    }

    Matrix transpose() {
        Matrix res(numCols, numRows);

        for (size_t i = 0; i < numRows; ++i) {
            for (size_t j = 0; j < numCols; ++j) {
                res.matrix[j][i] = matrix[i][j];
            }
        }

        return res;
    }

    /**
     * Applies given function to matrix and returns a result matrix.
     * @tparam F - Function type
     * @param f - unary function to apply
     */
    template<typename F>
    void applyFunction(F f) {
        for (size_t i = 0; i < getNumRows(); i++) {
            for (size_t j = 0; j < getNumCols(); j++) {
                matrix[i][j] = f(matrix[i][j]);
            }
        }
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

        for (size_t i = 0; i < getNumRows(); i++) {
            for (size_t j = 0; j < getNumCols(); j++) {
                matrix[i][j] += rhs.getItem(i,j);
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
            for (size_t j = 0; j < getNumCols(); ++j) {
                matrix[i][j] += rhs[j];
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
            for (size_t j = 0; j < getNumCols(); j++) {
                matrix[i][j] -= rhs.getItem(i,j);
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
            for (size_t j = 0; j < getNumCols(); j++) {
                matrix[i][j] *= rhs.getItem(i,j);
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

    friend auto operator*(Matrix<ELEMENT_TYPE> lhs, ELEMENT_TYPE x) {
        for (size_t i = 0; i < lhs.numRows; ++i) {
            for (size_t j = 0; j < lhs.numCols; ++j) {
                lhs.matrix[i][j] *= x;
            }
        }
        return lhs;
    }

    // Equal operators
    friend bool operator==(const Matrix<ELEMENT_TYPE> &lhs, const Matrix<ELEMENT_TYPE> &rhs) {
        return lhs.matrix == rhs.matrix;
    }

    friend bool operator!=(const Matrix<ELEMENT_TYPE> &lhs, const Matrix<ELEMENT_TYPE> &rhs) {
        return !(lhs == rhs);
    }

    friend class DataManager;

private:
    static ELEMENT_TYPE generateRandomDecimal(ELEMENT_TYPE min, ELEMENT_TYPE max) {
        return (max - min) * ((((ELEMENT_TYPE) rand()) / (ELEMENT_TYPE) RAND_MAX)) + min;
    }

    /**
     * Computes matrix multiplication in a dummy way
     * @param rhs - a matrix we are multiplying with
     * @return *this "matmul" rhs
     */
    Matrix slowMatmul(const Matrix &rhs, size_t numRowsToMultiply) const {
        if (numCols != rhs.numRows) {
            throw MatrixSizeException();
        }

        Matrix res(numRowsToMultiply, rhs.numCols, 0);

        for (size_t i = 0; i < numRowsToMultiply; ++i) {
            for (size_t j = 0; j < rhs.numCols; ++j) {
                for (size_t k = 0; k < numCols; ++k) {
                    res.matrix[i][j] += getItem(i, k) * rhs.getItem(k, j);
                }
            }
        }

        return res;
    }
};

#endif //FEEDFORWARDNEURALNET_MATRIX_H
