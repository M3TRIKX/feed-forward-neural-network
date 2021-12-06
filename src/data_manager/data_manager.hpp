//
// Created by Dominik Klement on 01/11/2021.
//

#ifndef FEEDFORWARDNEURALNET_DATA_MANAGER_H
#define FEEDFORWARDNEURALNET_DATA_MANAGER_H

#include "../data_structures/matrix.hpp"
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <random>
#include <vector>

class TrainingSetNotLargeEnoughException : public std::exception {};
class WrongInputMatricesException : public std::exception {};

struct DataLabelsShuffle_t {
    Matrix<float> data;
    std::vector<unsigned int> vectorLabels;
};

struct TrainValSplit_t {
    using elem_type = float;
    using label_type = unsigned int;

    Matrix<elem_type> trainData;
    std::vector<label_type> trainLabels;
    Matrix<elem_type> validationData;
    std::vector<label_type> validationLabels;
};

class DataManager {
    using elem_type = float;

public:

    /**
     * Splits data into training and validation set.
     * The class distribution is retained.
     *
     * @param data         Data we want to split
     * @param labelsMatrix Labels we want to split
     * @param trainRatio   Percentage of train data
     * @return Split dataset
     */
    static TrainValSplit_t
    trainValidateSplit(Matrix<elem_type> &&data, std::vector<unsigned int> &&labels, float trainRatio = 8.f / 10);

    /**
     * Shuffles the data and the labels randomly (both the same way).
     *
     * @param data   Data we want to shuffle
     * @param labels Labels we want to shuffle (corresponds to the data)
     * @return Shuffled matrices.
     */
    static DataLabelsShuffle_t randomShuffle(Matrix<elem_type> &&data, std::vector<unsigned int> &&labels);

    /**
    * Divides a matrix into batch-sized matrices
     *
    * @param mat - source matrix
    * @param batchSize - batch size
    * @return batch-sized matrices
    */
    static std::vector<Matrix<elem_type>> generateBatches(const Matrix<elem_type> &mat, size_t batchSize);

    /**
     * Generates batch-sized vectors from source vector
     *
     * @param vec - source vector
     * @param batchSize - batch size
     * @return batch-sized vectors
     */
    static std::vector<std::vector<unsigned int>>
    generateVectorBatches(const std::vector<unsigned int> &vec, size_t batchSize);
};


#endif //FEEDFORWARDNEURALNET_DATA_MANAGER_H
