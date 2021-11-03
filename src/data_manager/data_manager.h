//
// Created by Dominik Klement on 01/11/2021.
//

#ifndef FEEDFORWARDNEURALNET_DATA_MANAGER_H
#define FEEDFORWARDNEURALNET_DATA_MANAGER_H


#include <cstdlib>
#include <vector>
#include <exception>
#include <random>
#include <algorithm>
#include "../data_structures/matrix.h"

class TrainingSetNotLargeEnoughException : public std::exception{};
class WrongSplitBatchesException : public std::exception{};
class WrongInputMatricesException : public std::exception{};

struct DataLabelsShuffle_t {
    Matrix<float> data;
    Matrix<unsigned int> labels;
};

struct TrainValSplit_t {
    using elem_type = float;
    using label_type = unsigned int;

    Matrix<elem_type> trainData;
    Matrix<label_type> trainLabels;
    Matrix<elem_type> validationData;
    Matrix<label_type> validationLabels;
};

class DataManager {
    using elem_type = float;

public:

    /**
     * Splits data into training and validation set.
     * numOfTrainSamples must be greater or equal to the number of data samples.
     *
     * @param dataMatrix        Data we want to split
     * @param labelsMatrix      Labels we want to split
     * @param numOfTrainSamples Number of samples we want to have in the training set
     *                          (the validation one will contain the rest).
     * @return Split dataset
     */
    static TrainValSplit_t trainValidateSplit(const Matrix<elem_type> &dataMatrix, const Matrix<unsigned int> &labelsMatrix, size_t numOfTrainSamples);

    /**
     * Shuffles the data and the labels randomly (both the same way).
     *
     * @param dataMatrix   Data we want to shuffle
     * @param labelsMatrix Labels we want to shuffle (corresponds to the data)
     * @return Shuffled matrices.
     */
    static DataLabelsShuffle_t randomShuffle(Matrix<elem_type> &&dataMatrix, Matrix<unsigned int> &&labelsMatrix);
};


#endif //FEEDFORWARDNEURALNET_DATA_MANAGER_H
