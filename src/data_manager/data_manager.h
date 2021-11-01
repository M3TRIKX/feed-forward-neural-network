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
    using ELEMEN_TYPE = float;

    Matrix<ELEMEN_TYPE> data;
    Matrix<ELEMEN_TYPE> labels;
};

struct TrainValSplit_t {
    using elem_type = float;

    Matrix<elem_type> trainData;
    Matrix<elem_type> trainLabels;
    Matrix<elem_type> validationData;
    Matrix<elem_type> validationLabels;
};

class DataManager {
    using elem_type = float;

public:

    /**
     * Splits data into training and validation set.
     * numOfTrainSamples must be greater or equal to the number of data samples.
     */
    static TrainValSplit_t trainValidateSplit(const Matrix<elem_type> &dataMatrix, const Matrix<elem_type> &labelsMatrix, size_t numOfTrainSamples);

    static DataLabelsShuffle_t randomShuffle(Matrix<elem_type> &&dataMatrix, Matrix<elem_type> &&labelsMatrix);
};


#endif //FEEDFORWARDNEURALNET_DATA_MANAGER_H
