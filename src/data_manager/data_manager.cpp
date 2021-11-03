//
// Created by Dominik Klement on 01/11/2021.
//

#include "data_manager.h"

TrainValSplit_t DataManager::trainValidateSplit(const Matrix<elem_type> &dataMatrix, const Matrix<unsigned int> &labelsMatrix, size_t numOfTrainSamples) {
    if (dataMatrix.getNumRows() != labelsMatrix.getNumRows()) {
        throw WrongInputMatricesException();
    }

    if (numOfTrainSamples < dataMatrix.getNumRows() / 2) {
        throw TrainingSetNotLargeEnoughException();
    }

    // First batch contains training set, second one validation set.
    // The second one is smaller than the first one because the training set
    // contains more than 1/2 data.
    auto dataBatches = Matrix<elem_type>::generateBatches(dataMatrix, numOfTrainSamples);
    auto labelsBatches = Matrix<unsigned int>::generateBatches(labelsMatrix, numOfTrainSamples);

    if (dataBatches.size() != 2 || labelsBatches.size() != 2) {
        throw WrongSplitBatchesException();
    }

    return {
            .trainData=dataBatches[0],
            .trainLabels=labelsBatches[0],
            .validationData=dataBatches[1],
            .validationLabels=labelsBatches[1]
    };
}

DataLabelsShuffle_t DataManager::randomShuffle(Matrix<elem_type> &&dataMatrix, Matrix<unsigned int> &&labelsMatrix) {
    if (dataMatrix.getNumRows() != labelsMatrix.getNumRows()) {
        throw WrongInputMatricesException();
    }

    std::vector<size_t> indexes(dataMatrix.getNumRows());
    for (size_t i = 0; i < dataMatrix.getNumRows(); ++i) indexes[i] = i;

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indexes.begin(), indexes.end(), g);

    std::vector<std::vector<elem_type>> data(indexes.size());
    std::vector<std::vector<unsigned int>> labels(indexes.size());

    size_t i = 0;
    for (auto j : indexes) {
        data[i] = std::move(dataMatrix.matrix[j]);
        labels[i] = std::move(labelsMatrix.matrix[j]);
        ++i;
    }

    return {
            .data=Matrix<elem_type>(std::move(data)),
            .labels=Matrix<unsigned int>(std::move(labels)),
    };
}