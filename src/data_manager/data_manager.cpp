//
// Created by Dominik Klement on 01/11/2021.
//

#include "data_manager.h"

TrainValSplit_t DataManager::trainValidateSplit(const Matrix<elem_type> &dataMatrix, const std::vector<unsigned int> &labelsVector, size_t numOfTrainSamples) {
    if (dataMatrix.getNumRows() != labelsVector.size()) {
        throw WrongInputMatricesException();
    }

    if (numOfTrainSamples < dataMatrix.getNumRows() / 2) {
        throw TrainingSetNotLargeEnoughException();
    }

    // First batch contains training set, second one validation set.
    // The second one is smaller than the first one because the training set
    // contains more than 1/2 data.
    auto dataBatches = Matrix<elem_type>::generateBatches(dataMatrix, numOfTrainSamples);
    auto labelsBatches = Matrix<unsigned int>::generateVectorBatches(labelsVector, numOfTrainSamples);

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

DataLabelsShuffle_t DataManager::randomShuffle(Matrix<elem_type> &&dataMatrix, std::vector<unsigned int> &&labelsMatrix) {
    if (dataMatrix.getNumRows() != labelsMatrix.size()) {
        throw WrongInputMatricesException();
    }

    std::vector<size_t> indexes(dataMatrix.getNumRows());
    for (size_t i = 0; i < dataMatrix.getNumRows(); ++i) indexes[i] = i;

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indexes.begin(), indexes.end(), g);

    std::vector<std::vector<elem_type>> data(indexes.size(), std::vector<elem_type>(dataMatrix.getNumCols(), 0));
    std::vector<unsigned int> labels(indexes.size());

    size_t i = 0;
    for (auto j : indexes) {
        for (size_t k = 0; k < dataMatrix.getNumCols(); ++k) {
            data[i][k] = dataMatrix.matrix[j*dataMatrix.getNumCols() + k];
        }
        labels[i] = labelsMatrix[j];
        ++i;
    }
    DataLabelsShuffle_t result;
    result.data = Matrix<elem_type>(std::move(data));
    result.vectorLabels = labels;

    return result;
}
