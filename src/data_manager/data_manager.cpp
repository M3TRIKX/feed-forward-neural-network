//
// Created by Dominik Klement on 01/11/2021.
//

#include "data_manager.h"
#include <unordered_map>

TrainValSplit_t DataManager::trainValidateSplit(Matrix<elem_type> &&data, std::vector<unsigned int> &&labels,
                                                float trainRatio) {
    if (data.getNumRows() != labels.size()) {
        throw WrongInputMatricesException();
    }

    auto shuffled = randomShuffle(std::move(data), std::move(labels));
    auto numCols = shuffled.data.getNumCols();

    size_t numOfTrainSamples = trainRatio * shuffled.data.getNumRows();
    size_t numOfValSamples = shuffled.vectorLabels.size() - numOfTrainSamples;

    // Create a map that represents a number of samples in each class and draw
    // n per cent of samples from each class randomly.
    // By doing so, we retain the class distribution.
    std::unordered_map<unsigned int, std::vector<size_t>> classDistribution;
    for (size_t i = 0; i < shuffled.vectorLabels.size(); ++i) {
        classDistribution[shuffled.vectorLabels[i]].push_back(i);
    }

    TrainValSplit_t result{
            .trainData=Matrix<float>(numOfTrainSamples, numCols),
            .trainLabels=std::vector<unsigned int>(numOfTrainSamples),
            .validationData=Matrix<float>(numOfValSamples, numCols),
            .validationLabels=std::vector<unsigned int>(numOfValSamples)
    };

    size_t trainStartIndex = 0;
    size_t valStartIndex = 0;
    for (auto &dataClass: classDistribution) {
        size_t numOfTrainClassSamples = dataClass.second.size() * trainRatio;
        size_t numOfValClassSamples = dataClass.second.size() - numOfTrainClassSamples;

        for (size_t i = 0; i < numOfTrainClassSamples; ++i) {
            size_t index = dataClass.second[i];
            result.trainLabels[trainStartIndex + i] = shuffled.vectorLabels[index];
            for (size_t j = 0; j < numCols; ++j) {
                result.trainData.setItem(trainStartIndex + i, j, shuffled.data.getItem(index, j));
            }
        }

        for (size_t i = 0; i < numOfValClassSamples; ++i) {
            size_t index = dataClass.second[numOfTrainClassSamples + i];
            result.validationLabels[valStartIndex + i] = shuffled.vectorLabels[index];
            for (size_t j = 0; j < numCols; ++j) {
                result.validationData.setItem(valStartIndex + i, j, shuffled.data.getItem(index, j));
            }
        }

        trainStartIndex += numOfTrainClassSamples;
        valStartIndex += numOfValClassSamples;
    }

    return result;
}

DataLabelsShuffle_t DataManager::randomShuffle(Matrix<elem_type> &&data, std::vector<unsigned int> &&labels) {
    if (data.getNumRows() != labels.size()) {
        throw WrongInputMatricesException();
    }

    std::vector<size_t> indexes(data.getNumRows());
    for (size_t i = 0; i < data.getNumRows(); ++i) indexes[i] = i;

    std::random_device randomDevice;
    std::mt19937 generator(randomDevice());
    std::shuffle(indexes.begin(), indexes.end(), generator);

    std::vector<std::vector<elem_type>> newData(indexes.size(), std::vector<elem_type>(data.getNumCols(), 0));
    std::vector<unsigned int> newLabels(indexes.size());

    size_t newIndex = 0;
    for (auto index: indexes) {
        for (size_t k = 0; k < data.getNumCols(); ++k) {
            newData[newIndex][k] = data.matrix[index * data.getNumCols() + k];
        }
        newLabels[newIndex] = labels[index];
        ++newIndex;
    }

    return {
            .data=Matrix<elem_type>(std::move(newData)),
            .vectorLabels=newLabels
    };
}

std::vector<Matrix<DataManager::elem_type>>
DataManager::generateBatches(const Matrix<elem_type> &mat, size_t batchSize) {
    size_t alreadyProcessed = 0;
    size_t matRows = mat.numRows;

    std::vector<Matrix<elem_type>> res;

    while (alreadyProcessed < mat.getNumRows()) {
        auto minSize = std::min(matRows - alreadyProcessed, batchSize);
        Matrix<elem_type> currentSplit(minSize, mat.getNumCols());

        for (size_t i = 0; i < minSize; ++i) {
            for (size_t j = 0; j < mat.getNumCols(); ++j) {
                currentSplit.setItem(i, j, mat.getItem(alreadyProcessed + i, j));
            }
        }

        res.emplace_back(std::move(currentSplit));
        alreadyProcessed += minSize;
    }

    return res;
}

std::vector<std::vector<unsigned int>>
DataManager::generateVectorBatches(const std::vector<unsigned int> &vec, size_t batchSize) {
    auto currentIt = vec.begin();
    size_t alreadyProcessed = 0;
    size_t matRows = vec.size();

    std::vector<std::vector<unsigned int>> res;

    while (alreadyProcessed < vec.size()) {
        auto minSize = std::min(matRows - alreadyProcessed, batchSize);
        res.emplace_back(currentIt, currentIt + minSize);
        alreadyProcessed += batchSize;
        std::advance(currentIt, batchSize);
    }

    return res;
}
