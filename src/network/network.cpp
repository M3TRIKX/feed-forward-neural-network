//
// Created by dominik on 22. 10. 2021.
//

#include <chrono>
#include "network.h"
#include "../statistics/weights_info.h"

auto Network::forwardPass(const Matrix<ELEMENT_TYPE> &data, const std::vector<unsigned int> &labels) {
    activationResults.clear();
    activationDerivResults.clear();
    // Input layer has activation fn equal to identity.
    activationResults.push_back(data);

    auto tmp = data.matmul(weights[0]);
    tmp += biases[0];

    networkConfig.layersConfig[1].activationFunction(tmp);
    activationResults.push_back(tmp);

    if (weights.size() == 1) {
        activationDerivResults.emplace_back();
    }
    else {
        auto tmpCopy = tmp;
        networkConfig.layersConfig[1].activationDerivFunction(tmpCopy);
        activationDerivResults.push_back(tmpCopy);
    }

    for (size_t i = 1; i < weights.size(); ++i) {
        tmp = tmp.matmul(weights[i]);
        tmp += biases[i];

        // i + 1 due to the way we store activation functions.
        networkConfig.layersConfig[i + 1].activationFunction(tmp);
        activationResults.push_back(tmp);

        if (i == weights.size() - 1) {
            activationDerivResults.emplace_back();
        }
        else {
            auto tmpCopy = tmp;
            networkConfig.layersConfig[i + 1].activationDerivFunction(tmpCopy);
            activationDerivResults.push_back(tmpCopy);
        }
    }

    return StatsPrinter::getStats(tmp, labels);
}

void Network::backProp(const std::vector<unsigned int> &labels, float eta) {
    const auto &lastLayerConf = networkConfig.layersConfig[networkConfig.layersConfig.size() - 1];
    if (lastLayerConf.activationFunctionType != ActivationFunction::SoftMax) {
        throw WrongOutputActivationFunction();
    }

    float lambda = 0.02;
    float l2Coeff = 1 - eta * lambda / static_cast<float>(labels.size());

    size_t numLayers = networkConfig.layersConfig.size();
    float batchEta = eta / static_cast<float>(labels.size());

    deltas[numLayers - 2] = CrossentropyFunction::costDelta(activationResults[numLayers - 1], labels);
    auto *lastDelta = &(deltas[numLayers - 2]);

    for (int i = static_cast<int>(numLayers) - 2; i > 0; --i) {
        auto matmuls = lastDelta->matmul(weights[i].transpose());
        deltas[i - 1] = matmuls * activationDerivResults[i-1];
        lastDelta = &(deltas[i - 1]);
    }

    // Update weights
    for (size_t i = 0; i < numLayers - 1; ++i) {
        auto weightDelta = activationResults[i].transpose().matmul(deltas[i]);
        // L2 regularization (weight decay).
        weights[i] = weights[i] * l2Coeff;
        weights[i] -= weightDelta * batchEta;

        // Bias computation
        // ToDo: Make this code faster even tho i have no idea how.
        std::vector<ELEMENT_TYPE> biasDelta(biases[i].size(), 0);
        for (size_t j = 0; j < deltas[i].getNumRows(); ++j) {
            for (size_t k = 0; k < deltas[i].getNumCols(); ++k) {
                biasDelta[k] += deltas[i].getItem(j, k);
            }
        }

        for (size_t j = 0; j < biases[i].size(); ++j) {
            biases[i][j] -= batchEta * biasDelta[j];
        }
    }
}

void Network::fit(const TrainValSplit_t &trainValSplit, float eta, size_t numEpochs, size_t batchSize) {
    if (eta < 0) {
        throw NegativeEtaException();
    }

    auto &train_X = trainValSplit.trainData;
    auto &train_y = trainValSplit.trainLabels;
    auto &validation_X = trainValSplit.validationData;
    auto &validation_y = trainValSplit.validationLabels;

    auto trainBatches_X = Matrix<float>::generateBatches(train_X, batchSize);
    auto trainBatches_y = Matrix<unsigned int>::generateBatches(train_y, batchSize);

    float accSum = 0;
    float ceSum = 0;
    size_t numBatches = trainBatches_X.size();

//    const float etaDecrease = (eta - 0.01f) / static_cast<float>(numEpochs);

    for (size_t i = 0; i < numEpochs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        for (size_t j = 0; j < numBatches; ++j) {
            // ToDo: Optimise
            auto labels = trainBatches_y[j].getMatrixCol(0);
            auto stats = forwardPass(trainBatches_X[j], labels);
            accSum += stats.accuracy;
            ceSum += stats.crossEntropy;

            backProp(labels, eta);
        }

//        for (const auto &singleWeights : weights) {
//            WeightInfo::printWeightStats(singleWeights, true);
//        }

        auto valLabels = validation_y.getMatrixCol(0);
        auto predicted = predict(validation_X);
        auto valStats = StatsPrinter::getStats(predicted, valLabels);

        StatsPrinter::printProgressLine(accSum / static_cast<float>(numBatches),
                                        ceSum / static_cast<float>(numBatches),
                                        valStats.accuracy,
                                        valStats.crossEntropy,
                                        i + 1,
                                        numEpochs);

        accSum = 0;
        ceSum = 0;

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = duration_cast<std::chrono::microseconds>(end - start);

        std::cout << "Time taken by function: "
             << duration.count() << " microseconds" << std::endl;

//        eta -= etaDecrease;
    }
}

Matrix<Network::ELEMENT_TYPE> Network::predict(const Matrix<float> &data) {
    auto tmp = data.matmul(weights[0]);
    tmp += biases[0];

    networkConfig.layersConfig[1].activationFunction(tmp);

    for (size_t i = 1; i < weights.size(); ++i) {
        tmp = tmp.matmul(weights[i]);
        tmp += biases[i];

        // i + 1 due to the way we store activation functions.
        networkConfig.layersConfig[i + 1].activationFunction(tmp);
    }

    return tmp;
}