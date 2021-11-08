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

void Network::backProp(const std::vector<unsigned int> &labels) {
    deltaBiases = Matrix<ELEMENT_TYPE>(weights.size(), weights[0].getNumCols(), (float) 0);
    const auto &lastLayerConf = networkConfig.layersConfig[networkConfig.layersConfig.size() - 1];
    if (lastLayerConf.activationFunctionType != ActivationFunction::SoftMax) {
        throw WrongOutputActivationFunction();
    }

//    float lambda = 0.02;
//    float l2Coeff = 1 - eta * lambda / static_cast<float>(labels.size());

    size_t numLayers = networkConfig.layersConfig.size();

    deltaWeights[numLayers - 2] = CrossentropyFunction::costDelta(activationResults[numLayers - 1], labels);
    auto *lastDelta = &(deltaWeights[numLayers - 2]);

    for (int i = static_cast<int>(numLayers) - 2; i > 0; --i) {
        auto matmuls = lastDelta->matmul(weights[i].transpose());
        deltaWeights[i - 1] = matmuls * activationDerivResults[i - 1];
        lastDelta = &(deltaWeights[i - 1]);
    }

    for (size_t i = 0; i < numLayers - 1; ++i) {
        for (size_t j = 0; j < deltaWeights[i].getNumRows(); ++j) {
            for (size_t k = 0; k < deltaWeights[i].getNumCols(); ++k) {
                deltaBiases.setItem(i,k, deltaBiases.getItem(i,k) + deltaWeights[i].getItem(j, k));
            }
        }
    }
}

void Network::updateWeights(size_t batchSize) {
    optimizer.update(deltaWeights, activationResults, deltaBiases, batchSize);
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

            backProp(labels);
            // ToDo: L2
            updateWeights(batchSize);
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