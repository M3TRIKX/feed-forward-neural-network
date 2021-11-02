//
// Created by dominik on 22. 10. 2021.
//

#ifndef FEEDFORWARDNEURALNET_NETWORK_H
#define FEEDFORWARDNEURALNET_NETWORK_H

#include <vector>
#include "../data_structures/matrix.h"
#include "config.h"
#include "../statistics/stats_printer.h"

class WrongInputDataDimension : public std::exception {};

class WrongOutputActivationFunction : public std::exception{};

class Network {
    using ELEMENT_TYPE = float;

    const Config &networkConfig;
    std::vector<Matrix<ELEMENT_TYPE>> weights;
    std::vector<std::vector<ELEMENT_TYPE>> biases;

    std::vector<Matrix<ELEMENT_TYPE>> activationDerivResults;
    std::vector<Matrix<ELEMENT_TYPE>> activationResults;

    std::vector<Matrix<ELEMENT_TYPE>> deltas;

public:
    Network(const Config &config): networkConfig(config) {
//        srand(time(NULL));

        // ToDo: Alloc based on the num of threads
        activationDerivResults.emplace_back();
        activationResults.emplace_back();

        deltas.reserve(config.layersConfig.size());

        // We are initializing weights between each two layers.
        // weight[k][i][j] corresponds to the weight between neuron ith neuron in layer k
        // and jth neuron in layer k+1.
        for (size_t i = 0; i < config.layersConfig.size() - 1; ++i) {
            const auto &layer = config.layersConfig[i];
            const auto &nextLayer = config.layersConfig[i + 1];
            // ToDo: Add more sophisticated weight initialization.
            weights.push_back(Matrix<float>::generateRandomMatrix(layer.numNeurons, nextLayer.numNeurons, -1, 1));

            // Init biases as zero
            biases.emplace_back(nextLayer.numNeurons, 0);

            deltas.emplace_back(0, 0, 0);
        }
    }

    const auto &forwardPass(const Matrix<ELEMENT_TYPE> &data, const std::vector<ELEMENT_TYPE> &labels) {
        activationResults.clear();
        activationDerivResults.clear();
        // Input layer has activation fn equal to identity.
        activationResults.push_back(data);

        auto tmp = data.matmul(weights[0]);
        tmp += biases[0];

        networkConfig.layersConfig[1].activationFunction(tmp);
        activationResults.push_back(tmp);

        auto tmpCopy = tmp;
        networkConfig.layersConfig[1].activationDerivFunction(tmpCopy);
        activationDerivResults.push_back(tmpCopy);

//        tmp.printMatrix();
//        std::cout << std::endl;

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

//            tmp.printMatrix();
//            std::cout << std::endl;
        }

        // ToDo: Change size_t to int during reading - choose the type.
        auto stats = StatsPrinter::getStats(tmp, std::vector<size_t>(labels.cbegin(), labels.cend()));
        std::cout << "ACC: " << stats.accuracy << " CE: " << stats.crossEntropy << std::endl;

        return activationResults[activationResults.size() - 1];
    }

    void backProp(const std::vector<ELEMENT_TYPE> &labels, float eta) {
        const auto &lastLayerConf = networkConfig.layersConfig[networkConfig.layersConfig.size() - 1];
        if (lastLayerConf.activationFunctionType != ActivationFunction::SoftMax) {
            throw WrongOutputActivationFunction();
        }

//        size_t k = std::min(labels.size(), networkConfig.batchSize);
        size_t numLayers = networkConfig.layersConfig.size();
        // Derivative of Softmax and CrossEntropy = (y' - y), where y' is predicted vector
        // and y is ground truth vector.
        auto lastLayerOutputDelta = activationResults[numLayers - 1];
        for (size_t i = 0; i < lastLayerOutputDelta.getNumRows(); ++i) {
            for (size_t j = 0; j < lastLayerOutputDelta.getNumCols(); ++j) {
                if (j == labels[i]) {
                    lastLayerOutputDelta.setItem(i, j, lastLayerOutputDelta.getItem(i, j) - 1);
                }
            }
        }
        deltas[numLayers - 2] = lastLayerOutputDelta;

        auto lastDelta = std::move(lastLayerOutputDelta);
        for (int i = numLayers - 2; i > 0; --i) {
            auto matmuls = lastDelta.matmul(weights[i].transpose());
            auto newDelta = matmuls * activationDerivResults[i-1];
            deltas[i - 1] = newDelta;
            lastDelta = newDelta;
        }

        size_t numOfObservations = labels.size();

        // Update weights
        for (size_t i = 0; i < numLayers - 1; ++i) {
            auto weightDelta = activationResults[i].transpose().matmul(deltas[i]);
            weights[i] -= weightDelta * (eta / numOfObservations);

            // Bias computation
            // ToDo: Make this code faster even tho i have no idea how.
            std::vector<ELEMENT_TYPE> biasDelta(biases[i].size(), 0);
            for (size_t j = 0; j < deltas[i].getNumRows(); ++j) {
                for (size_t k = 0; k < deltas[i].getNumCols(); ++k) {
                    biasDelta[k] += deltas[i].getItem(j, k);
                }
            }

            for (size_t j = 0; j < biases[i].size(); ++j) {
                biases[i][j] -= (eta / numOfObservations) * biasDelta[j];
            }
        }
    }
};

#endif //FEEDFORWARDNEURALNET_NETWORK_H
