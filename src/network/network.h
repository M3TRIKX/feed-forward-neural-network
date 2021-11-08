//
// Created by dominik on 22. 10. 2021.
//

#ifndef FEEDFORWARDNEURALNET_NETWORK_H
#define FEEDFORWARDNEURALNET_NETWORK_H

#include <vector>
#include "../data_structures/matrix.h"
#include "config.h"
#include "../statistics/stats_printer.h"
#include "../data_manager/data_manager.h"
#include "../optimizers/optimizer_template.h"
#include "../optimizers/adam.h"

class WrongInputDataDimension : public std::exception {};
class WrongOutputActivationFunction : public std::exception{};
class NegativeEtaException : public std::exception{};

class Network {
    using ELEMENT_TYPE = float;

    const Config &networkConfig;
    AdamOptimizer optimizer;
    std::vector<Matrix<ELEMENT_TYPE>> weights;
    std::vector<std::vector<ELEMENT_TYPE>> biases;

    std::vector<Matrix<ELEMENT_TYPE>> activationDerivResults;
    std::vector<Matrix<ELEMENT_TYPE>> activationResults;

    std::vector<Matrix<ELEMENT_TYPE>> deltaWeights;
    Matrix<ELEMENT_TYPE> deltaBiases;

public:
    Network(const Config &config): networkConfig(config) {
//        srand(time(NULL));

        // ToDo: Alloc based on the num of threads
        activationDerivResults.emplace_back();
        activationResults.emplace_back();

        deltaWeights.reserve(config.layersConfig.size());

        // We are initializing weights between each two layers.
        // weight[k][i][j] corresponds to the weight between neuron ith neuron in layer k
        // and jth neuron in layer k+1.
        for (size_t i = 0; i < config.layersConfig.size() - 1; ++i) {
            const auto &layer = config.layersConfig[i];
            const auto &nextLayer = config.layersConfig[i + 1];

            // Uniform HE initialization
            if (nextLayer.activationFunctionType == ActivationFunction::ReLU) {
                float limit = 6 / sqrt(layer.numNeurons);
                weights.push_back(Matrix<float>::generateRandomUniformMatrix(layer.numNeurons, nextLayer.numNeurons, -limit, limit));
            }
            // Uniform Glorot initialization
            else if (nextLayer.activationFunctionType == ActivationFunction::SoftMax) {
                float limit = 6 / sqrt(layer.numNeurons + nextLayer.numNeurons);
                weights.push_back(Matrix<float>::generateRandomUniformMatrix(layer.numNeurons, nextLayer.numNeurons, -limit, limit));
            }
            // Random
            else {
                weights.push_back(Matrix<float>::generateRandomMatrix(layer.numNeurons, nextLayer.numNeurons, -1, 1));
            }

            // Init biases as zero
            biases.emplace_back(nextLayer.numNeurons, 0);

            deltaWeights.emplace_back(0, 0, 0);
        }
        optimizer = AdamOptimizer(weights, biases);
    }

    /**
     * Trains the network.
     * @param trainValSplit Training and validation datasets
     * @param eta           Learning rate
     * @param numEpochs     Number of loops through the training dataset
     * @param batchSize     Number of samples used for a single weight update
     */
    void fit (const TrainValSplit_t &trainValSplit, float eta=0.5, size_t numEpochs = 1, size_t batchSize = 32);

    /**
     * Predicts the data labels (should be ran on a trained network, otherwise it's just a random projection).
     * @param data Data vectors
     * @return Predicted labels (output activations per sample).
     */
    Matrix<ELEMENT_TYPE> predict(const Matrix<float> &data);

private:
    /**
     * Do a single forward pass (with multiple data samples at once).
     * The method also sets the activation and activation derivative results
     * for the backprop.
     * @param data   Data vectors we want to evaluate the network on
     * @param labels Label vectors (not one-hot encoded)
     * @return stats (accuracy and cross-entropy)
     */
    auto forwardPass(const Matrix<ELEMENT_TYPE> &data, const std::vector<unsigned int> &labels);

    /**
     * Updates weights and biases based on the forward pass
     * The forward pass must be ran beforehand
     * @param labels raw labels (not one-hot encoded)
     */
    void backProp(const std::vector<unsigned int> &labels);

    /**
     * Updates weights using selected optimizer
     */
    void updateWeights(size_t batchSize = 32);
};

#endif //FEEDFORWARDNEURALNET_NETWORK_H
