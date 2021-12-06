#ifndef FEEDFORWARDNEURALNET_NETWORK_H
#define FEEDFORWARDNEURALNET_NETWORK_H

#include <vector>
#include "../data_structures/matrix.hpp"
#include "config.hpp"
#include "../statistics/stats.hpp"
#include "../data_manager/data_manager.hpp"
#include "../optimizers/optimizer_template.hpp"
#include "../schedulers/lr_sheduler.hpp"

#ifndef NUM_NET_THREADS
#define NUM_NET_THREADS 5
#endif

class WrongInputDataDimension : public std::exception {
};

class WrongOutputActivationFunction : public std::exception {
};

class NegativeEtaException : public std::exception {
};

class Network {
    using ELEMENT_TYPE = float;

    const Config &networkConfig;
    Optimizer *optimizer;
    std::vector<Matrix<ELEMENT_TYPE>> weights;
    std::vector<Matrix<ELEMENT_TYPE>> weightsTransposed;
    std::vector<std::vector<ELEMENT_TYPE>> biases;

    std::vector<std::vector<Matrix<ELEMENT_TYPE>>> parallelActivationResults;
    std::vector<std::vector<Matrix<ELEMENT_TYPE>>> parallelActivationDerivResults;

    std::vector<std::vector<ELEMENT_TYPE>> deltaBiases;
    std::vector<Matrix<ELEMENT_TYPE>> weightDeltas;

    std::vector<std::vector<Matrix<ELEMENT_TYPE>>> parallelDeltaWeights;
    std::vector<std::vector<std::vector<ELEMENT_TYPE>>> parallelDeltaBiases;

public:
    Network(const Config &config, Optimizer *optimizer) : networkConfig(config), optimizer(optimizer) {
        for (size_t i = 0; i < NUM_NET_THREADS; ++i) {
            parallelActivationResults.emplace_back(config.layersConfig.size());
            parallelActivationDerivResults.emplace_back(config.layersConfig.size());
            parallelDeltaWeights.emplace_back();
            parallelDeltaBiases.emplace_back();
        }

        // We are initializing weights between each two layers.
        // weight[k][i][j] corresponds to the weight between neuron ith neuron in layer k
        // and jth neuron in layer k+1.
        for (size_t i = 0; i < config.layersConfig.size() - 1; ++i) {
            const auto &layer = config.layersConfig[i];
            const auto &nextLayer = config.layersConfig[i + 1];

            // Uniform HE initialization
            if (nextLayer.activationFunctionType == ActivationFunction::ReLU) {
                float limit = 6 / sqrt(layer.numNeurons);
                weights.push_back(
                        Matrix<float>::generateRandomUniformMatrix(layer.numNeurons, nextLayer.numNeurons, -limit,
                                                                   limit));
            }
                // Uniform Glorot initialization
            else {
                float limit = 6 / sqrt(layer.numNeurons + nextLayer.numNeurons);
                weights.push_back(
                        Matrix<float>::generateRandomUniformMatrix(layer.numNeurons, nextLayer.numNeurons, -limit,
                                                                   limit));
            }

            weightDeltas.emplace_back(layer.numNeurons, nextLayer.numNeurons, 0);
            weightsTransposed.push_back(weights[i].transpose());

            // Init biases as zero
            biases.emplace_back(nextLayer.numNeurons, 0);
            deltaBiases.emplace_back(nextLayer.numNeurons, 0);

            for (size_t k = 0; k < NUM_NET_THREADS; ++k) {
                parallelDeltaWeights[k].emplace_back();
                parallelDeltaBiases[k].emplace_back(nextLayer.numNeurons, 0);
            }
        }

        optimizer->setMatrices(weights, weightsTransposed, biases);
        optimizer->init();
    }

    /**
     * Trains the network.
     * @param trainValSplit Training and validation datasets
     * @param eta           Learning rate
     * @param numEpochs     Number of loops through the training dataset
     * @param batchSize     Number of samples used for a single weight update
     */
    void fit(const TrainValSplit_t &trainValSplit, size_t numEpochs = 1, size_t batchSize = 32, float eta = 0.1,
             float lambda = 1e-6, uint8_t verboseLevel = 0, LRScheduler *sched = nullptr,
             size_t earlyStopping = 0,
             long maxTimeMs = 0);

    /**
     * Predicts the data labels (should be ran on a trained network, otherwise it's just a random projection).
     * @param data Data vectors
     * @return Predicted labels (output activations per sample).
     */
    Matrix<ELEMENT_TYPE> predict(const Matrix<float> &data);

    auto predictParallel(const std::vector<Matrix<float>> &data, const std::vector<std::vector<unsigned int>> &labels);

private:
    /**
     * Do single thread forward pass
     * @param data      Train data vectors
     * @param labels    Train labels
     * @param kthThread Thread number
     * @return Single thread batch stats
     */
    auto forwardPass(const Matrix<float> &data, const std::vector<unsigned int> &labels, size_t kthThread);

    /**
     * Do parallel forward & backward pass and compute weight deltas
     * @param data   Train data vectors
     * @param labels Train labels
     * @return Batch train stats
     */
    auto forwardBackwardPass(const std::vector<Matrix<ELEMENT_TYPE>> &data,
                             const std::vector<std::vector<unsigned int>> &labels);

    /**
     * Updates weights using selected optimizer
     */
    void updateWeights(size_t batchSize, float eta);

    /**
     * Calculate weight decay
     * @param lambda Decay rate
     */
    void weightDecay(float lambda);
};

#endif //FEEDFORWARDNEURALNET_NETWORK_H
