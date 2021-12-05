//
// Created by Dáša Pawlasová on 06.11.2021.
//

#ifndef FEEDFORWARDNEURALNET_OPTIMIZER_TEMPLATE_H
#define FEEDFORWARDNEURALNET_OPTIMIZER_TEMPLATE_H

#include "../data_structures/matrix.h"

/**
 * Class representing optimizer
 */
class Optimizer{
protected:
    std::vector<Matrix<float>> *weights = NULL;
    std::vector<Matrix<float>> *weightsTransposed = NULL;
    std::vector<std::vector<float>> *biases = NULL;

public:
    Optimizer() = default;

    /**
     * Initializes optimizer
     */
    virtual void init() {}

    /**
     * Sets weights and biases to optimizer
     * @param weights - Network weights
     * @param biases - Network biases
     * @param weightsTransposed - Transposed weights
     */
    void setMatrices(std::vector<Matrix<float>> &weights, std::vector<Matrix<float>> &weightsTransposed, std::vector<std::vector<float>> &biases) {
        this->weights = &weights;
        this->weightsTransposed = &weightsTransposed;
        this->biases = &biases;
    }

    /**
     * Updates weights and biases using choosen optimization technique
     * @param deltaWeights - Weight deltas
     * @param activationResults - Neuron potential values after application of activation function
     * @param deltaBias - Bias deltas
     * @param batchSize - Data batch size
     * @param eta - Learning rate
     */
    virtual void update(std::vector<Matrix<float>> &weightDeltas, std::vector<Matrix<float>> &deltaWeights, std::vector<Matrix<float>> &activationResults,
                        std::vector<std::vector<float>> &deltaBias, size_t batchSize, float eta) = 0;
};

#endif //FEEDFORWARDNEURALNET_OPTIMIZER_TEMPLATE_H
