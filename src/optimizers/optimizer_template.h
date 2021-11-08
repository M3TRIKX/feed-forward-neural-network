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
    std::vector<std::vector<float>> *biases = NULL;
public:
    Optimizer() = default;

    /**
     * Updates weights and biases using choosen optimization technique
     * @param deltaWeights - Weight deltas
     * @param deltaBias - Bias deltas
     */
    virtual void update(std::vector<Matrix<float>> &deltaWeights, std::vector<Matrix<float>> &activationResults, Matrix<float> &deltaBias, size_t batchSize, float eta) = 0;

    virtual void init() {}

    void setMatrices(std::vector<Matrix<float>> &weights, std::vector<std::vector<float>> &biases) {
        this->weights = &weights;
        this->biases = &biases;
    }
};

#endif //FEEDFORWARDNEURALNET_OPTIMIZER_TEMPLATE_H
