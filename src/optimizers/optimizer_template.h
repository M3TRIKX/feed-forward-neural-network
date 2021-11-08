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
    float eta;
public:
    Optimizer() = default;

    Optimizer(std::vector<Matrix<float>> &weights, std::vector<std::vector<float>> &biases, float eta): weights(&weights), biases(&biases){
        this->eta = eta;
    }

    /**
     * Updates weights and biases using choosen optimization technique
     * @param deltaWeights - Weight deltas
     * @param deltaBias - Bias deltas
     */
    virtual void update(std::vector<Matrix<float>> &deltaWeights, std::vector<Matrix<float>> &activationResults, Matrix<float> &deltaBias, size_t batchSize){
        float batchEta = eta/batchSize;
        for (size_t layer = 0; layer < weights->size(); layer++) {
            auto weightDelta = activationResults[layer].transpose().matmul(deltaWeights[layer]);
            (*weights)[layer] -= weightDelta * batchEta;
            for (size_t i = 0; i < (*biases)[layer].size(); i++){
                (*biases)[layer][i] -= batchEta * deltaBias.getItem(layer, i);
            }
        }
    };
};

#endif //FEEDFORWARDNEURALNET_OPTIMIZER_TEMPLATE_H
