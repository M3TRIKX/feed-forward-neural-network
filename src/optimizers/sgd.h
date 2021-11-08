//
// Created by Dominik Klement on 08/11/2021.
//

#ifndef FEEDFORWARDNEURALNET_SGD_H
#define FEEDFORWARDNEURALNET_SGD_H

#include "optimizer_template.h"

class SGD : public Optimizer {

public:
    SGD() = default;


    /**
     * Updates weights and biases using choosen optimization technique
     * @param deltaWeights - Weight deltas
     * @param deltaBias - Bias deltas
     */
    virtual void update(std::vector<Matrix<float>> &deltaWeights, std::vector<Matrix<float>> &activationResults, Matrix<float> &deltaBias, size_t batchSize, float eta) override {
        float batchEta = eta / static_cast<float>(batchSize);
        for (size_t layer = 0; layer < weights->size(); layer++) {
            auto weightDelta = activationResults[layer].transpose().matmul(deltaWeights[layer]);
            (*weights)[layer] -= weightDelta * batchEta;
            for (size_t i = 0; i < (*biases)[layer].size(); i++){
                (*biases)[layer][i] -= batchEta * deltaBias.getItem(layer, i);
            }
        }
    };
};

#endif //FEEDFORWARDNEURALNET_SGD_H
