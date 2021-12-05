//
// Created by Dominik Klement on 08/11/2021.
//

#ifndef FEEDFORWARDNEURALNET_SGD_H
#define FEEDFORWARDNEURALNET_SGD_H

#include "optimizer_template.h"

class SGDOptimizer : public Optimizer {

public:
    SGDOptimizer() = default;

    virtual void update(const std::vector<Matrix<float>> &weightDeltas,
                        const std::vector<std::vector<float>> &deltaBias,
                        size_t batchSize, float eta) override {

        float batchEta = eta / static_cast<float>(batchSize);

#pragma omp parallel for default(none) shared(weightDeltas, batchEta, deltaBias)
        for (size_t layer = 0; layer < weights->size(); layer++) {
            auto &weightDelta = weightDeltas[layer];
            (*weights)[layer] -= weightDelta * batchEta;

            for (size_t i = 0; i < (*biases)[layer].size(); i++) {
                (*biases)[layer][i] -= batchEta * deltaBias[layer][i];
            }
        }
    };
};

#endif //FEEDFORWARDNEURALNET_SGD_H
