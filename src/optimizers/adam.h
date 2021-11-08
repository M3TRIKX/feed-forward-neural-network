//
// Created by Dáša Pawlasová on 06.11.2021.
//

#ifndef FEEDFORWARDNEURALNET_ADAM_H
#define FEEDFORWARDNEURALNET_ADAM_H

#include <cmath>
#include "../data_structures/matrix.h"
#include "../network/config.h"
#include "optimizer_template.h"

class AdamOptimizer : public Optimizer{

    constexpr static float eps = 1e-8;
    float beta1;
    float beta2;
    size_t t;
    std::vector<Matrix<float>> mw;
    std::vector<std::vector<float>> mb;
    std::vector<Matrix<float>> vw;
    std::vector<std::vector<float>> vb;

public:
    AdamOptimizer() = default;
    AdamOptimizer(std::vector<Matrix<float>> &networkWeights, std::vector<std::vector<float>> &networkBiases,
            float beta1 = 0.9, float beta2 = 0.999, float eta = 0.01): Optimizer(networkWeights, networkBiases, eta){

        this->beta1 = beta1;
        this->beta2 = beta2;
        t = 1;

        for (size_t i = 0; i < weights->size(); ++i){
            auto rows = (*weights)[i].getNumRows();
            auto cols = (*weights)[i].getNumCols();
            mw.push_back(Matrix<float>(rows, cols, 0));
            vw.push_back(Matrix<float>(rows, cols, 0));
            mb.push_back(std::vector<float>((*biases)[i].size(), 0));
            vb.push_back(std::vector<float>((*biases)[i].size(), 0));
        }
    }

    void update(std::vector<Matrix<float>> &deltaWeights, std::vector<Matrix<float>> &activationResults, Matrix<float> &deltaBiases, size_t batchSize) override {
        float batchEta = eta/batchSize;
        for (size_t layer = 0; layer < weights->size(); ++layer) {
            auto weightDelta = activationResults[layer].transpose().matmul(deltaWeights[layer]);
            for (size_t i = 0; i < (*weights)[layer].getNumRows(); ++i){
                for (size_t j = 0; j < (*weights)[layer].getNumCols(); ++j){
                    mw[layer].setItem(i, j, beta1 * mw[layer].getItem(i, j) + (1 - beta1) * weightDelta.getItem(i, j));
                    vw[layer].setItem(i, j, beta2 * vw[layer].getItem(i, j) + (1 - beta2) * std::pow(weightDelta.getItem(i, j), 2));

                    auto mw_corr = mw[layer].getItem(i, j) / (1 - std::pow(beta1, t));
                    auto vw_corr = vw[layer].getItem(i, j) / (1 - std::pow(beta2, t));

                    (*weights)[layer].setItem(i, j, (*weights)[layer].getItem(i, j) - batchEta * (mw_corr / (std::sqrt(vw_corr) + eps)));
                    if(i == 0){
                        mb[layer][j] = beta1 * mb[layer][j] + (1 - beta1) * deltaBiases.getItem(layer,j);
                        vb[layer][j] = beta2 * vb[layer][j] + (1 - beta2) * std::pow(deltaBiases.getItem(layer,j), 2);
                        auto mb_corr = mb[layer][j] / (1 - std::pow(beta1, t));
                        auto vb_corr = vb[layer][j] / (1 - std::pow(beta2, t));
                        (*biases)[layer][j] -= batchEta * (mb_corr / (std::sqrt(vb_corr) + eps));
                    }
                }
            }
        }
        t += 1;
    }

};

#endif //FEEDFORWARDNEURALNET_ADAM_H
