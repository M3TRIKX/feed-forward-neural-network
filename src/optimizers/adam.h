//
// Created by Dáša Pawlasová on 06.11.2021.
//

#ifndef FEEDFORWARDNEURALNET_ADAM_H
#define FEEDFORWARDNEURALNET_ADAM_H

#include <cmath>
#include "../data_structures/matrix.h"
#include "../network/config.h"
#include "optimizer_template.h"

/**
 * Class representing Adam optimizer
 */
class AdamOptimizer : public Optimizer{

    constexpr static float eps = 1e-8; // Small value to avoid dividing by zero
    float beta1;
    float beta2;
    float beta1Power;
    float beta2Power;
    size_t t; // Current iteration
    std::vector<Matrix<float>> mw;
    std::vector<std::vector<float>> mb;
    std::vector<Matrix<float>> vw;
    std::vector<std::vector<float>> vb;

public:
    AdamOptimizer(float beta1 = 0.9, float beta2 = 0.999)
        : beta1(beta1), beta2(beta2), beta1Power(beta1), beta2Power(beta2), t(1) {}

    void init() override {
        for (size_t i = 0; i < weights->size(); ++i){
            auto rows = (*weights)[i].getNumRows();
            auto cols = (*weights)[i].getNumCols();
            mw.push_back(Matrix<float>(rows, cols, 0));
            vw.push_back(Matrix<float>(rows, cols, 0));
            mb.push_back(std::vector<float>((*biases)[i].size(), 0));
            vb.push_back(std::vector<float>((*biases)[i].size(), 0));
        }
    }

//    void update(std::vector<Matrix<float>> &deltaWeights, std::vector<Matrix<float>> &activationResults, std::vector<std::vector<float>> &deltaBiases,
//            size_t batchSize, float eta) override {
//        float batchEta = eta / static_cast<float>(batchSize);
//
//#pragma omp parallel for num_threads(3) default(none) shared(deltaWeights, deltaBiases, activationResults, batchEta)
//        for (size_t layer = 0; layer < weights->size(); ++layer) {
//            // Update weights
//            auto weightDelta = activationResults[layer].transpose().matmul(deltaWeights[layer]);
//            mw[layer] *= beta1;
//            mw[layer] += weightDelta * (1 - beta1);
//            vw[layer] *= beta2;
//            vw[layer] += weightDelta.pow(2) * (1 - beta2);
//            auto mw_corr = mw[layer] / (1 - beta1Power);
//            auto vw_corr = vw[layer] / (1 - beta2Power);
//            (*weights)[layer] -= (mw_corr / (vw_corr.sqrt() + eps)) * batchEta;
//
//            // Update biases
//            for (int j = 0; j < (*weights)[layer].getNumCols(); j++){
//                mb[layer][j] = beta1 * mb[layer][j] + (1 - beta1) * deltaBiases[layer][j];
//                vb[layer][j] = beta2 * vb[layer][j] + (1 - beta2) * std::pow(deltaBiases[layer][j], 2);
//                auto mb_corr = mb[layer][j] / (1 - beta1Power);
//                auto vb_corr = vb[layer][j] / (1 - beta2Power);
//                (*biases)[layer][j] -= batchEta * (mb_corr / (std::sqrt(vb_corr) + eps));
//            }
//
//            (*weights)[layer].transpose((*weightsTransposed)[layer]);
//        }
//#pragma omp barrier
//
//        ++t;
//        beta1Power *= beta1;
//        beta2Power *= beta2;
//    }

    void update(std::vector<Matrix<float>> &deltaWeights, std::vector<Matrix<float>> &activationResults, std::vector<std::vector<float>> &deltaBiases,
                size_t batchSize, float eta) override {
        float batchEta = eta / static_cast<float>(batchSize);

#pragma omp parallel for default(none) shared(deltaWeights, deltaBiases, activationResults, batchEta)
        for (size_t layer = 0; layer < weights->size(); ++layer) {
            auto weightDelta = activationResults[layer].transpose().matmul(deltaWeights[layer]);
            for (size_t i = 0; i < (*weights)[layer].getNumRows(); ++i) {
#pragma omp simd
                for (size_t j = 0; j < (*weights)[layer].getNumCols(); ++j) {
                    mw[layer].setItem(i, j, beta1 * mw[layer].getItem(i, j) + (1 - beta1) * weightDelta.getItem(i, j));
                    vw[layer].setItem(i, j, beta2 * vw[layer].getItem(i, j) + (1 - beta2) * std::pow(weightDelta.getItem(i, j), 2));

                    auto mw_corr = mw[layer].getItem(i, j) / (1 - beta1Power);
                    auto vw_corr = vw[layer].getItem(i, j) / (1 - beta2Power);

                    (*weights)[layer].setItem(i, j, (*weights)[layer].getItem(i, j) - batchEta * (mw_corr / (std::sqrt(vw_corr) + eps)));
                }
            }

#pragma omp simd
            for (size_t j = 0; j < (*weights)[layer].getNumCols(); ++j) {
                mb[layer][j] = beta1 * mb[layer][j] + (1 - beta1) * deltaBiases[layer][j];
                vb[layer][j] = beta2 * vb[layer][j] + (1 - beta2) * std::pow(deltaBiases[layer][j], 2);
                auto mb_corr = mb[layer][j] / (1 - beta1Power);
                auto vb_corr = vb[layer][j] / (1 - beta2Power);
                (*biases)[layer][j] -= batchEta * (mb_corr / (std::sqrt(vb_corr) + eps));
            }

//            (*weightsTransposed)[layer] = (*weights)[layer].transpose();
            (*weights)[layer].transpose((*weightsTransposed)[layer]);
        }

#pragma omp barrier

        t += 1;
        beta1Power *= beta1;
        beta2Power *= beta2;
    }


};

#endif //FEEDFORWARDNEURALNET_ADAM_H
