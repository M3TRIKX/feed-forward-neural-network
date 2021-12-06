//
// Created by Dáša Pawlasová on 06.11.2021.
//

#ifndef FEEDFORWARDNEURALNET_ADAM_H
#define FEEDFORWARDNEURALNET_ADAM_H

#include "../data_structures/matrix.hpp"
#include "../network/config.hpp"
#include "optimizer_template.hpp"
#include <cmath>
#include <math.h>

class AdamOptimizer : public Optimizer {

    constexpr static float eps = 1e-7; // Small value to avoid dividing by zero
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
    /**
     * Creates Adam optimizer with it|s parameters
     * @param beta1 - exponential decay rate for the first moment estimates
     * @param beta2 - exponential decay rate for the second-moment estimates
     */
    AdamOptimizer(float beta1 = 0.9, float beta2 = 0.999)
            : beta1(beta1), beta2(beta2), beta1Power(beta1), beta2Power(beta2), t(1) {}

    void init() override {
        for (size_t i = 0; i < weights->size(); ++i) {
            auto rows = (*weights)[i].getNumRows();
            auto cols = (*weights)[i].getNumCols();
            mw.push_back(Matrix<float>(rows, cols, 0));
            vw.push_back(Matrix<float>(rows, cols, 0));
            mb.push_back(std::vector<float>((*biases)[i].size(), 0));
            vb.push_back(std::vector<float>((*biases)[i].size(), 0));
        }
    }

    void update(const std::vector<Matrix<float>> &weightDeltas, const std::vector<std::vector<float>> &deltaBiases,
                size_t batchSize, float eta) override {

        float batchEta = eta / static_cast<float>(batchSize);
        float beta1Prime = 1 - beta1;
        float beta2Prime = 1 - beta2;
        float beta1PrimePower = 1 - beta1Power;
        float beta2PrimePower = 1 - beta2Power;

#pragma omp parallel default(none) shared(weightDeltas, deltaBiases, batchEta, beta1Prime, beta2Prime, beta1PrimePower, beta2PrimePower)
        {
#pragma omp for nowait
            for (size_t layer = 0; layer < weights->size(); ++layer) {
                auto &weightDelta = weightDeltas[layer];

                for (size_t i = 0; i < (*weights)[layer].getNumRows(); ++i) {
#pragma omp simd
                    for (size_t j = 0; j < (*weights)[layer].getNumCols(); ++j) {
                        mw[layer].setItem(i, j,
                                          beta1 * mw[layer].getItem(i, j)
                                          + beta1Prime * weightDelta.getItem(i, j));
                        vw[layer].setItem(i, j,
                                          beta2 * vw[layer].getItem(i, j)
                                          + beta2Prime * powf(weightDelta.getItem(i, j), 2));

                        auto mw_corr = mw[layer].getItem(i, j) / beta1PrimePower;
                        auto vw_corr = vw[layer].getItem(i, j) / beta2PrimePower;

                        (*weights)[layer].setItem(i, j,
                                                  (*weights)[layer].getItem(i, j)
                                                  - batchEta * (mw_corr / (sqrtf(vw_corr) + eps)));
                    }
                }

                (*weights)[layer].transpose((*weightsTransposed)[layer]);
            }

#pragma omp for
            for (size_t layer = 0; layer < weights->size(); ++layer) {
#pragma omp simd
                for (size_t j = 0; j < (*weights)[layer].getNumCols(); ++j) {
                    mb[layer][j] = beta1 * mb[layer][j] + beta1Prime * deltaBiases[layer][j];
                    vb[layer][j] = beta2 * vb[layer][j] + beta2Prime * powf(deltaBiases[layer][j], 2);

                    auto mb_corr = mb[layer][j] / beta1PrimePower;
                    auto vb_corr = vb[layer][j] / beta2PrimePower;

                    (*biases)[layer][j] -= batchEta * (mb_corr / (sqrtf(vb_corr) + eps));
                }
            }
        }

        t += 1;
        beta1Power *= beta1;
        beta2Power *= beta2;
    }
};

#endif //FEEDFORWARDNEURALNET_ADAM_H
