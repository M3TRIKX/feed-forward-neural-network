//
// Created by Dáša Pawlasová on 06.11.2021.
//

#ifndef FEEDFORWARDNEURALNET_OPTIMIZER_TEMPLATE_H
#define FEEDFORWARDNEURALNET_OPTIMIZER_TEMPLATE_H

#include "../data_structures/matrix.hpp"

/**
 * Class representing optimizer
 */
class Optimizer {
protected:
    std::vector<Matrix<float>> *weights = nullptr;
    std::vector<Matrix<float>> *weightsTransposed = nullptr;
    std::vector<std::vector<float>> *biases = nullptr;

public:
    Optimizer() = default;

    virtual ~Optimizer() = default;

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
    void setMatrices(std::vector<Matrix<float>> &weights, std::vector<Matrix<float>> &weightsTransposed,
                     std::vector<std::vector<float>> &biases) {

        this->weights = &weights;
        this->weightsTransposed = &weightsTransposed;
        this->biases = &biases;
    }

    /**
     * Updates weights and biases using chosen optimization technique
     * @param weightDeltas - Derivative of the loss function w.r.t. weights
     * @param batchSize - Data batch size
     * @param eta - Learning rate
     */
    virtual void
    update(const std::vector<Matrix<float>> &weightDeltas, const std::vector<std::vector<float>> &deltaBias,
           size_t batchSize, float eta) = 0;
};

#endif //FEEDFORWARDNEURALNET_OPTIMIZER_TEMPLATE_H
