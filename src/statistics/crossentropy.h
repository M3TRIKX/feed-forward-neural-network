//
// Created by Dáša Pawlasová on 19.10.2021.
//

#ifndef FEEDFORWARDNEURALNET_CROSSENTROPY_H
#define FEEDFORWARDNEURALNET_CROSSENTROPY_H

#include "../data_structures/matrix.h"
#include <math.h>

/**
 * Class containing cross-entropy function and its derivative
 */
class CrossentropyFunction {
    constexpr static float zeroCorrection = 1e-7;
public:
    /**
     * Calculates cross-entropy of predictions
     * @param predicted - matrix of predictions
     * @param expected - expected labels
     * @return cross-entropy
     */
    auto static crossentropy(const Matrix<float> &predicted, const std::vector<unsigned int> &expected) {
        float crossEntropyRes = 0;

        for (size_t i = 0; i < predicted.getNumRows(); i++) {
            for (size_t j = 0; j < predicted.getNumCols(); j++) {
                float expectedValue = expected[i] == j ? 1.f : 0.f;
                float calculatedValue = predicted.getItem(i, j);
                crossEntropyRes -= expectedValue * logf(calculatedValue + zeroCorrection) +
                                   (1 - expectedValue) * logf(1 - calculatedValue + zeroCorrection);
            }
        }
        return crossEntropyRes / static_cast<float>(expected.size());
    }

    /**
     * Calculates the derivative CE with SoftMax act. fn in the last layer.
     * The derivative is: (y' - y), where y' is the predicted vector.
     * @param predicted
     * @param expected
     * @param labels
     * @return
     */
    auto static costDelta(const Matrix<float> &lastLayerActivationResults, const std::vector<unsigned int> &labels) {
        auto delta = lastLayerActivationResults;

        for (size_t i = 0; i < delta.getNumRows(); ++i) {
            auto j = static_cast<size_t>(labels[i]);
            delta.setItem(i, j, delta.getItem(i, j) - 1);
        }

        return delta;
    }
};

#endif //FEEDFORWARDNEURALNET_CROSSENTROPY_H
