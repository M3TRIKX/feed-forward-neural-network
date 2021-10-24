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
public:
    auto static crossentropy(const Matrix<float> &predicted, const std::vector<int> &expected){
        float crossentropy = 0;
        for (int i = 0; i < (int) predicted.getNumRows(); i++){
            for (int j = 0; j < (int) predicted.getNumCols(); j++){
                int expectedValue = expected[i] == j ? 1 : 0;
                float calculatedValue = predicted.getItem(i, j);
                crossentropy -= expectedValue * log(calculatedValue) + (1 - expectedValue) * log(1 - calculatedValue);
            }
        }
        return crossentropy;
    }

    auto static crossentropyDerivative(Matrix<float> &predicted, const std::vector<int> &expected){
        float crossentropy = 0;
        for (int i = 0; i < (int) predicted.getNumRows(); i++){
            for (int j = 0; j < (int) predicted.getNumCols(); j++){
                int expectedValue = expected[i] == j ? 1 : 0;
                float calculatedValue = predicted.getItem(i, j);
                crossentropy -= expectedValue / calculatedValue + (1 - expectedValue) / (1 - calculatedValue);
            }
        }
        return crossentropy;
    }
};
#endif //FEEDFORWARDNEURALNET_CROSSENTROPY_H
