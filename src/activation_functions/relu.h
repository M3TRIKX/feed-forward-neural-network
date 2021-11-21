//
// Created by Dominik Klement on 17/10/2021.
//

#ifndef FEEDFORWARDNEURALNET_RELU_H
#define FEEDFORWARDNEURALNET_RELU_H

#include "template.h"
#include <algorithm>
#include <math.h>

/**
 * ReLU activation function
 */
class ReLU: public ActivationFunctionTemplate {
public:
    static void normal(Matrix<type> &matrix) {
        auto fn = [](type x) {
            // ToDo: Change to float in the end.
            return std::max(0.f, x);
        };
        matrix.applyFunction(fn);
//        return std::max(0.f, x);

    }

    static void derivative(Matrix<type> &matrix) {
        auto fn = [](type x) {
            return x > 0 ? 1 : 0;
        };
        matrix.applyFunction(fn);
    }
};

#endif //FEEDFORWARDNEURALNET_RELU_H
