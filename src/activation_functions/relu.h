//
// Created by Dominik Klement on 17/10/2021.
//

#ifndef FEEDFORWARDNEURALNET_RELU_H
#define FEEDFORWARDNEURALNET_RELU_H

#include "template.h"
#include <algorithm>

class ReLU : public ActivationFunctionTemplate {
public:
    static void normal(Matrix<type> &matrix) {
        auto fn = [](type x) {
            return std::max(type{}, x);
        };
        matrix.applyFunction(fn);
    }

    static void derivative(Matrix<type> &matrix) {
        auto fn = [](type x) {
            return x > 0 ? 1 : 0;
        };
        matrix.applyFunction(fn);
    }
};

#endif //FEEDFORWARDNEURALNET_RELU_H
