//
// Created by Dominik Klement on 17/10/2021.
//

#ifndef FEEDFORWARDNEURALNET_FAST_SIGMOID_H
#define FEEDFORWARDNEURALNET_FAST_SIGMOID_H

#include "template.hpp"

/**
 * FastSigmoid is an approximation to the sigmoid function.
 * The reason for using the fast version over the normal one is the speed
 * of computation.
 */
class FastSigmoid : public ActivationFunctionTemplate {
public:
    static void normal(Matrix<type> &matrix) {
        auto fn = [](type x) {
            return x / (1 + abs(x));
        };
        matrix.applyFunction(fn);
    }

    static void derivative(Matrix<type> &matrix) {
        auto fn = [](type x) {
            return x / (1 + abs(x));
        };

        matrix.applyFunction(fn);
    }
};

#endif //FEEDFORWARDNEURALNET_FAST_SIGMOID_H
