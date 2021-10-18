//
// Created by Dominik Klement on 17/10/2021.
//

#ifndef FEEDFORWARDNEURALNET_FAST_SIGMOID_H
#define FEEDFORWARDNEURALNET_FAST_SIGMOID_H

#include "template.h"
#include <math.h>

/**
 * FastSigmoid is an approximation to the sigmoid function.
 * The reason for using the fast version over the normal one is the speed
 * of computation.
 */
class FastSigmoid: public ActivationFunction {
public:
    static type normal(type x) {
        return x / (1 + abs(x));
    }

    static type derivative(type x) {
        type n = normal(x);
        return n * n;
    }
};

#endif //FEEDFORWARDNEURALNET_FAST_SIGMOID_H
