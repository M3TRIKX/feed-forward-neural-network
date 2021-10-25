//
// Created by Dominik Klement on 12/10/2021.
//

#ifndef FEEDFORWARDNEURALNET_SIGMOID_H
#define FEEDFORWARDNEURALNET_SIGMOID_H

#include "template.h"
#include <math.h>

class Sigmoid: public ActivationFunctionTemplate {
public:
    static void normal(Matrix<type> &matrix) {
        auto fn = [](type x) {
            return 1 / (1 + exp(-x));
        };
        matrix.applyFunction(fn);
//        return 1 / (1 + exp(-x));
    }

    static type derivative(type x) {
        type n = 1 / (1 + exp(-x));
        return n * (1 - n);
    }
};


#endif //FEEDFORWARDNEURALNET_SIGMOID_H
