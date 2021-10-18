//
// Created by Dominik Klement on 12/10/2021.
//

#ifndef FEEDFORWARDNEURALNET_SIGMOID_H
#define FEEDFORWARDNEURALNET_SIGMOID_H

#include "template.h"
#include <math.h>

class Sigmoid: public ActivationFunction {
public:
    static type normal(type x) {
        return 1 / (1 + exp(-x));
    }

    static type derivative(type x) {
        type n = normal(x);
        return n * (1 - n);
    }
};


#endif //FEEDFORWARDNEURALNET_SIGMOID_H
