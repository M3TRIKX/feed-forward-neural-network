//
// Created by Dominik Klement on 17/10/2021.
//

#ifndef FEEDFORWARDNEURALNET_RELU_H
#define FEEDFORWARDNEURALNET_RELU_H
#include "template.h"
#include <math.h>
#include <algorithm>

class ReLU: public ActivationFunction {
public:
    static type normal(type x) {
        return std::max(0.f, x);
    }

    static type derivative(type x) {
        if (x > 0)
            return 1;
        return 0;
    }
};

#endif //FEEDFORWARDNEURALNET_RELU_H
