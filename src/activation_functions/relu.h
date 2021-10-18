//
// Created by Dominik Klement on 17/10/2021.
//

#ifndef FEEDFORWARDNEURALNET_FAST_SIGMOID_H
#define FEEDFORWARDNEURALNET_FAST_SIGMOID_H

#include "template.h"
#include <math.h>

class ReLU: public ActivationFunction {
public:
    static type normal(type x) {
        return max(0, x);
    }

    static type derivative(type x) {
        if (x > 0)
            return 1;
        return 0;
    }
};

#endif //FEEDFORWARDNEURALNET_FAST_SIGMOID_H
