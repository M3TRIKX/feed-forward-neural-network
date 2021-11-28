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
    }

    static void derivative(Matrix<type> &matrix) {
        auto fn = [](type x) {
            type y = 1 / (1 + exp(-x));
            return y * (1 - y);
        };
        matrix.applyFunction(fn);
    }
};


#endif //FEEDFORWARDNEURALNET_SIGMOID_H
