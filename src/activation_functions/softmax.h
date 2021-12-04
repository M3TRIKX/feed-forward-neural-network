//
// Created by dominik on 23. 10. 2021.
//

#ifndef FEEDFORWARDNEURALNET_SOFTMAX_H
#define FEEDFORWARDNEURALNET_SOFTMAX_H

#include <numeric>
#include <math.h>
#include "template.h"
#include "../data_structures/matrix.h"

class SoftMax: public ActivationFunctionTemplate {
public:
    static void normal(Matrix<type> &matrix) {
        for (size_t i = 0; i < matrix.getNumRows(); ++i) {
            type rowSum = 0;
            type rowMax = matrix.getMaxRowElement(i);

            for (size_t j = 0; j < matrix.getNumCols(); ++j) {
                // ToDo: Solve possible numeric error (due to float addition).
                float item = std::exp(matrix.getItem(i, j) - rowMax);
                rowSum += item;
                matrix.setItem(i, j, item);
            }

            for (size_t j = 0; j < matrix.getNumCols(); ++j) {
                matrix.setItem(i, j, matrix.getItem(i, j) / rowSum);
            }
        }
    }

    static type derivative(type x) {
        // ToDo: Implement.
        throw std::exception();
        return -1;
    }
};


#endif //FEEDFORWARDNEURALNET_SOFTMAX_H
