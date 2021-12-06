//
// Created by dominik on 23. 10. 2021.
//

#ifndef FEEDFORWARDNEURALNET_SOFTMAX_H
#define FEEDFORWARDNEURALNET_SOFTMAX_H

#include "template.hpp"

class SoftMax : public ActivationFunctionTemplate {
public:
    static void normal(Matrix<type> &matrix) {
        for (size_t i = 0; i < matrix.getNumRows(); ++i) {
            type rowSum = 0;
            type rowMax = matrix.getMaxRowElement(i);

            for (size_t j = 0; j < matrix.getNumCols(); ++j) {
                float item = expf(matrix.getItem(i, j) - rowMax);
                rowSum += item;
                matrix.setItem(i, j, item);
            }

            for (size_t j = 0; j < matrix.getNumCols(); ++j) {
                matrix.setItem(i, j, matrix.getItem(i, j) / rowSum);
            }
        }
    }

    // Derivative is implemented ih the cross entropy delta.
};

#endif //FEEDFORWARDNEURALNET_SOFTMAX_H
