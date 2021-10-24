//
// Created by Dáša Pawlasová on 19.10.2021.
//

#ifndef FEEDFORWARDNEURALNET_ARGMAX_H
#define FEEDFORWARDNEURALNET_ARGMAX_H

#include "../data_structures/matrix.h"

/**
 * Class containing argmax function
 */
class ArgmaxFunction {
public:
    /**
     * Function calculating argmax or each row in the matrix
     * @param matrix - matrix to compute argmax on
     * @return - vector of classes
     */
    auto static argmax(const Matrix<float> &matrix){
        auto classes = std::vector<size_t>(matrix.getNumRows());
        for (size_t i = 0; i < matrix.getNumRows(); i++){
            float currentMax = 0;
            for (size_t j = 0; j < matrix.getNumCols(); j++){
                if (matrix.getItem(i,j) > currentMax){
                    currentMax = matrix.getItem(i,j);
                    classes[i] = j;
                }
            }
        }
        return classes;
    }
};

#endif //FEEDFORWARDNEURALNET_ARGMAX_H
