//
// Created by Dáša Pawlasová on 25.10.2021.
//

#ifndef FEEDFORWARDNEURALNET_WEIGHTS_INFO_H
#define FEEDFORWARDNEURALNET_WEIGHTS_INFO_H

#include "../data_structures/matrix.h"
#include <math.h>
#include <algorithm>

struct WeightStats {
    float sum = 0;
    float minimum = 0;
    float maximum = 0;
    float median = 0;
    float average = 0;
};

/**
 * Class containing different ways of printing and retrieving stats about weights
 */
class WeightInfo {
    static const int DECIMAL_PLACES_IN_PRINT = 4;

public:

    /**
     * Gets stats about weights
     * @param weights - matrix of weights
     * @return - weight stats
     */
    static WeightStats statsOfWeights(const Matrix<float> &weights){
        float sum = 0;
        auto vectorOfWeights = std::vector<float>(weights.getNumRows()*weights.getNumCols());
        for (size_t i = 0; i < weights.getNumRows(); i++){
            for(size_t j = 0; j < weights.getNumCols(); j++){
                sum += weights.getItem(i,j);
                vectorOfWeights[i * weights.getNumCols() + j] = weights.getItem(i,j);
            }
        }
        std::sort(vectorOfWeights.begin(), vectorOfWeights.end());
        return {.sum = sum, .minimum = vectorOfWeights[0], .maximum = vectorOfWeights[vectorOfWeights.size() - 1],
                .median = vectorOfWeights[vectorOfWeights.size()/2], .average = sum/vectorOfWeights.size()};
    }

    /**
     * Prints weights
     * @param weights - matrix[i][j] of weights from neuron j to neuron i
     */
    static void printWeights(const Matrix<float> &weights){
        std::cout << "from -(weight)→ to" << std::endl;
        for (size_t i = 0; i < weights.getNumCols(); i++){
            for (size_t j = 0; j < weights.getNumRows(); j++){
                std::cout << i << " -(" << std::fixed << std::setprecision(DECIMAL_PLACES_IN_PRINT) << weights.getItem(j,i) << ")→ " << j << "    ";
            }
            std::cout << std::endl;
        }
    }

    /**
     * Prints stats about weights
     * @param weights - matrix of weights
     * @param amountWeights - print amount of weights
     * @param printWeightMatrix - print weights
     */
    static void printWeightStats(const Matrix<float> &weights, bool amountWeights=false, bool printWeightMatrix=false){
        auto stats = statsOfWeights(weights);
        std::cout << "-----------Info about weights-----------" << std::endl;
        if(amountWeights){
            std::cout << "Neurons in lower layer: " << weights.getNumRows() << std::endl;
            std::cout << "Neurons in upper layer: " << weights.getNumCols() << std::endl;
            std::cout << "Amount of weights: " << weights.getNumCols() * weights.getNumRows() << std::endl;
        }
        std::cout << "Sum: " << stats.sum << std::endl;
        std::cout << "Minimum: " << stats.minimum << std::endl;
        std::cout << "Maximum: " << stats.maximum << std::endl;
        std::cout << "Average: " << stats.average << std::endl;
        std::cout << "Median: " << stats.median << std::endl;
        if (printWeightMatrix){
            printWeights(weights);
        }
        std::cout << "-----------------------------------------" << std::endl;
    }
};

#endif //FEEDFORWARDNEURALNET_WEIGHTS_INFO_H
