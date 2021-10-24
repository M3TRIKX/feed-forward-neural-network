//
// Created by Dáša Pawlasová on 19.10.2021.
//

#ifndef FEEDFORWARDNEURALNET_ACCURACY_H
#define FEEDFORWARDNEURALNET_ACCURACY_H

#include "../data_structures/matrix.h"

/**
 * Class containing accuracy function
 */
class AccuracyFunction {
public:
    /**
     * Function calculating accuracy
     * @param calculated - predicted labels
     * @param expected - expected labels
     * @return accuracy of NN predictions
     */
    auto static accuracy(const std::vector<int> &predicted, const std::vector<int> &expected){
        float correctPredictions = 0;
        for (size_t i = 0; i < predicted.size(); i++){
            if (predicted[i] == expected[i]){
                correctPredictions++;
            }
        }
        return correctPredictions / static_cast<float>(totalSamples) * 100;
    }
};
#endif //FEEDFORWARDNEURALNET_ACCURACY_H
