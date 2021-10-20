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
    auto static accuracy(std::vector<int> predicted, std::vector<int> expected){
        int totalSamples = predicted.size();
        float correctPredictions = 0;
        for (int i = 0; i < totalSamples; i++){
            if (predicted[i] == expected[i]){
                correctPredictions++;
            }
        }
        return correctPredictions / totalSamples * 100;
    }
};
#endif //FEEDFORWARDNEURALNET_ACCURACY_H
