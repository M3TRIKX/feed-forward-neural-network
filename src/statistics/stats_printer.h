//
// Created by Dáša Pawlasová on 20.10.2021.
//

#ifndef FEEDFORWARDNEURALNET_STATS_PRINTER_H
#define FEEDFORWARDNEURALNET_STATS_PRINTER_H

#include "./accuracy.h"
#include "./crossentropy.h"
#include "./argmax.h"

struct Stats {
    float accuracy = 0;
    float crossEntropy = 0;
};

/**
 * Class containing different ways of printing and retrieving stats
 */
class StatsPrinter {
    static const int DECIMAL_PLACES_IN_PRINT = 4;

public:

    /**
     * Returns stats of given matrix
     * @param predicted - matrix predicted by NN
     * @param expected - expected labels
     * @return stats as a hashmap
     */
    static Stats getStats(const Matrix<float> &predicted, const std::vector<int> &expected) {
        float accuracy = AccuracyFunction::accuracy(ArgmaxFunction::argmax(predicted), expected);
        float crossentropy = CrossentropyFunction::crossentropy(predicted, expected);
        return { .accuracy=accuracy, .crossEntropy=crossentropy };
//        return std::map<std::string, float> {{"accuracy", accuracy}, {"crossentropy", crossentropy}};
    }

    /**
     * Prints line of current state of NN, containing epoch, training stats and validation stats
     * @param trainOutput - NN output on training dataset
     * @param trainExpectedLabels - Expected labels of training dataset
     * @param valOutput - NN output on validation dataset
     * @param valExpectedLabels - Expected labels of validation dataset
     * @param epoch - current epoch
     * @param totalEpochs - total amount of epochs
     */
    static void printProgressLine(const Matrix<float> &trainOutput, const std::vector<int> &trainExpectedLabels,
                                  const Matrix<float> &valOutput, const std::vector<int> &valExpectedLabels, int epoch, int totalEpochs) {
        auto trainStats = getStats(trainOutput, trainExpectedLabels);
        auto valStats = getStats(valOutput, valExpectedLabels);
        std::cout << "Epoch: " << epoch << "/" << totalEpochs;
        std::cout << "    Accuracy: " << std::fixed << std::setprecision(DECIMAL_PLACES_IN_PRINT) << trainStats.accuracy;
        std::cout << "%    Loss: " << std::fixed << std::setprecision(DECIMAL_PLACES_IN_PRINT) << trainStats.crossEntropy;
        std::cout << "    ValAccuracy: " << std::fixed << std::setprecision(DECIMAL_PLACES_IN_PRINT) << valStats.accuracy;
        std::cout << "%    ValLoss: " << std::fixed << std::setprecision(DECIMAL_PLACES_IN_PRINT) << valStats.crossEntropy << std::endl;
    }
};
#endif //FEEDFORWARDNEURALNET_STATS_PRINTER_H
