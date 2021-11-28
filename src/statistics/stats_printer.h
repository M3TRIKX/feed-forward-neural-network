//
// Created by Dáša Pawlasová on 20.10.2021.
//

#ifndef FEEDFORWARDNEURALNET_STATS_PRINTER_H
#define FEEDFORWARDNEURALNET_STATS_PRINTER_H

#include "./accuracy.h"
#include "./crossentropy.h"

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
     * @return stats as a map
     */
    static Stats getStats(const Matrix<float> &predicted, const std::vector<unsigned int> &expected) {
        float accuracy = AccuracyFunction::accuracy(argmax(predicted), expected);
        float crossentropy = CrossentropyFunction::crossentropy(predicted, expected);
        return {.accuracy=accuracy, .crossEntropy=crossentropy};
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
    static void
    printProgressLine(const Matrix<float> &trainOutput, const std::vector<unsigned int> &trainExpectedLabels,
                      const Matrix<float> &valOutput, const std::vector<unsigned int> &valExpectedLabels,
                      size_t epoch, size_t totalEpochs) {
        auto trainStats = getStats(trainOutput, trainExpectedLabels);
        auto valStats = getStats(valOutput, valExpectedLabels);
        printProgressLine(trainStats.accuracy, trainStats.crossEntropy, valStats.accuracy, valStats.crossEntropy, epoch,
                          totalEpochs);
    }

    /**
     * Prints progress line from given values
     * @param avgTrainAcc - average train accuracy
     * @param avgTrainCE - average train cross-entropy
     * @param valPredicted - predicted values
     * @param valExpected - expected values
     * @param epoch - current epoch
     * @param totalEpochs - total epochs
     */
    static void printProgressLine(float avgTrainAcc, float avgTrainCE, const Matrix<float> &valPredicted,
                                  const std::vector<unsigned int> &valExpected, size_t epoch, size_t totalEpochs) {
        auto valStats = getStats(valPredicted, valExpected);
        printProgressLine(avgTrainAcc, avgTrainCE, valStats.accuracy, valStats.crossEntropy, epoch, totalEpochs);
    }

    /**
     * Prints progress line from given values
     * @param trainAcc - average train accuracy
     * @param trainCE - average train cross-entropy
     * @param valAcc - validation set accuracy
     * @param valCE - validation set cross-entropy
     * @param epoch - current epoch
     * @param totalEpochs - total epochs
     */
    static void
    printProgressLine(float trainAcc, float trainCE, float valAcc, float valCE, size_t epoch, size_t totalEpochs) {
        std::cout << "Epoch: " << epoch << "/" << totalEpochs;
        std::cout << "    Accuracy: " << std::fixed << std::setprecision(DECIMAL_PLACES_IN_PRINT) << trainAcc;
        std::cout << "%    Loss: " << std::fixed << std::setprecision(DECIMAL_PLACES_IN_PRINT) << trainCE;
        std::cout << "    ValAccuracy: " << std::fixed << std::setprecision(DECIMAL_PLACES_IN_PRINT) << valAcc;
        std::cout << "%    ValLoss: " << std::fixed << std::setprecision(DECIMAL_PLACES_IN_PRINT) << valCE << std::endl;
    }

private:
    /**
     * Function calculating argmax or each row in the matrix
     * @param matrix - matrix to compute argmax on
     * @return - vector of classes
     */
    static std::vector<unsigned int> argmax(const Matrix<float> &matrix) {
        auto classes = std::vector<unsigned int>(matrix.getNumRows());
        for (size_t i = 0; i < matrix.getNumRows(); i++) {
            float currentMax = 0;
            for (size_t j = 0; j < matrix.getNumCols(); j++) {
                if (matrix.getItem(i, j) > currentMax) {
                    currentMax = matrix.getItem(i, j);
                    classes[i] = j;
                }
            }
        }
        return classes;
    }
};

#endif //FEEDFORWARDNEURALNET_STATS_PRINTER_H
