//
// Created by Dáša Pawlasová on 21.11.2021.
//

#ifndef FEEDFORWARDNEURALNET_UTIL_FUNCTIONS_H
#define FEEDFORWARDNEURALNET_UTIL_FUNCTIONS_H

#include "../statistics/stats_printer.h"
#include <iostream>

/**
 * Calculates minimum, maximum and average of values in given vector
 * @param vec - vector of values
 * @return (minimum, maximum, average)
 */
std::tuple<float, float, float> getStats(std::vector<float> vec);

/**
 * Converts number of minutes to minutes and seconds and returns formatted test
 * @param timeMin - time in minutes
 * @return formatted string
 */
std::string convertToMinSecText(float timeMin);

/**
 * Prints progress line of test configuration
 * @param current - current run
 * @param max - total runs
 * @param text - text to print before the progress line
 */
void printProgressLine(size_t current, size_t max, std::string text);

/**
 * Prints formatted test results
 * @param firstHidden - number of neurons in first hidden layer
 * @param secondHidden - number of neurons in second hidden layer
 * @param batchSize - batch size
 * @param eta - learning rate
 * @param lambda - regularization rate
 * @param decayRate - decay rate
 * @param stepsDecay - decay steps
 * @param minEta - minimum learning rate
 * @param earlyStopping - early stopping
 * @param stats - results on test set
 * @param runTime - time the configuration has run
 */
void printTestResultsForConfig(size_t firstHidden, size_t secondHidden, size_t batchSize, float eta, float lambda,
                               float decayRate, size_t stepsDecay, float minEta, size_t earlyStopping, Stats stats, float runTime);



#endif //FEEDFORWARDNEURALNET_UTIL_FUNCTIONS_H
