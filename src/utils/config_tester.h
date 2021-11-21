//
// Created by Dáša Pawlasová on 21.11.2021.
//

#ifndef FEEDFORWARDNEURALNET_CONFIG_TESTER_H
#define FEEDFORWARDNEURALNET_CONFIG_TESTER_H

#include "../network/config.h"
#include "../network/network.h"
#include "../activation_functions/functions_enum.h"
#include "../csv/csv_reader.h"
#include "../data_manager/data_manager.h"
#include "../optimizers/adam.h"
#include "../utils/util_functions.h"
#include <iostream>
#include <chrono>
#include <omp.h>
#include <map>

/**
 * Neural network configuration and hyper-parameters
 */
struct Configuration {
    size_t firstLayerSize;
    size_t secondLayerSize;
    size_t batchSize;
    float eta;
    float lambda;
    float decayRate;
    size_t stepsDecay;
    float minEta;
    size_t earlyStopping;
    long timeMsLimit;
    size_t maxEpochs;

    /**
     * Get values as a tuple
     * @return configuration values
     */
    std::tuple<size_t, size_t, size_t, float, float, float, size_t, float, size_t, long, size_t> getConfigTuple() {
        return std::make_tuple(firstLayerSize, secondLayerSize, batchSize, eta, lambda, decayRate, stepsDecay, minEta,
                               earlyStopping, timeMsLimit, maxEpochs);
    }
};

/**
 * Configuration tester
 */
class ConfigTester{
    TrainValSplit_t &data;
    CsvReader<float> &testVectors;
    CsvReader<unsigned int> &testLabels;

    public:
        ConfigTester(TrainValSplit_t &data, CsvReader<float> &testVectors, CsvReader<unsigned int> &testLabels) : data(data), testVectors(testVectors), testLabels(testLabels){}

        /**
         * Test given configurations
         * @param configurations - configurations to test
         * @param runsPerConfig - how many times should each configuration run
         * @param verbose - verbose level
         * @param printProgress - true if you want to print progress
         */
        void testConfigs(std::vector<Configuration> &configurations, size_t runsPerConfig, size_t verbose, bool printProgress);

        /**
         * Print info about configuration
         * @param configuration - configuration to print information about
         * @param runs - how many times the configuration should be ran
         */
        void printConfigInfo(Configuration configuration, size_t runs);

        /**
         * Prints result statistics of the configuration
         * @param accuracies - resulting accuracies of runs
         * @param losses - resulting loss values of runs
         * @param times - run-times of runs
         */
        void printConfigResults(std::vector<float> &accuracies, std::vector<float> &losses, std::vector<float> &times);

        /**
         * Prints results sorted by the best accuracy with topology information
         * @param results - accuracy of each configuration
         */
        void printFinalResults(std::multimap<float, std::string, std::greater<int>> &results);

        /**
         * Runs parallel configuration test - each configuration once. Ideal for global parameter search
         * @param configurations - configurations to test
         * @param verbose - verbose level
         * @param threads - on how many threads should the test run
         */
        void runParallelConfigTest(std::vector<Configuration> &configurations, size_t verbose, size_t threads);
};

#endif //FEEDFORWARDNEURALNET_CONFIG_TESTER_H
