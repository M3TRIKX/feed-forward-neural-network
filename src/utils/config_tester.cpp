//
// Created by Dáša Pawlasová on 21.11.2021.
//

#include "config_tester.hpp"

void ConfigTester::printConfigInfo(Configuration config, size_t runs) {
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "          Configuration test of " << runs << " runs" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Topology: 784x" << config.firstLayerSize << "x" << config.secondLayerSize << "x10" << std::endl;
    std::cout << "  Eta: " << config.eta << std::endl;
    std::cout << "  Min eta: " << config.minEta << std::endl;
    std::cout << "  Lambda: " << config.lambda << std::endl;
    std::cout << "  Decay rate: " << config.decayRate << std::endl;
    std::cout << "  Decay steps: " << config.stepsDecay << std::endl;
    std::cout << "\nLimits:" << std::endl;
    std::cout << "  Early stopping: " << config.earlyStopping << std::endl;
    std::cout << "  Time limit: " << convertToMinSecText((float) config.timeMsLimit / 60000.0) << std::endl;
    std::cout << "  Max epochs: " << config.maxEpochs << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
}

void ConfigTester::printConfigResults(std::vector<float> &accuracies, std::vector<float> &losses,
                                      std::vector<float> &times) {
    auto[minTime, maxTime, avgTime] = getStats(times);
    auto[minAcc, maxAcc, avgAcc] = getStats(accuracies);
    auto[minLoss, maxLoss, avgLoss] = getStats(losses);
    std::cout << "\n---------------------------------------------------" << std::endl;
    std::cout << "Run-time:\n    Average: " << convertToMinSecText(avgTime) << "\n    Best: "
              << convertToMinSecText(minTime) << "\n    Worst: " << convertToMinSecText(maxTime) << std::endl;
    std::cout << "Accuracy:\n    Average: " << avgAcc << "\n    Best: " << maxAcc << "\n    Worst: " << minAcc
              << std::endl;
    std::cout << "Cross-entropy:\n    Average: " << avgLoss << "\n    Best: " << minLoss << "\n    Worst: " << maxLoss
              << std::endl;
    std::cout << "---------------------------------------------------\n" << std::endl;
}

void ConfigTester::printFinalResults(std::multimap<float, std::string, std::greater<int>> &results) {
    std::cout << "\n---------------------------------------------------" << std::endl;
    std::cout << "                 Final results" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    for (auto const &entry: results) {
        std::cout << "Accuracy: " << entry.first << "% Topology: " << entry.second << std::endl;
    }
    std::cout << "---------------------------------------------------\n" << std::endl;
}

void ConfigTester::testConfigs(std::vector<Configuration> &configurations, size_t runsPerConfig, size_t verbose,
                               bool printProgress) {

    std::multimap<float, std::string, std::greater<int>> sorted_map;

    for (Configuration config: configurations) {
        std::vector<float> accuracies = {};
        std::vector<float> losses = {};
        std::vector<float> times = {};

        printConfigInfo(config, runsPerConfig);

        auto[firstLayerSize, secondLayerSize, batchSize, eta, lambda, decayRate, stepsDecay, minEta, earlyStopping, timeMsLimit, maxEpochs] = config.getConfigTuple();
        std::string configText =
                "784x" + std::to_string(firstLayerSize) + "x" + std::to_string(secondLayerSize) + "x10";

        for (size_t i = 0; i < runsPerConfig; i++) {
            if (printProgress) {
                printProgressLine(i, runsPerConfig, "Testing configuration " + configText + "... ");
            }
            Config config;
            config.addLayer(784)
                    .addLayer(firstLayerSize, ActivationFunction::ReLU)
                    .addLayer(secondLayerSize, ActivationFunction::ReLU)
                    .addLayer(10, ActivationFunction::SoftMax);

            AdamOptimizer adam;
            Network network(config, &adam);
            LRScheduler sched(minEta, decayRate, stepsDecay);

            auto startTime = std::chrono::high_resolution_clock::now();
            network.fit(data, maxEpochs, batchSize, eta, lambda, verbose, &sched, earlyStopping, timeMsLimit);
            auto endTime = std::chrono::high_resolution_clock::now();
            float runTimeMin =
                    std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() / 60000.0;

            auto predicted = network.predict(testVectors.getDataMatrix());
            auto testStats = Stats::getStats(predicted, testLabels.getDataMatrix().getMatrixCol(0));

            times.push_back(runTimeMin);
            losses.push_back(testStats.crossEntropy);
            accuracies.push_back(testStats.accuracy);

            if (printProgress) {
                std::cout << "Accuracy: " << testStats.accuracy << " Crossentropy: " << testStats.crossEntropy
                          << " Run-time: " << convertToMinSecText(runTimeMin) << std::endl;
            }
        }
        auto[minTime, maxTime, avgTime] = getStats(times);
        auto[minAcc, maxAcc, avgAcc] = getStats(accuracies);
        auto[minLoss, maxLoss, avgLoss] = getStats(losses);

        sorted_map.insert(std::make_pair(avgAcc, configText));
        printConfigResults(accuracies, losses, times);
    }

    printFinalResults(sorted_map);
}

void
ConfigTester::runParallelConfigTest(std::vector<Configuration> &configurations, size_t verboseLevel, size_t threads) {
#pragma omp parallel num_threads(threads) default(none) shared(data, configurations, verboseLevel, testVectors, testLabels)
    {
#pragma omp for
        for (size_t i = 0; i < configurations.size(); ++i) {
            auto[firstHidden, secondHidden, batchSize, eta, lambda, decayRate, stepsDecay, minEta, earlyStopping, timeMsLimit, maxEpochs] = configurations[i];
            Config config;
            config.addLayer(784)
                    .addLayer(firstHidden, ActivationFunction::ReLU)
                    .addLayer(secondHidden, ActivationFunction::ReLU)
                    .addLayer(10, ActivationFunction::SoftMax);

            AdamOptimizer adam;
            Network network(config, &adam);

            LRScheduler sched(minEta, decayRate, stepsDecay);
            auto startTime = std::chrono::high_resolution_clock::now();
            network.fit(data, maxEpochs, batchSize, eta, lambda, verboseLevel,
                        &sched, earlyStopping, timeMsLimit);
            auto endTime = std::chrono::high_resolution_clock::now();
            auto runTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() / 60000.0;

            auto predicted = network.predict(testVectors.getDataMatrix());
            auto testStats = Stats::getStats(predicted,
                                             testLabels.getDataMatrix().getMatrixCol(
                                                            0));
            printTestResultsForConfig(firstHidden, secondHidden, batchSize, eta, lambda, decayRate,
                                      stepsDecay, minEta, earlyStopping, testStats, runTime);
        }
    }
}
