#include <chrono>
#include <cassert>
#include "network.hpp"
#include "../statistics/weights_info.hpp"

void Network::updateWeights(size_t batchSize, float eta) {
    optimizer->update(weightDeltas, deltaBiases, batchSize, eta);
}

Matrix<Network::ELEMENT_TYPE> Network::predict(const Matrix<float> &data) {
    auto tmp = data.matmul(weights[0]);
    tmp += biases[0];

    networkConfig.layersConfig[1].activationFunction(tmp);

    for (size_t i = 1; i < weights.size(); ++i) {
        tmp = tmp.matmul(weights[i]);
        tmp += biases[i];

        // i + 1 due to the way we store activation functions.
        networkConfig.layersConfig[i + 1].activationFunction(tmp);
    }

    return tmp;
}

auto Network::forwardPass(const Matrix<ELEMENT_TYPE> &data, const std::vector<unsigned int> &labels,
                          size_t kthThread) {

    parallelActivationDerivResults[kthThread].clear();
    parallelActivationResults[kthThread].clear();

    parallelActivationResults[kthThread].push_back(data);

    auto tmp = data.matmul(weights[0]);
    tmp += biases[0];

    networkConfig.layersConfig[1].activationFunction(tmp);
    parallelActivationResults[kthThread].push_back(tmp);

    if (weights.size() == 1) {
        parallelActivationDerivResults[kthThread].emplace_back();
    } else {
        auto tmpCopy = tmp;
        networkConfig.layersConfig[1].activationDerivFunction(tmpCopy);
        parallelActivationDerivResults[kthThread].push_back(tmpCopy);
    }

    for (size_t i = 1; i < weights.size(); ++i) {
        tmp = tmp.matmul(weights[i]);
        tmp += biases[i];

        // kthThread + 1 due to the way we store activation functions.
        networkConfig.layersConfig[i + 1].activationFunction(tmp);
        parallelActivationResults[kthThread].push_back(tmp);

        if (i == weights.size() - 1) {
            parallelActivationDerivResults[kthThread].emplace_back();
        } else {
            auto tmpCopy = tmp;
            networkConfig.layersConfig[i + 1].activationDerivFunction(tmpCopy);
            parallelActivationDerivResults[kthThread].push_back(tmpCopy);
        }
    }

    return Stats::getStats(tmp, labels);
}

auto Network::forwardBackwardPass(const std::vector<Matrix<ELEMENT_TYPE>> &data,
                                  const std::vector<std::vector<unsigned int>> &labels) {
    float acc = 0;
    float ce = 0;
    size_t batchSize = 0;
    for (auto &subBatch: data) batchSize += subBatch.getNumRows();

    size_t currentStartRows = 0;
    std::vector<size_t> startRows;
    for (auto &d: data) {
        startRows.push_back(currentStartRows);
        currentStartRows += d.getNumRows();
    }

    deltaBiases.clear();

    for (auto &weight: weights) {
        deltaBiases.emplace_back(weight.getNumCols());

        if (weight.getNumCols() == 0) {
            throw std::exception();
        }
    }

#pragma omp parallel for default(none) shared(weightDeltas)
    for (size_t i = 0; i < weightDeltas.size(); ++i) {
        weightDeltas[i].reset();
        std::fill(deltaBiases[i].begin(), deltaBiases[i].end(), 0);
    }

#pragma omp parallel for default(none) shared(acc, ce, data, labels, startRows, parallelActivationResults, parallelActivationDerivResults, deltaBiases, networkConfig, weightsTransposed)
    for (size_t k = 0; k < NUM_NET_THREADS; ++k) {
        if (data.size() - 1 < k)
            continue;

        auto stats = forwardPass(data[k], labels[k], k);

        // Do backprop and then aggregate values into deltaWeights and deltaBiases
        const auto &lastLayerConf = networkConfig.layersConfig[networkConfig.layersConfig.size() - 1];
        if (lastLayerConf.activationFunctionType != ActivationFunction::SoftMax) {
            throw WrongOutputActivationFunction();
        }

        size_t numLayers = networkConfig.layersConfig.size();

        auto lastLayerDelta = CrossentropyFunction::costDelta(parallelActivationResults[k][numLayers - 1],
                                                              labels[k]);
        auto *lastDelta = &lastLayerDelta;
        auto wDelta = parallelActivationResults[k][numLayers - 2].transpose().matmul(lastLayerDelta);

#pragma omp critical
        {
            acc += stats.accuracy;
            ce += stats.crossEntropy;
            weightDeltas[numLayers - 2] += wDelta;
        };

        for (int i = static_cast<int>(numLayers) - 2; i > 0; --i) {
            auto matmuls = lastDelta->matmul(weightsTransposed[i]);
            matmuls *= parallelActivationDerivResults[k][i - 1];
            lastLayerDelta = matmuls;
            lastDelta = &lastLayerDelta;
            wDelta = parallelActivationResults[k][i - 1].transpose().matmul(matmuls);

#pragma omp critical
            {
                weightDeltas[i - 1] += wDelta;
            };
        }

        for (size_t i = 0; i < numLayers - 1; ++i) {
            std::fill(parallelDeltaBiases[k][i].begin(), parallelDeltaBiases[k][i].end(), 0);
            for (size_t j = 0; j < parallelDeltaWeights[k][i].getNumRows(); ++j) {
#pragma omp simd
                for (size_t l = 0; l < parallelDeltaWeights[k][i].getNumCols(); ++l) {
                    parallelDeltaBiases[k][i][l] += parallelDeltaWeights[k][i].getItem(j, l);
                }
            }

#pragma omp critical
            {
#pragma omp simd
                for (size_t j = 0; j < deltaBiases[i].size(); ++j) {
                    deltaBiases[i][j] += parallelDeltaBiases[k][i][j] / static_cast<float>(data[k].getNumRows());
                }
            };
        }
    }

    return Stats_t{.accuracy=acc / NUM_NET_THREADS, .crossEntropy=ce / NUM_NET_THREADS};
}

auto Network::predictParallel(const std::vector<Matrix<float>> &dataBatches,
                              const std::vector<std::vector<unsigned int>> &labels) {
    float acc = 0;
    float ce = 0;

#pragma omp parallel for default(none) shared(dataBatches, labels, acc, ce)
    for (size_t k = 0; k < NUM_NET_THREADS; ++k) {
        auto &data = dataBatches[k];
        auto tmp = data.matmul(weights[0]);
        tmp += biases[0];

        networkConfig.layersConfig[1].activationFunction(tmp);

        for (size_t i = 1; i < weights.size(); ++i) {
            tmp = tmp.matmul(weights[i]);
            tmp += biases[i];

            // i + 1 due to the way we store activation functions.
            networkConfig.layersConfig[i + 1].activationFunction(tmp);
        }

        auto stats = Stats::getStats(tmp, labels[k]);

#pragma omp critical
        {
            acc += stats.accuracy;
            ce += stats.crossEntropy;
        };
    }

    return Stats_t{.accuracy = acc / static_cast<float>(NUM_NET_THREADS),
            .crossEntropy = ce / static_cast<float>(NUM_NET_THREADS)};
}

void Network::weightDecay(float lambda) {
    if (lambda == 0)
        return;

    float decayCoeff = 1.f - lambda;

#pragma omp parallel for default(none) shared(decayCoeff, weights)
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] *= decayCoeff;
    }
}

void Network::fit(const TrainValSplit_t &trainValSplit, size_t numEpochs, size_t batchSize, float eta, float lambda,
                  uint8_t verboseLevel, LRScheduler *sched, size_t earlyStopping, long maxTimeMs) {
    if (eta < 0) {
        throw NegativeEtaException();
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    auto &train_X = trainValSplit.trainData;
    auto &train_y = trainValSplit.trainLabels;
    auto &validation_X = trainValSplit.validationData;
    auto &validation_y = trainValSplit.validationLabels;

    auto validationBatches_X = DataManager::generateBatches(validation_X,
                                                            (validation_X.getNumRows() + NUM_NET_THREADS - 1) / NUM_NET_THREADS);
    auto validationBatches_y = DataManager::generateVectorBatches(validation_y,
                                                                  (validation_y.size() + NUM_NET_THREADS - 1) / NUM_NET_THREADS);

    // Copy train data
    auto shuffledTrain_X = train_X;
    auto shuffledTrain_y = train_y;

    auto trainBatches_X = DataManager::generateBatches(train_X, batchSize);
    auto trainBatches_y = DataManager::generateVectorBatches(train_y, batchSize);

    float accSum = 0;
    float ceSum = 0;

    size_t numBatches = train_X.getNumRows() / batchSize;
    size_t t = 0;

    float currentBestCE = 10000;
    size_t epochOfBestCE = 0;

    sched->setEta(eta);

    for (size_t i = 0; i < numEpochs; ++i) {
        auto startPrep = std::chrono::high_resolution_clock::now();
        // Reshuffle data
        auto shuffledData = DataManager::randomShuffle(std::move(shuffledTrain_X), std::move(shuffledTrain_y));
        shuffledTrain_X = std::move(shuffledData.data);
        shuffledTrain_y = std::move(shuffledData.vectorLabels);

        // Create new batches after reshuffling the data
        trainBatches_X = DataManager::generateBatches(shuffledTrain_X, batchSize);
        trainBatches_y = DataManager::generateVectorBatches(shuffledTrain_y, batchSize);

        std::vector<std::vector<Matrix<float>>> parallelBatches_X(trainBatches_X.size());
        std::vector<std::vector<std::vector<unsigned int>>> parallelBatches_y(trainBatches_X.size());

#pragma omp parallel default(none) shared(parallelBatches_X, parallelBatches_y, batchSize, trainBatches_X, trainBatches_y)
        {
#pragma omp for nowait
            for (size_t j = 0; j < trainBatches_X.size(); ++j) {
                parallelBatches_X[j] = (
                        DataManager::generateBatches(trainBatches_X[j], (trainBatches_X[j].getNumRows() + NUM_NET_THREADS - 1) / NUM_NET_THREADS));
            }

#pragma omp for
            for (size_t k = 0; k < trainBatches_y.size(); ++k) {
                parallelBatches_y[k] = (
                        DataManager::generateVectorBatches(trainBatches_y[k], (trainBatches_y[k].size() + NUM_NET_THREADS - 1) / NUM_NET_THREADS));
            }
        }

        auto start = std::chrono::high_resolution_clock::now();

        for (size_t j = 0; j < numBatches; ++j) {
            eta = sched->exponential(t);

            auto stats = forwardBackwardPass(parallelBatches_X[j], parallelBatches_y[j]);
            accSum += stats.accuracy;
            ceSum += stats.crossEntropy;

            weightDecay(lambda);
            updateWeights(batchSize, eta);

            t += batchSize;
        }

        auto valStats = predictParallel(validationBatches_X, validationBatches_y);

        if (verboseLevel >= 3) {
            for (const auto &singleWeights: weights) {
                WeightInfo::printWeightStats(singleWeights, true);
            }
        }

        if (verboseLevel >= 1) {
            Stats::printProgressLine(accSum / static_cast<float>(numBatches),
                                            ceSum / static_cast<float>(numBatches),
                                     valStats.accuracy,
                                     valStats.crossEntropy,
                                            i + 1,
                                     numEpochs);

            accSum = 0;
            ceSum = 0;
        }

        if (verboseLevel >= 2) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            std::cout << "Time taken by function: "
                      << duration.count() << " microseconds" << std::endl;
            std::cout << "ETA: " << eta << std::endl;
        }

        if (earlyStopping != 0) {
            if (valStats.crossEntropy < currentBestCE) {
                currentBestCE = valStats.crossEntropy;
                epochOfBestCE = i;
            }
            if (i - epochOfBestCE == earlyStopping) {
                break;
            }
        }

        auto currentTime = std::chrono::high_resolution_clock::now();
        auto currentDuration = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime).count();
        if (maxTimeMs != 0 && currentDuration >= maxTimeMs) {
            if (verboseLevel >= 1) {
                std::cout << "Time exceeded" << std::endl;
            }
            break;
        }
    }
}