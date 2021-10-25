//
// Created by dominik on 22. 10. 2021.
//

#ifndef FEEDFORWARDNEURALNET_NETWORK_H
#define FEEDFORWARDNEURALNET_NETWORK_H

#include <vector>
#include "../data_structures/matrix.h"
#include "config.h"
#include "../statistics/stats_printer.h"

class WrongInputDataDimension : public std::exception {};

class Network {
    using ELEMENT_TYPE = float;

    const Config &networkConfig;
    std::vector<Matrix<ELEMENT_TYPE>> weights;

public:
    Network(const Config &config): networkConfig(config) {
        // We are initializing weights between each two layers.
        // weight[k][i][j] corresponds to the weight between neuron ith neuron in layer k
        // and jth neuron in layer k+1.
        for (size_t i = 0; i < config.layersConfig.size() - 1; ++i) {
            const auto &layer = config.layersConfig[i];
            const auto &nextLayer = config.layersConfig[i + 1];
            weights.push_back(Matrix<float>::generateRandomMatrix(layer.numNeurons, nextLayer.numNeurons, -0.3, 0.3));
        }
    }

    void forwardPass(const Matrix<ELEMENT_TYPE> &data) {
        auto tmp = data.matmul(weights[0], 10);
        networkConfig.layersConfig[1].activationFunction(tmp);

        for (size_t i = 1; i < weights.size(); ++i) {
            tmp = tmp.matmul(weights[i]);
            // i + 1 due to the way we store activation functions.
            networkConfig.layersConfig[i + 1].activationFunction(tmp);
        }

        tmp.printMatrix();
        auto stats = StatsPrinter::getStats(tmp, {0,1,2,2,3,2,8,6,5,0});
        std::cout << "ACC: " << stats.accuracy << " CE: " << stats.crossEntropy << std::endl;
    }
};


#endif //FEEDFORWARDNEURALNET_NETWORK_H
