//
// Created by dominik on 22. 10. 2021.
//

#ifndef FEEDFORWARDNEURALNET_CONFIG_H
#define FEEDFORWARDNEURALNET_CONFIG_H

#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>
#include "../activation_functions/functions_enum.h"
#include "../activation_functions/fast_sigmoid.h"
#include "../activation_functions/relu.h"
#include "../activation_functions/sigmoid.h"
#include "../activation_functions/softmax.h"

struct LayerConfig {
    using ELEMENT_TYPE = float;
    using ActivationFunction_t = std::function<void(Matrix<ELEMENT_TYPE> &)>;

    size_t numNeurons = 1;
    ActivationFunction_t activationFunction;
    LayerConfig(size_t numNeurons, ActivationFunction_t activationFunction = {}):
        numNeurons(numNeurons), activationFunction(std::move(activationFunction)) {}
};

class WrongActivationFunction : public std::exception {};

class Config {
    using ELEMENT_TYPE = float;
    using ActivationFunction_t = std::function<void(Matrix<ELEMENT_TYPE> &)>;

    std::vector<LayerConfig> layersConfig;
    size_t batchSize = 1;

public:
    auto &addLayer(size_t numNeurons, ActivationFunction activationFunction = ActivationFunction::Identity) {
        ActivationFunction_t fn{};

        switch (activationFunction) {
            case Identity:
                // Leave fn empty
                break;

            case ReLU:
                fn = ReLU::normal;
                break;

            case Sigmoid:
                fn = Sigmoid::normal;
                break;

            case FastSigmoid:
                fn = FastSigmoid::normal;
                break;

            case SoftMax:
                fn = SoftMax::normal;
                break;

            default:
                throw WrongActivationFunction();
        }

        layersConfig.emplace_back(numNeurons, fn);
        return *this;
    }

    auto &setBatchSize(size_t size) {
        batchSize = size;
        return *this;
    }

private:
    friend class Network;
};


#endif //FEEDFORWARDNEURALNET_CONFIG_H
