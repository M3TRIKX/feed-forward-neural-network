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
    ActivationFunction_t activationDerivFunction;
    ActivationFunction activationFunctionType;
    LayerConfig(size_t numNeurons,
                ActivationFunction fnType,
                ActivationFunction_t fn = {},
                ActivationFunction_t fnDeriv = {}):
            numNeurons(numNeurons), activationFunction(std::move(fn)),
            activationDerivFunction(fnDeriv), activationFunctionType(fnType) {}
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
        ActivationFunction_t fnDeriv{};

        switch (activationFunction) {
            case Identity:
                // Leave fn empty
                break;

            case ReLU:
                fn = ReLU::normal;
                fnDeriv = ReLU::derivative;
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

        layersConfig.emplace_back(numNeurons, activationFunction, fn, fnDeriv);
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
