//
// Created by dominik on 22. 10. 2021.
//

#ifndef FEEDFORWARDNEURALNET_CONFIG_H
#define FEEDFORWARDNEURALNET_CONFIG_H

#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>
#include <functional>
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

public:
    Config &addLayer(size_t nNeurons, ActivationFunction activationFunction = ActivationFunction::Identity);

private:
    friend class Network;
};


#endif //FEEDFORWARDNEURALNET_CONFIG_H
