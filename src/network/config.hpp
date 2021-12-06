#ifndef FEEDFORWARDNEURALNET_CONFIG_H
#define FEEDFORWARDNEURALNET_CONFIG_H

#include "../activation_functions/functions_enum.hpp"
#include "../activation_functions/fast_sigmoid.hpp"
#include "../activation_functions/relu.hpp"
#include "../activation_functions/sigmoid.hpp"
#include "../activation_functions/softmax.hpp"
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>
#include <functional>

/**
 * Layer configuration
 */
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
                ActivationFunction_t fnDeriv = {}) :
            numNeurons(numNeurons), activationFunction(std::move(fn)),
            activationDerivFunction(fnDeriv), activationFunctionType(fnType) {}
};

class WrongActivationFunction : public std::exception {
};

/**
 * Network configuration
 */
class Config {
    using ELEMENT_TYPE = float;
    using ActivationFunction_t = std::function<void(Matrix<ELEMENT_TYPE> &)>;

    std::vector<LayerConfig> layersConfig;

public:
    /**
     * Adds layer to network
     * @param nNeurons - number of neurons in layer
     * @param activationFunction - activation function of layer
     * @return configuration
     */
    Config &addLayer(size_t nNeurons, ActivationFunction activationFunction = ActivationFunction::Identity);

private:
    friend class Network;
};


#endif //FEEDFORWARDNEURALNET_CONFIG_H
