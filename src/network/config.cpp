//
// Created by dominik on 22. 10. 2021.
//

#include "config.h"

Config &Config::addLayer(size_t numNeurons, ActivationFunction activationFunction) {
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
//            fnDeriv = Sigmoid::derivative;
            break;

        case FastSigmoid:
            fn = FastSigmoid::normal;
            fnDeriv = FastSigmoid::derivative;
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
