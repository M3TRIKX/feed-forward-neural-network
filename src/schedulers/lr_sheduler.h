//
// Created by Dominik Klement on 09/11/2021.
//

#ifndef FEEDFORWARDNEURALNET_LR_SHEDULER_H
#define FEEDFORWARDNEURALNET_LR_SHEDULER_H

#include <algorithm>
#include <cstddef>

/**
 * Class representing LRScheduler
 */
class LRScheduler {
    float currentEta;
    float minEta;
    float decayRate;
    unsigned int stepsDecay;
    unsigned int numSteps = 0;
public:
    /**
     *
     * @param eta        Initial eta
     * @param minEta     Min learning rate at which the decay stops
     * @param decayRate  LR change rate (scheduled eta might differ with each scheduler)
     * @param stepsDecay LR decay occurs after we pass at least stepsDecay examples through the network
     */
    LRScheduler(float eta, float minEta, float decayRate, unsigned int stepsDecay)
            : currentEta(eta), minEta(minEta), decayRate(decayRate), stepsDecay(stepsDecay) {}

    LRScheduler(float minEta, float decayRate, unsigned int stepsDecay)
            : currentEta(1e-3), minEta(minEta), decayRate(decayRate), stepsDecay(stepsDecay) {}

    /**
     * Sets eta
     * @param eta - eta to set
     */
    void setEta(float eta) { currentEta = eta; }

    /**
     * Exponential learning rate decay.
     * @param t Time unit (number of examples passed through the network)
     * @return
     */
    float exponential(unsigned int t);
};


#endif //FEEDFORWARDNEURALNET_LR_SHEDULER_H
