//
// Created by Dominik Klement on 09/11/2021.
//

#include "lr_sheduler.hpp"

float LRScheduler::exponential(unsigned int t) {
    if (currentEta == minEta) {
        return currentEta;
    }

    if (t / stepsDecay > numSteps) {
        ++numSteps;
        currentEta *= decayRate;
        currentEta = std::max(currentEta, minEta);
    }

    return currentEta;
}
