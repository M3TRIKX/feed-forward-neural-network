//
// Created by Dáša Pawlasová on 21.11.2021.
//

#include "util_functions.h"

std::tuple<float, float, float> getStats(std::vector<float> vec){
    auto min = vec[0];
    auto max = vec[0];
    float sum = 0;
    for (size_t i = 0; i < vec.size(); i++){
        min = std::min(min, vec[i]);
        max = std::max(max, vec[i]);
        sum += vec[i];
    }
    return std::make_tuple(min, max, sum / (float) vec.size());
}

std::string convertToMinSecText(float timeMin){
    size_t minutes = timeMin;
    size_t seconds = (timeMin - minutes) * 60;
    return std::to_string(minutes) + "min " + std::to_string(seconds) + "sec";
}

void printProgressLine(size_t current, size_t max, std::string text){
    auto line = text + '[';
    for (size_t i = 0; i < max; i++){
        if (i > current){
            line += '~';
        } else if (i == current){
            line += '>';
        } else{
            line += '=';
        }
    }
    line += ']';
    std::cout << line << std::endl;
}

void printTestResultsForConfig(size_t firstHidden, size_t secondHidden, size_t batchSize, float eta, float lambda,
                               float decayRate, size_t stepsDecay, float minEta, size_t earlyStopping, Stats stats, float runTime){
    std::cout << "Accuracy: " << stats.accuracy << "% Cross-entropy: " << stats.crossEntropy << " Run-time: " << convertToMinSecText(runTime);
    std::cout << " Topology: 784x" << firstHidden << "x" << secondHidden << "x10 Batch size: " << batchSize << " Eta: " << eta << " Lambda: " << lambda;
    std::cout << " Decay rate: " << decayRate << " Decay steps: " << stepsDecay << " Min eta: " << minEta << " Early stopping: " << earlyStopping <<  std::endl;
}

