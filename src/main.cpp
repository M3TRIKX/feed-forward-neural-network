#include <iostream>
#include "csv/csv_reader.hpp"
#include "data_manager/data_manager.hpp"
#include "network/config.hpp"
#include "optimizers/adam.hpp"
#include "schedulers/lr_sheduler.hpp"
#include "network/network.hpp"
#include "csv/csv_writer.hpp"

int main() {
    CsvReader<float> trainVectors("./data/fashion_mnist_train_vectors.csv", 784);
    CsvReader<unsigned int> trainLabels("./data/fashion_mnist_train_labels.csv", 1);

    CsvReader<float> testVectors("./data/fashion_mnist_test_vectors.csv", 784);
    CsvReader<unsigned int> testLabels("./data/fashion_mnist_test_labels.csv", 1);

    trainVectors.normalize();
    testVectors.normalize();

    auto trainValSplit = DataManager::trainValidateSplit(trainVectors.getDataMatrixRvalRef(),
                                                         trainLabels.getDataMatrix().getMatrixCol(0), 9.f / 10);

    Config config;
    config.addLayer(784)
            .addLayer(900, ActivationFunction::ReLU)
            .addLayer(450, ActivationFunction::ReLU)
            .addLayer(10, ActivationFunction::SoftMax);

    AdamOptimizer adam;
    Network network(config, &adam);

    LRScheduler sched(0.001, 0.85, 30000);
    network.fit(trainValSplit, 30, 64, 0.1, 1e-6, 1, &sched, 5);

    std::cout << "\nTest set: ";
    auto predicted = network.predict(testVectors.getDataMatrix());
    auto testStats = Stats::getStats(predicted, testLabels.getDataMatrix().getMatrixCol(0));
    std::cout << "Accuracy: " << testStats.accuracy << "% Loss: " << testStats.crossEntropy << std::endl;

    CsvWriter<unsigned int>::writeCsv("./actualPredictions", Stats::argmax(predicted));
    return 0;
}
