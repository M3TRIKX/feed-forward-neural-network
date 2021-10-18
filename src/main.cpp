#include <iostream>
#include "csv/csv_reader.h"

int main() {
    std::cout << "Hello, World!" << std::endl;

    CsvReader<784> reader("./data/fashion_mnist_train_vectors.csv");
    std::cout << reader.getNumOfLines() << std::endl;

    return 0;
}
