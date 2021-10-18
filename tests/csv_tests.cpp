#include <stdio.h>
#include "gtest/gtest.h"
#include "../src/csv/csv_writer.h"
#include "../src/csv/csv_reader.h"

namespace {
    TEST(CSV, TrainVectorsRW) {
        CsvReader reader("./data/fashion_mnist_train_vectors.csv", 784);
        CsvWriter<uint8_t>::writeCsv("out.csv", reader.getDataMatrix());
        CsvReader reader2("out.csv", 784);

        remove("out.csv");

        EXPECT_EQ(reader.getDataMatrix(), reader2.getDataMatrix());
    }

    TEST(CSV, TrainLabelsRW) {
        CsvReader reader("./data/fashion_mnist_train_labels.csv", 1);
        CsvWriter<uint8_t>::writeCsv("out.csv", reader.getDataMatrix());
        CsvReader reader2("out.csv", 1);

        remove("out.csv");

        EXPECT_EQ(reader.getDataMatrix(), reader2.getDataMatrix());
    }
}
