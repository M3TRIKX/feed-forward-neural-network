#include <stdio.h>
#include "gtest/gtest.h"
#include "../src/csv/csv_writer.h"
#include "../src/csv/csv_reader.h"

namespace {
    TEST(CSV, TrainVectorsRW) {
        CsvReader<uint8_t> reader("./data/fashion_mnist_train_vectors.csv", 784);
        CsvWriter<uint8_t>::writeCsv("out.csv", reader.getDataMatrix());
        CsvReader<uint8_t> reader2("out.csv", 784);

        remove("out.csv");

        EXPECT_EQ(reader.getDataMatrix(), reader2.getDataMatrix());
    }

    TEST(CSV, TrainLabelsRW) {
        CsvReader<uint8_t> reader("./data/fashion_mnist_train_labels.csv", 1);
        CsvWriter<uint8_t>::writeCsv("out.csv", reader.getDataMatrix());
        CsvReader<uint8_t> reader2("out.csv", 1);

        remove("out.csv");

        EXPECT_EQ(reader.getDataMatrix(), reader2.getDataMatrix());
    }
}
