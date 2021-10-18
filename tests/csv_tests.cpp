#include <stdio.h>
#include "gtest/gtest.h"
#include "../src/csv/csv_writer.h"
#include "../src/csv/csv_reader.h"

namespace {
    TEST(CSV, TrainVectorsRW) {
        CsvReader<784> reader("./data/fashion_mnist_train_vectors.csv");
        CsvWriter<784>::writeCsv("out.csv", reader.getDataVector());
        CsvReader<784> reader2("out.csv");

        remove("out.csv");

        EXPECT_EQ(reader.getDataVector(), reader2.getDataVector());
    }

    TEST(CSV, TrainLabelsRW) {
        CsvReader<1> reader("./data/fashion_mnist_train_labels.csv");
        CsvWriter<1>::writeCsv("out.csv", reader.getDataVector());
        CsvReader<1> reader2("out.csv");

        remove("out.csv");

        EXPECT_EQ(reader.getDataVector(), reader2.getDataVector());
    }
}
