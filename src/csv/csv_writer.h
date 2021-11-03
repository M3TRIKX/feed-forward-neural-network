//
// Created by Dominik Klement on 16/10/2021.
//

#ifndef FEEDFORWARDNEURALNET_CSV_WRITER_H
#define FEEDFORWARDNEURALNET_CSV_WRITER_H

#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>
#include "../data_structures/matrix.h"

class CsvWriteError: std::exception {};

template<typename T>
class CsvWriter {
public:
    // ToDo: Error handling...
    static void writeCsv(const char *path, const Matrix<T> &matrix) {
        // Create CSV content in-memory
        std::string csvContent;

        for (unsigned i = 0; i < matrix.getNumRows(); ++i) {
            csvContent += std::to_string(matrix.getItem(i, 0));

            for (unsigned j = 1; j < matrix.getNumCols(); ++j) {
                csvContent.push_back(',');
                csvContent += std::to_string(matrix.getItem(i, j));
            }

            csvContent += '\n';
        }

        writeToFile(path, csvContent);
    }

    static void writeCsv(const char *path, const std::vector<T> &data) {
        // Create CSV content in-memory
        std::string csvContent;

        for (unsigned i = 0; i < data.size(); ++i) {
            csvContent += std::to_string(data[i]);
            csvContent.push_back('\n');
        }

        writeToFile(path, csvContent);
    }

private:
    static void writeToFile(const char *path, const std::string &content) {
        std::ofstream outputFile(path);
        outputFile << content;
        outputFile.close();
    }
};

#endif //FEEDFORWARDNEURALNET_CSV_WRITER_H
