//
// Created by Dominik Klement on 16/10/2021.
//

#ifndef FEEDFORWARDNEURALNET_CSV_WRITER_H
#define FEEDFORWARDNEURALNET_CSV_WRITER_H

#include <iostream>
#include <sstream>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <sys/stat.h>
#include <sys/mman.h>
#include <array>
#include <vector>
#include <cstring>
#include <fstream>

class CsvWriteError: std::exception {};

template<size_t ROW_SIZE>
class CsvWriter {
public:
    // ToDo: Error handling...
    static void writeCsv(const char *path, const std::vector<std::array<uint8_t, ROW_SIZE>> &data) {
        // Create CSV content in-memory
        std::string csvContent;

        for (const auto &row: data) {
            csvContent += std::to_string(row[0]);

            for (unsigned i = 1; i < ROW_SIZE; ++i) {
                csvContent.push_back(',');
                csvContent += std::to_string(row[i]);
            }

            csvContent += '\n';
        }

        std::ofstream outputFile(path);
        outputFile << csvContent;
        outputFile.close();
    }
};

#endif //FEEDFORWARDNEURALNET_CSV_WRITER_H
