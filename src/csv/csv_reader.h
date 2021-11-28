//
// Created by Dominik Klement on 12/10/2021.
//

#ifndef FEEDFORWARDNEURALNET_CSV_READER_H
#define FEEDFORWARDNEURALNET_CSV_READER_H

#include "../data_structures/matrix.h"
#include <algorithm>
#include <cstring>
#include <sstream>
#include <vector>
#include <fstream>

class CsvReadError: std::exception {};

template <typename ELEMENT_TYPE>
class CsvReader {
    std::vector<std::vector<ELEMENT_TYPE>> data;
    Matrix<ELEMENT_TYPE> dataMatrix;

public:
    /**
     * Constructor reads and parses a CSV file using memory mapping.
     *
     * @param path - string path of the file we want to read
     */
    explicit CsvReader(const char *path, int numCols) {
        std::ifstream f(path);
        std::string line;
        std::string elem;

        while (f) {
            while (std::getline(f, line)) {
                data.emplace_back(numCols, 0);
                auto &currentRow = data[data.size() - 1];
                size_t currentIndex = 0;

                std::istringstream ss(line);
                while (std::getline(ss, elem, ',')) {
                    currentRow[currentIndex++] = std::stof(elem);
                }
            }
        }

        dataMatrix = Matrix<ELEMENT_TYPE>(std::move(data));
    }

    void normalize() {
        for (size_t i = 0; i < dataMatrix.getNumRows(); ++i) {
            ELEMENT_TYPE maxRowVal = *(std::max_element(dataMatrix.getMatrix()[i].cbegin(),
                                                        dataMatrix.getMatrix()[i].cend()));

            for (size_t j = 0; j < dataMatrix.getNumCols(); ++j) {
                dataMatrix.setItem(i, j, dataMatrix.getItem(i, j) / maxRowVal);
            }
        }
    }

    /**
     * @return 2D matrix
     */
    const auto &getDataMatrix() {
        return dataMatrix;
    }

    auto &&getDataMatrixRvalRef() {
        return std::move(dataMatrix);
    }

    /**
     *
     * @return num of CSV rows
     */
    auto getNumOfRows() const {
        return dataMatrix.getNumRows();
    }
};


#endif //FEEDFORWARDNEURALNET_CSV_READER_H
