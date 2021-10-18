//
// Created by Dominik Klement on 12/10/2021.
//

#ifndef FEEDFORWARDNEURALNET_CSV_READER_H
#define FEEDFORWARDNEURALNET_CSV_READER_H

#include <iostream>
#include <sstream>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <sys/stat.h>
#include <sys/mman.h>
#include <array>
#include <vector>

class CsvReadError: std::exception {};


template<size_t ROW_SIZE>
class CsvReader {
    int fd = 0;
    bool closed = false;
    std::vector<std::array<uint8_t, ROW_SIZE>> data;

public:
    /**
     * Constructor reads and parses a CSV file using memory mapping.
     *
     * @param path - string path of the file we want to read
     */
    explicit CsvReader(const char *path) {
        // Open the file and obtain the descriptor.
        fd = open(path, O_RDONLY);
        if (fd == -1) {
            throw CsvReadError();
        }

        // Kernel optimizations for read.
//        posix_fadvise(fd, 0, 0, 1);

        struct stat sb{};
        if (fstat(fd, &sb) == -1) {
            throw CsvReadError();
        }

        size_t fileLength = sb.st_size;

        // Contains string representation of the current number (between two commas).
        char number[4] = {};
        // Current position in the array (line).
        int currentArrPos = 0;
        // Current position in the number arr.
        int currentNumPos = 0;

        data.emplace_back();
        auto *currentArr = &data[data.size() - 1];

        // Memory mapped file -> faster reading.
        const char* addr = static_cast<const char*>(mmap(nullptr, fileLength, PROT_READ, MAP_PRIVATE, fd, 0u));
        if (addr == MAP_FAILED) {
            throw CsvReadError();
        }

        // Read the file char by char and save the content into the vector.
        // ToDo: Use pointer arithmetics instead of number array (atoi on number using pointer shifting).
        // Thus having a pointer to the first digit and change , to \0 and on the substring, do atoi (which
        // converts a string up to the \0 char).
        for (size_t i = 0; i < fileLength; ++i) {
            char c = addr[i];
            if (c == '\n') {
                // We didn't obtain enough columns in the current row.
                if (currentArrPos != ROW_SIZE - 1) {
                    throw CsvReadError();
                }

                addNumToArray(number, currentNumPos, currentArrPos, *currentArr);

                // We don't want to add an empty vector due to the last '\n'
                if (i < fileLength - 1) {
                    data.emplace_back();
                    currentArr = &data[data.size() - 1];
                }

                ++currentArrPos;
                currentArrPos = 0;
                currentNumPos = 0;
            }
            else if (c == ',') {
                addNumToArray(number, currentNumPos, currentArrPos, *currentArr);

                ++currentArrPos;
                currentNumPos = 0;
            }
            else if (c >= '0' && c <= '9') {
                if (currentNumPos > 3) {
                    throw CsvReadError();
                }

                number[currentNumPos++] = c;
            }
            else {
                throw CsvReadError();
            }
        }

        // In case the \n is not at the end of the last line
        if (currentNumPos > 0) {
            addNumToArray(number, currentNumPos, currentArrPos, *currentArr);
        }


        close(fd);
        closed = true;
    }

    ~CsvReader() {
        if (!closed) {
            close(fd);
            closed = true;
        }
    }

    /**
     * @return 2D matrix
     */
    const auto &getDataVector() {
        return data;
    }

    /**
     *
     * @return num of CSV lines
     */
    auto getNumOfLines() {
        return data.size();
    }

private:
    /**
     * Adds a number to the array, which represents a row in a CSV file.
     *
     * @param num           - num array
     * @param currentNumPos - current position in num array
     * @param currentArrPos - current array position (single row)
     * @param arr           - array representing current line
     */
    void addNumToArray (char num[4], int currentNumPos, int currentArrPos, std::array<uint8_t, ROW_SIZE> &arr) {
        for (int j = 3; j >= currentNumPos; --j) num[j] = '\0';
        arr[currentArrPos] = atoi(num);
        memset(num, 0, 4);
    }
};


#endif //FEEDFORWARDNEURALNET_CSV_READER_H
