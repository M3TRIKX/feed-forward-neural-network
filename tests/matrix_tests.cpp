#include "gtest/gtest.h"
#include "../src/data_structures/matrix.h"

namespace {
    TEST(MatrixTests, Properties2) {
        const int numRows = 24;
        const int numCols = 42;
        Matrix<int> m(numRows, numCols);

        EXPECT_EQ(m.getNumRows(), numRows);
        EXPECT_EQ(m.getNumCols(), numCols);
    }

    TEST(MatrixTests, BasicMatmul) {
        std::vector<std::vector<float>> m1 = {
                {1, 2, 3},
                {4, 5, 6},
        };
        std::vector<std::vector<float>> m2 = {
                {9, 8, 7},
                {6, 5, 4},
                {3, 2, 1},
        };
        std::vector<std::vector<float>> mres = {
                {30, 24, 18},
                {84, 69, 54},
        };

        Matrix<float> mat1(std::move(m1));
        Matrix<float> mat2(std::move(m2));
        Matrix<float> matRes(std::move(mres));

        EXPECT_EQ(mat1.matmul(mat2).getMatrix(), matRes.getMatrix());
    }
}
