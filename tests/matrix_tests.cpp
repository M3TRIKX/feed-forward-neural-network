#include "gtest/gtest.h"
#include "../src/data_structures/matrix.h"

namespace {
    class MatrixTests : public ::testing::Test {
    protected:
        Matrix<double> m1;

        MatrixTests(): m1(6, 5) {
            // Constructor
        }

        ~MatrixTests() override {
            // Destructor
        }

        void SetUp() override {
            // Code here will be called immediately after the constructor (right
            // before each test).
        }

        void TearDown() override {
            // Code here will be called immediately after each test (right
            // before the destructor).
        }
    };

    // Using Matrix instance defined inside the MatrixTests class.
    TEST_F(MatrixTests, Properties) {
        EXPECT_EQ(m1.getNumRows(), 6);
        EXPECT_EQ(m1.getNumCols(), 5);
    }

    // Another approach is to instantiate Matrix inside the test.
    TEST(MatrixTests2, Properties) {
        const int numRows = 24;
        const int numCols = 42;
        Matrix<int> m(numRows, numCols);

        EXPECT_EQ(m.getNumRows(), numRows);
        EXPECT_EQ(m.getNumCols(), numCols);
    }

    TEST(MatrixMul, Basic) {
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

        EXPECT_EQ(mat1.slowMatmul(mat2).getMatrix(), matRes.getMatrix());
    }
}
