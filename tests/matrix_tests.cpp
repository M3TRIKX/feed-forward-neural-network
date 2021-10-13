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
}
