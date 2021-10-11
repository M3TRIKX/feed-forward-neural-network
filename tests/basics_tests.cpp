#include "gtest/gtest.h"
#include "../src/basics.h"

namespace {
    // Init everything - read https://google.github.io/googletest/primer.html
    class NumbersTests : public ::testing::Test {
    protected:
        Numbers numbers1;
        Numbers numbers2;
        Numbers numbers3;

        NumbersTests(): numbers1(1, 2), numbers2(2, 3), numbers3(3, 4) {
            // Constructor
        }

        ~NumbersTests() override {
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

    // Test add function
    TEST(AddTest, Basic) {
        EXPECT_EQ(add(1, 2), 3);
        EXPECT_EQ(add(2, 3), 5);
    }

    TEST(AddTest, Advanced) {
        EXPECT_EQ(add(1, 2), 3);
    }

    // Test Numbers class
    TEST_F(NumbersTests, Add) {
        EXPECT_EQ(numbers1.add(), 3);
        EXPECT_EQ(numbers2.add(), 5);
        EXPECT_EQ(numbers3.add(), 7);
    }

    TEST_F(NumbersTests, Sub) {
        EXPECT_EQ(numbers1.sub(), -1);
        EXPECT_EQ(numbers2.sub(), -1);
        EXPECT_EQ(numbers3.sub(), -1);
    }

    TEST_F(NumbersTests, Mul) {
        EXPECT_EQ(numbers1.mul(), 2);
        EXPECT_EQ(numbers2.mul(), 6);
        EXPECT_EQ(numbers3.mul(), 12);
    }
}
