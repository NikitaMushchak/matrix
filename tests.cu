#include "gtest/gtest.h"
#include "Mtrx_Mul_Sh.cu"
#include <omp.h>



TEST(SimpleTest, aa) {
    ASSERT_FALSE(false);
}
TEST(SimpleCUDA_test, zero) {
    //test1(arg_v, arg_c);
}

TEST(OpenMPTest, sipmletest) {

}
TEST(MatrixclassTest, zero) {
    Matrix<int> mat(1, 10);
    mat.fillRand(1, 10);
    mat.showMatrix();
}
TEST(simpleoutofrangeTest, outofrange) {
    Matrix<int> a(10, 10);
    a.fillValue(10);
    try {
        std::cout << "a[100] = " << a[100] << '\n';
        FAIL() << "Expected std::out_of_range";
    }
    catch (std::out_of_range const& err) {
        EXPECT_EQ(err.what(), std::string("num is out of range!"));
    }
    catch (...) {
        FAIL() << "Expected std::out_of_range";
    }

}

TEST(simpleSummTest, first) {
    Matrix<int> a(5, 5);
    Matrix<int> b(5, 5);
    a.fillValue(10);
    b.fillValue(2);
    Matrix<int> c(5, 5);
    c.simpleSumm(a,b);
    Matrix<int> res(5, 5);
    res.fillValue(12);
    ASSERT_EQ(res, c);
}
TEST(simpleMultTest, first) {
    Matrix<int> a(2, 2);
    Matrix<int> b(2, 2);
    a.fillValue(10);
    b.fillValue(2);
    Matrix<int>c(2, 2);
    c.simpleMult(a, b);
    Matrix<int> res(2, 2);
    res.fillValue(40);
    ASSERT_EQ(res, c);
}

TEST(simpleMultTest, second) {
    Matrix<int> a(5, 2);
    Matrix<int> b(2, 10);
    a.fillValue(10);
    b.fillValue(2);
    Matrix<int>c(5, 10);
    c.simpleMult(a, b);
    Matrix<int> res(5, 10);
    res.fillValue(40);
    ASSERT_EQ(res, c);
}
int main(int arg_v, char** arg_c) {
    //test1(arg_v, arg_c);
    testing::InitGoogleTest(&arg_v, arg_c);
    return RUN_ALL_TESTS();
}