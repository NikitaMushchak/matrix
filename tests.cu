#include "gtest/gtest.h"
//#include "Matrix.cpp"
#include <omp.h>
#include <chrono>
#include "cudatools.cu"

#define NUM_THREADS 4
#define NUM_START 1
#define NUM_END 10

TEST(SimpleTest, aa) {
    ASSERT_FALSE(false);
}
TEST(SimpleCUDA_test, zero) {
    omp_set_num_threads(4);
#pragma omp parallel
    {
        int id = omp_get_thread_num();
        printf("Hello (%d)", id);
        printf("World (%d)\n", id);
    }
}

TEST(OpenMPTest, sipmletest) {
    int i, nRet = 0, nSum = 0, nStart = NUM_START, nEnd = NUM_END;
    int nThreads = 0, nTmp = nStart + nEnd;
    unsigned uTmp = (unsigned((abs(nStart - nEnd) + 1)) *
        unsigned(abs(nTmp))) / 2;
    int nSumCalc = uTmp;

    if (nTmp < 0)
        nSumCalc = -nSumCalc;

    omp_set_num_threads(NUM_THREADS);

#pragma omp parallel default(none) private(i) shared(nSum, nThreads, nStart, nEnd)
    {
#pragma omp master
        nThreads = omp_get_num_threads();

#pragma omp for
        for (i = nStart; i <= nEnd; ++i) {
#pragma omp atomic
            nSum += i;
        }
    }

    if (nThreads == NUM_THREADS) {
        printf_s("%d OpenMP threads were used.\n", NUM_THREADS);
        nRet = 0;
    }
    else {
        printf_s("Expected %d OpenMP threads, but %d were used.\n",
            NUM_THREADS, nThreads);
        nRet = 1;
    }

    if (nSum != nSumCalc) {
        printf_s("The sum of %d through %d should be %d, "
            "but %d was reported!\n",
            NUM_START, NUM_END, nSumCalc, nSum);
        nRet = 1;
    }
    else
        printf_s("The sum of %d through %d is %d\n",
            NUM_START, NUM_END, nSum);
}

TEST(SimpleompTest, f) {
#pragma omp parallel num_threads(2)
    {
#pragma omp single
        // Only a single thread can read the input.
        printf_s("read input\n");

        // Multiple threads in the team compute the results.
        printf_s("compute results\n");

#pragma omp single
        // Only a single thread can write the output.
        printf_s("write output\n");
    }
}
TEST(FastMatrixTEST, first) {
    FastMatrix<int> a(7, 8);
    a.fillValue(10);
    a.showMatrix();
    ASSERT_EQ(10, a(3, 4));
}
TEST(SimpleMultiplyTest, first) {
    FastMatrix<int> a(5, 2);
    FastMatrix<int> b(2, 10);
    a.fillValue(10);
    b.fillValue(2);
    FastMatrix<int>c(5, 10);
    c.CUDAMultiply(a, b);
    FastMatrix<int> res(5, 10);
    res.fillValue(40);
    ASSERT_EQ(res, c);
}
int main(int arg_v, char** arg_c) {
    /*std::cout << "Global started\n";
    testglob();
    std::cout << "Global finished!\n";
    std::cout << "Shared started\n";
    testshared();
    std::cout << "Shared finished!\n";*/
    testing::InitGoogleTest(&arg_v, arg_c);
    return RUN_ALL_TESTS();
}