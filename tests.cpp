#include "gtest/gtest.h"
#include "Matrix.cpp"
#include <omp.h>
#include <chrono>


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

TEST(MatrixclassTest, zero) {
    Matrix<int> mat(1, 10);
    mat.fillRand(1, 10);
    
}
TEST(simpleoutofrangeTest, outofrange) {
    Matrix<int> a(10, 10);
    a.fillValue(10);
    try {
        std::cout << "a[100] = " << a[100] << '\n';
        FAIL() << "Expected std::out_of_range";
    }
    catch (std::out_of_range const& err) {
        EXPECT_EQ(err.what(), std::string(" operator [] num is out of range!"));
    }
    catch (...) {
        FAIL() << "Expected std::out_of_range";
    }

}
TEST(FillDataTEST, test) {
    Matrix<int> a(5, 5);
    Matrix<int> b(5, 5);
    int* mass = new int[25];
    for (int i = 0; i < 25; ++i) {
        mass[i] = 25 - i;
        b[i] = 25 - i;
    }

    a.FillData(mass);
    ASSERT_EQ(a, b);
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

TEST(SimpleTransposeTest, first) {
    Matrix<int> a(2, 5);
    a.fillValue(10);
    Matrix<int> b;
    b.Transpose(a);
    ASSERT_EQ(b.GetMaxCols(), 2);
    ASSERT_EQ(b.GetMaxRows(), 5);
    Matrix<int> res(5, 2);
    res.fillValue(10);
    ASSERT_EQ(res, b);
}

TEST(SimpleTransposeTest, second) {
    Matrix<int> a(2, 5);
    a.fillRand(1, 10);
    Matrix<int> b;
    b = a;
    ASSERT_EQ(a, b);
    Matrix<int> c;
    c.Transpose(a);
    Matrix<int> s(5,2);
    for (size_t i = 0; i < s.GetMaxRows(); ++i)
        for (size_t j = 0; j < s.GetMaxCols(); ++j) 
            s[i * s.GetMaxCols() + j] = a[j*a.GetMaxCols() +i];

    ASSERT_EQ(s, c);
}
TEST(SimpleTransposeTest, third) {
    Matrix<int> a(2, 5);
    a.fillRand(1, 10);
    Matrix<int> b;
    b.Transpose(a);

    try {
        b.Transpose(b);
        FAIL() << "Expected std::exception";
    }
    catch (std::exception const& err) {
        EXPECT_EQ(err.what(), std::string("don't transpose the same matrix"));
    }
    catch (...) {
        FAIL() << "Expected std::exception";
    }
}
TEST(TransposeOpenMP, first) {
    Matrix<int> a(2, 5);
    a.fillRand(1, 10);
    Matrix<int> b;
    b = a;
    ASSERT_EQ(a, b);
    Matrix<int> c;
    c.TransposeOPENMP(a);
  
    Matrix<int> s(5, 2);
    for (size_t i = 0; i < s.GetMaxRows(); ++i)
        for (size_t j = 0; j < s.GetMaxCols(); ++j)
            s[i * s.GetMaxCols() + j] = a[j * a.GetMaxCols() + i];

    ASSERT_EQ(s, c);
}
TEST(TransposetimeComp, first) {
    Matrix<int> a(10000, 10000);
    Matrix<int> resa;
    a.fillRand(1, 1000);
    auto start = std::chrono::high_resolution_clock::now();
    resa.Transpose(a);
    auto finish = std::chrono::high_resolution_clock::now();

    Matrix<int> resomp;
    auto starta = std::chrono::high_resolution_clock::now();
    resomp.TransposeOPENMP(a);
    auto finisha = std::chrono::high_resolution_clock::now();

    std::cout << " Duration of seq method = " <<
        std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() << " miccosec\n";

    std::cout << " Duration of omp method = " <<
        std::chrono::duration_cast<std::chrono::microseconds>(finisha - starta).count() << " miccosec\n";

    std::cout << "Accleration : " << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() /
        std::chrono::duration_cast<std::chrono::microseconds>(finisha - starta).count() << " times\n";
    ASSERT_EQ(resomp, resa);
}


TEST(TransposeBlockOpenMP, first) {
    Matrix<int> a(16, 16);
    a.fillRand(1, 10);
    Matrix<int> b;
    b = a;
    ASSERT_EQ(a, b);
    Matrix<int> c;
    c.Transpose_Block(a);

    Matrix<int> s(16, 16);
    for (size_t i = 0; i < s.GetMaxRows(); ++i)
        for (size_t j = 0; j < s.GetMaxCols(); ++j)
            s[i * s.GetMaxCols() + j] = a[j * a.GetMaxCols() + i];

    ASSERT_EQ(s, c);
}

TEST(TransposeBlockOpenMP, second) {
    Matrix<int> a(4096, 4096);
    Matrix<int> resa;
    a.fillRand(1, 1000);
    auto start = std::chrono::high_resolution_clock::now();
    resa.Transpose(a);
    auto finish = std::chrono::high_resolution_clock::now();

    Matrix<int> resomp;
    auto starta = std::chrono::high_resolution_clock::now();
    resomp.Transpose_Block(a);
    auto finisha = std::chrono::high_resolution_clock::now();

    std::cout << " Duration of seq method = " <<
        std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() << " miccosec\n";

    std::cout << " Duration of omp method = " <<
        std::chrono::duration_cast<std::chrono::microseconds>(finisha - starta).count() << " miccosec\n";

    std::cout << "Accleration : " << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() /
        std::chrono::duration_cast<std::chrono::microseconds>(finisha - starta).count() << " times\n";

    ASSERT_EQ(resomp, resa);
}
TEST(TransposeBlockOpenMP, third) {
    Matrix<int> a(16, 16);
    a.fillRand(1, 10);
    Matrix<int> b;
    b = a;
    ASSERT_EQ(a, b);
    Matrix<int> c;
    c.TSPOSE(a);

    Matrix<int> s(16, 16);
    for (size_t i = 0; i < s.GetMaxRows(); ++i)
        for (size_t j = 0; j < s.GetMaxCols(); ++j)
            s[i * s.GetMaxCols() + j] = a[j * a.GetMaxCols() + i];

    ASSERT_EQ(s, c);
}
TEST(Simplemult3TEST, first) {
    Matrix<int> a(5, 2);
    Matrix<int> b(2, 10);
    a.fillValue(10);
    b.fillValue(2);
    Matrix<int>c;
    c.simpleMult3(a, b);
    Matrix<int> res(5, 10);
    res.fillValue(40);
    ASSERT_EQ(res, c);
}
TEST(Simplemult3TEST, second) {
    Matrix<int> a(512, 512);
    a.fillRand(1, 10000);
    Matrix<int> b(512, 512);
    b.fillRand(1, 10000);
    Matrix<int> resa;

    auto start = std::chrono::high_resolution_clock::now();
    resa.simpleMult(a,b);
    auto finish = std::chrono::high_resolution_clock::now();

    Matrix<int> resomp;
    auto starta = std::chrono::high_resolution_clock::now();
    resomp.simpleMult3(a,b);
    auto finisha = std::chrono::high_resolution_clock::now();

    std::cout << " Duration of seq method = " <<
        std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() << " miccosec\n";

    std::cout << " Duration of omp method = " <<
        std::chrono::duration_cast<std::chrono::microseconds>(finisha - starta).count() << " miccosec\n";

    std::cout << "Accleration : " << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() /
        std::chrono::duration_cast<std::chrono::microseconds>(finisha - starta).count() << " times\n";

    ASSERT_EQ(resomp, resa);
}
TEST(OPENMPPerformanceest, stresstest) {
    for (int power = 3; power < 13; power++) {
        int dim = std::pow(2, power);
        std::cout << "OpenMP test with dimention of matrix = " << dim << "\n";

        Matrix<int> A(dim, dim);
        A.fillRand(-1000000, 1000000);
        Matrix<int> B(dim, dim);
        B.fillRand(-1000000, 1000000);
        Matrix<int> C(dim, dim);
        auto start = std::chrono::high_resolution_clock::now();
        C.simpleMult3(A, B);
        auto finish = std::chrono::high_resolution_clock::now();
        std::cout << " Duration of omp method = " <<std::setprecision(5)<<
            std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << " msec\n";
    }

}


int main(int arg_v, char** arg_c) {
    testing::InitGoogleTest(&arg_v, arg_c);
    return RUN_ALL_TESTS();
}