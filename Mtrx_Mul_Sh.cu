

// includes, system
#include <stdio.h>

// includes CUDA Runtime
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper utility functions 
#include <random>


// Matrix N M size
template<class T>
class Matrix {
public:
    Matrix(size_t N, size_t M);
    Matrix(const Matrix<T>& matr);
    ~Matrix();
    void fillRand(int low, int high);
    void fillValue(T value);
    void showMatrix();
    size_t GetMaxCols() const;
    size_t GetMaxCols();
    size_t GetMaxRows();
   
    size_t GetMaxRows() const;

    Matrix& simpleSumm(const Matrix& matr1, const Matrix& matr2);
    Matrix& simpleMult(const Matrix& matr1, const Matrix& matr2);
    Matrix& operator =(const Matrix& matr);

    const T& operator()(size_t row, size_t col);
    T& operator()(size_t row, size_t col) const;
    T& operator [](size_t num);
    T& operator [](size_t num) const;
private:
    T* data;
    size_t length;
    size_t maxrows;
    size_t maxcols;
};

template <class T>
Matrix<T>::Matrix(size_t N, size_t M) :maxrows(N), maxcols(M) {
    length = N * M;
    data = new T[length]();
}
template<class T>
Matrix<T>::Matrix(const Matrix<T>& matr): length(matr.length), maxrows(matr.maxrows), maxcols(matr.maxcols){
    data = new T[matr.length]();
    std::memcpy(data, matr.data, sizeof(T) * matr.length);
}
template<class T>
Matrix<T>::~Matrix() {
    if (data)
        delete[]data;
}
template <class T>
void Matrix<T>::fillRand(int lowest, int highest) {
    std::random_device rd;
    std::mt19937 mt;
    std::uniform_int_distribution<int> dist(lowest, highest);
    for(size_t i = 0; i<length; ++i)
        data[i] = dist(mt);
}
template<class T>
void Matrix<T>::fillValue(T value){
    for (size_t i = 0; i < length; ++i)
        data[i] = value;
}
template<class T>
void Matrix<T>::showMatrix() {
    std::cout << "Matrix num of rows : " << maxrows << " max of cols : " << maxcols;
    for (size_t i = 0; i < maxrows; ++i) {
        std::cout << "\n";
        for (size_t j = 0; j < maxcols; ++j) {
            std::cout << data[i * maxcols + j] << "\t";
        }
    }
    std::cout << '\n';
}
template<class T>
size_t Matrix<T>::GetMaxCols() const{
    return maxcols;
}
template <class T>
size_t Matrix<T>::GetMaxRows() const{
    return maxrows;
}
// TODO : check all T types and no int types!!!

template<class T>
Matrix<T>& Matrix<T>::simpleSumm(const Matrix<T>& matr1, const Matrix<T>& matr2) {
    if (matr1.GetMaxCols() == matr2.GetMaxCols() && matr1.GetMaxRows() == matr2.GetMaxRows()) {
        size_t maxCols = matr1.GetMaxCols();
        size_t maxRows = matr1.GetMaxRows();
        delete[] data;
        data = new T[matr1.length]();
        size_t index = 0;
        for (size_t i = 0; i < maxRows; ++i) {
            for (size_t j = 0; j < maxCols; ++j) {
                index = i * maxCols + j;
                data[index] = matr1[index] + matr2[index];
            }
        }
        maxrows = matr1.maxrows;
        maxcols = matr1.maxcols;
        length = matr1.length;
      
        return *this;
    }
    else {
        throw std::exception("matrices is not the same dimentions!");
    }
}
template<class T>
Matrix<T>& Matrix<T>::simpleMult(const Matrix<T>& matr1, const Matrix<T>& matr2){
    if (matr1.GetMaxCols() != matr2.GetMaxRows())
        throw std::exception("thease matricies is cannot be multiplied!");

    maxrows = matr1.maxrows;
    maxcols = matr2.maxcols;
    length = maxrows * maxcols;
    delete[] data;
    data = new T[length]();
    size_t i, j, k;
    for (i = 0; i < matr1.GetMaxRows(); i++) {
        for (j = 0; j < matr2.GetMaxCols(); j++) {
            for (k = 0; k < matr1.GetMaxCols(); k++) {
                data[i*maxcols + j] += matr1[i* matr1.GetMaxCols()+ k] * matr2[k* matr2.GetMaxCols() +j];
            }
        }
    }
    return *this;
}
template<class T>
Matrix<T>& Matrix<T>::operator=(const Matrix& matr){

    if (&matr == this) {
        return *this;
    }
    if (length <= matr.length) {
        std::memcpy(data, matr.data, matr.length * sizeof(T));
    }
    else {
        delete[] data;
        data = new T[matr.length]();
        std::memcpy(data, matr.data, matr.length * sizeof(T));
    }
    maxrows = matr.maxrows;
    maxcols = matr.maxcols;
    length = matr.length;

    return *this;
}
template<class T>
bool operator==(const Matrix<T>& rhm, const Matrix<T>& lhm){
    if (rhm.GetMaxCols() == lhm.GetMaxCols() && rhm.GetMaxRows() == lhm.GetMaxRows()) {
        for (size_t i = 0; i < lhm.GetMaxCols()*lhm.GetMaxRows(); ++i)
            if (rhm[i] != lhm[i])
                return false;

        return true;
    }
    else
        throw std::exception("Matrices is not the same size!");
    return false;
}

template <class T>
const T& Matrix<T>::operator()(size_t row, size_t col) {
    if (row >= maxrows || col >= maxcols) {
        throw std::out_of_range("indices is out of range");
    }
    else {
        return data[row * maxcols + col];
    }
}
template<class T>
T& Matrix<T>::operator()(size_t row, size_t col) const{
    if (row >= maxrows || col >= maxcols) {
        throw std::out_of_range("indices is out of range");
    }
    else {
        return data[row * maxcols + col];
    }
}

template<class T>
T& Matrix<T>::operator[](size_t num){
    // TODO: insert return statement here
    if (num >= length) {
        throw std::out_of_range("num is out of range!");
    }
    else {
        return data[num];
    }
}

template<class T>
T& Matrix<T>::operator[](size_t num) const{

    if (num >= length) {
        throw std::out_of_range("num is out of range!");
    }
    else {
        return data[num];
    }
}


__global__ void increment_kernel(int *g_data, int inc_value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    g_data[idx] = g_data[idx] + inc_value;
}

bool correct_output(int *data, const int n, const int x)
{
    for (int i = 0; i < n; i++)
        if (data[i] != x)
        {
            printf("Error! data[%d] = %d, ref = %d\n", i, data[i], x);
            return false;
        }

    return true;
}

int test1(int argc, char *argv[])
{
    int devID;
    cudaDeviceProp deviceProps;

    printf("[%s] - Starting...\n", argv[0]);

    // This will pick the best possible CUDA capable device
    devID = findCudaDevice(argc, (const char **)argv);

    // get device name
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
    printf("CUDA device [%s]\n", deviceProps.name);

    int n = 16 * 1024 * 1024;
    int nbytes = n * sizeof(int);
    int value = 26;

    // allocate host memory
    int *a = 0;
    checkCudaErrors(cudaMallocHost((void **)&a, nbytes));
    memset(a, 0, nbytes);

    // allocate device memory
    int *d_a=0;
    checkCudaErrors(cudaMalloc((void **)&d_a, nbytes));
    checkCudaErrors(cudaMemset(d_a, 255, nbytes));

    // set kernel launch configuration
    dim3 threads = dim3(512, 1);
    dim3 blocks  = dim3(n / threads.x, 1);

    // create cuda event handles
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    checkCudaErrors(cudaDeviceSynchronize());
    float gpu_time = 0.0f;

    // asynchronously issue work to the GPU (all to stream 0)
    sdkStartTimer(&timer);
    cudaEventRecord(start, 0);
    cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice, 0);
    increment_kernel<<<blocks, threads, 0, 0>>>(d_a, value);
    cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);
    cudaEventRecord(stop, 0);
    sdkStopTimer(&timer);

    // have CPU do some work while waiting for stage 1 to finish
    unsigned long int counter=0;

    while (cudaEventQuery(stop) == cudaErrorNotReady)
    {
        counter++;
    }

    checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));

    // print the cpu and gpu times
    printf("time spent executing by the GPU: %.2f\n", gpu_time);
    printf("time spent by CPU in CUDA calls: %.2f\n", sdkGetTimerValue(&timer));
    printf("CPU executed %lu iterations while waiting for GPU to finish\n", counter);

    // check the output for correctness
    bool bFinalResults = correct_output(a, n, value);

    // release resources
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaFreeHost(a));
    checkCudaErrors(cudaFree(d_a));
    char s;
     std::cin >> s;
    exit(bFinalResults ? EXIT_SUCCESS : EXIT_FAILURE);
}
