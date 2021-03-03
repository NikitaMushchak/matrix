
// includes, system
#include <stdio.h>

#include <random>
#include <omp.h>


#define ROUND_UP(x, s) (((x)+((s)-1)) & -(s))
#define NUM_THREADS 8;
// Matrix N M size
template<class T>
class Matrix {
public:
    Matrix() { data = nullptr; maxrows = 0; maxcols = 0; length = 0; }
    Matrix(size_t N, size_t M);
    //copy constructor
    Matrix(const Matrix<T>& matr);
    //Move constructor///
    Matrix(Matrix<T>&& matr);
    ~Matrix();
    void fillRand(int low, int high);
    void fillValue(T value);
    void showMatrix();
    size_t GetMaxCols() const;
    size_t GetMaxRows() const;
    size_t GetMaxCols();
    size_t GetMaxRows();
    size_t GetLength() const;
    size_t GetLength();
    T* GetData();
   
    

    Matrix& simpleSumm(const Matrix& matr1, const Matrix& matr2);
    Matrix& simpleMult(const Matrix& matr1, const Matrix& matr2);

    Matrix& Transpose(const Matrix& matr1);
    Matrix<T>& TransposeOPENMP(const Matrix<T>& matr1);

    Matrix& Transpose_Block(const Matrix& matr);

    
    // delete copy assignment operator
    Matrix& operator =(const Matrix& matr);
    // move copy operator
    Matrix& operator =(Matrix&& matr); 

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
Matrix<T>::Matrix(Matrix<T>&& matr): length(matr.length), maxrows(matr.maxrows), maxcols(matr.maxcols)
{
    delete[]matr.data;
    matr.length = 0;
    matr.maxcols = 0;
    matr.maxrows = 0;
    matr.data = nullptr;
}
template<class T>
Matrix<T>::~Matrix() {
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
template<class T>
size_t Matrix<T>::GetMaxCols()
{
    return maxcols;
}
template<class T>
size_t Matrix<T>::GetMaxRows()
{
    return maxrows;
}
template<class T>
size_t Matrix<T>::GetLength() const
{
    return length;
}
template<class T>
size_t Matrix<T>::GetLength()
{
    return length;
}
template<class T>
T* Matrix<T>::GetData(){
    if (data == nullptr) return nullptr;
    else return data;
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
Matrix<T>& Matrix<T>::Transpose(const Matrix<T>& matr1){
    if (this == &matr1) {
        //TODO realize
        throw std::exception("don't transpose the same matrix");
        /*
        auto cols = matr1.GetMaxCols();
        auto rows = matr1.GetMaxRows();
        maxcols = rows
        maxrows = cols;
        return *this;*/
    }
    else {
        maxrows = matr1.GetMaxCols();
        maxcols = matr1.GetMaxRows();
        length = matr1.GetMaxCols() * matr1.GetMaxRows();
        delete[]data;
        data = new T[maxrows * maxcols]();
        for (size_t i = 0; i < maxrows; ++i) {
            for (size_t j = 0; j < maxcols; ++j) {
                data[i * maxcols + j] = matr1(j, i);
            }
        }
        return *this;
    }
}
template<class T>
Matrix<T>& Matrix<T>::TransposeOPENMP(const Matrix<T>& matr1) {
    int NumProc = omp_get_num_procs();
    omp_set_num_threads(NumProc);
    std::cout << "Number of available cores = " << NumProc << std::endl;
    std::cout<<" omp_get_max_threads() = "<<omp_get_max_threads() << '\n';
  /*  #pragma omp parallel num_threads(8)
    {
        std::cout << "Hello World from Thread " << omp_get_thread_num() << " of " << omp_get_num_threads() << std::endl;
    }
    while (1);*/
    maxrows = matr1.GetMaxCols();
    maxcols = matr1.GetMaxRows();
    length = matr1.GetMaxCols() * matr1.GetMaxRows();
    delete[]data;
    data = new T[maxrows * maxcols]();
    size_t i, j;
    omp_set_num_threads(8);
    std::cout << "threads = " << omp_get_num_threads() << std::endl;
    
   
    #pragma omp parallel for num_threads(8)
    for (auto it = 0; it < length; ++it) {
        i = it / maxcols;
        j = it % maxcols;
        data[it] = matr1[j * maxrows + i];
    }
    
    
    return *this;
}
template<class T>
Matrix<T>& Matrix<T>::Transpose_Block(const Matrix<T>& matr){
    if (matr.GetMaxCols() % 16 != 0 && matr.GetMaxRows() % 16 != 0)
        throw std::exception("this method is acceptable if dim of matrix divite by 16");

    maxrows = matr.GetMaxCols();
    maxcols = matr.GetMaxRows();
    length = matr.GetMaxCols() * matr.GetMaxRows();
    delete[]data;
    data = new T[maxrows * maxcols]();
    const size_t block_size = 16;
    const size_t lda = ROUND_UP(maxrows, block_size);
    const size_t ldb = ROUND_UP(maxcols, block_size);

    transpose_block(matr.data, data, maxrows, maxcols, lda, ldb, block_size);
    
    return *this;
}
template<class T>
Matrix<T>& Matrix<T>::operator=(const Matrix& matr){

    if (&matr == this) {
        return *this;
    }
    if (length >= matr.length) {
        
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
Matrix<T>& Matrix<T>::operator=(Matrix<T>&& matr){
    if (&matr == this)
        return *this;

    delete[]data;
    
    std::memcpy(data, matr.data,  sizeof(T) * matr.length);
    
    maxrows = matr.GetMaxRows();
    maxcols = matr.GetMaxCols();
    length = matr.GetMaxRows()* matr.GetMaxCols();
    
    
    matr.length = 0;
    matr.maxcols = 0;
    matr.maxrows = 0;
    matr.data = nullptr;
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
        throw std::out_of_range("operator () indices is out of range");
    }
    else {
        return data[row * maxcols + col];
    }
}
template<class T>
T& Matrix<T>::operator()(size_t row, size_t col) const{
    if (row >= maxrows || col >= maxcols) {
        throw std::out_of_range("operator ()indices is out of range");
    }
    else {
        return data[row * maxcols + col];
    }
}
//template<class T>
//T& Matrix<T>::operator()(const size_t row, const size_t col) {
//    if (row >= maxrows || col >= maxcols) {
//        throw std::out_of_range("indices is out of range");
//    }
//    else {
//        return data[row * maxcols + col];
//    }
//}
template<class T>
T& Matrix<T>::operator[](size_t num){
    // TODO: insert return statement here
    if (num >= length) {
        throw std::out_of_range(" operator [] num is out of range!");
    }
    else {
        return data[num];
    }
}

template<class T>
T& Matrix<T>::operator[](size_t num) const{

    if (num >= length) {
        throw std::out_of_range("operator [] num is out of range!");
    }
    else {
        return data[num];
    }
}
template<class T>
Matrix<T> simpleMult2(const Matrix<T>& matr1, const Matrix<T>& matr2) {
    if (matr1.GetMaxCols() != matr2.GetMaxRows())
        throw std::exception("thease matricies is cannot be multiplied!");

    auto maxrows = matr1.GetMaxRows();
    auto maxcols = matr2.GetMaxCols();
    Matrix<T> matr3(maxrows, maxcols);

    size_t i, j, k;
    for (i = 0; i < matr1.GetMaxRows(); i++) {
        for (j = 0; j < matr2.GetMaxCols(); j++) {
            for (k = 0; k < matr1.GetMaxCols(); k++) {
                matr3[i * maxcols + j] += matr1[i * matr1.GetMaxCols() + k] * matr2[k * matr2.GetMaxCols() + j];
            }
        }
    }
    return matr3;
}
template <class T>
inline void transpose_scalar_block(T* A, T* B, const size_t lda, const size_t ldb, const size_t block_size) {
    omp_set_num_threads(8);
    /*std::cout << "threads = " << omp_get_num_threads() << std::endl;*/
    #pragma omp parallel for
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            B[j * ldb + i] = A[i * lda + j];
        }
    }
}
template <class T>
inline void transpose_block(T* A, T* B, const size_t n, const size_t m, const size_t lda, const size_t ldb, const size_t block_size) {
    omp_set_num_threads(8);
    std::cout << "threads = " << omp_get_num_threads() << std::endl;
    #pragma omp parallel for
    for (int i = 0; i < n; i += block_size) {
        for (int j = 0; j < m; j += block_size) {
            transpose_scalar_block(&A[i * lda + j], &B[j * ldb + i], lda, ldb, block_size);
        }
    }
}
