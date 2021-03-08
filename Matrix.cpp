
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
    virtual ~Matrix();
    void fillRand(int low, int high);
    void fillValue(T value);
    void showMatrix();
    size_t GetMaxCols() const;
    size_t GetMaxRows() const;
    size_t GetMaxCols();
    size_t GetMaxRows();
    size_t GetLength() const;
    size_t GetLength();
    void SetMaxCols(const size_t cols);

    void SetMaxRows(const size_t rows);

    void SetLenth(const size_t length);

    void AllocateMatr(size_t length);
    void AddData(T num, const size_t row, const size_t col);
    void AddData(T num, const size_t len);

    T* GetData();

    void FillData(T* dat);
   
    Matrix& simpleSumm(const Matrix& matr1, const Matrix& matr2);
    Matrix& simpleMult(const Matrix& matr1, const Matrix& matr2);
    Matrix& simpleMult3(const Matrix& matr1, const Matrix& matr2);

    Matrix& Transpose(const Matrix& matr1);
    Matrix& TransposeOPENMP(const Matrix& matr1);

    Matrix& Transpose_Block(const Matrix& matr);
    Matrix& Matrix::TSPOSE(const Matrix& matr1);
    
    // delete copy assignment operator
    Matrix& operator =(const Matrix& matr);
    // move copy operator
    Matrix& operator =(Matrix&& matr); 

    const T& operator()(size_t row, size_t col);
    T& operator()(size_t row, size_t col) const;
    //T& operator()(size_t row, size_t col);
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
void Matrix<T>::SetMaxCols(const size_t cols){
    maxcols = cols;
}
template<class T>
void Matrix<T>::SetMaxRows(const size_t rows){
    maxrows = rows;
}
template<class T>
void Matrix<T>::SetLenth(const size_t length){
    this->length = length;
}
template<class T>
void Matrix<T>::AllocateMatr(size_t length){
    //data[] delete;
    //data = new T[length]();
}
template<class T>
void Matrix<T>::AddData(T num, const size_t row, const size_t col){
    if (col >= maxcols || row >= maxrows) throw std::exception("col or row is out of range!");

    data[row*maxcols + col] = num;
}

template<class T>
void Matrix<T>::AddData(T num, const size_t len){
    if (len >= length) throw std::exception("num of element is out of range!");

    data[len] = num;
}

template<class T>
T* Matrix<T>::GetData(){
    if (data == nullptr) return nullptr;
    else return data;
}
template<class T>
void Matrix<T>::FillData(T* dat){
    data = dat;
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
Matrix<T>& Matrix<T>::simpleMult3(const Matrix<T>& matr1, const Matrix<T>& matr2){
    if (matr1.GetMaxCols() != matr2.GetMaxRows())
        throw std::exception("thease matricies is cannot be multiplied!");

    maxrows = matr1.GetMaxRows();
    maxcols = matr2.GetMaxCols();
    length = matr1.GetMaxRows() * matr2.GetMaxCols();

    delete[] data;
    data = new T[length]();

    Matrix<T> matr2tr; // transpose second matrix
    matr2tr.TSPOSE(matr2);

    int NumProc = omp_get_num_procs();
    omp_set_num_threads(NumProc);
#pragma omp parallel
    {
        int i, j, k;
        T sum;
#pragma omp for
        for (i = 0; i < matr1.GetMaxRows(); i++) {
            for (j = 0; j < matr2.GetMaxCols(); j++) {
                sum = 0;
                for (k = 0; k < matr1.GetMaxCols(); k++) {
                    sum += matr1[i * matr1.GetMaxCols() + k] * matr2tr[j * matr2tr.GetMaxCols() + k];
                }
                data[i * maxcols + j] = sum;
            }
        }
    }
   
    return *this;
}
template<class T>
Matrix<T>& Matrix<T>::Transpose(const Matrix<T>& matr1){
    if (this == &matr1) {
          throw std::exception("don't transpose the same matrix");
     
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
Matrix<T>& Matrix<T>::TSPOSE(const Matrix<T>& matr1) {
    if (matr1.GetMaxCols() % 16 != 0 && matr1.GetMaxRows() % 16 != 0)
        return Matrix<T>::TransposeOPENMP(matr1);
    else
        return Matrix<T>::Transpose_Block(matr1);
}

template<class T>
Matrix<T>& Matrix<T>::TransposeOPENMP(const Matrix<T>& matr1) {
    int NumProc = omp_get_num_procs();
    omp_set_num_threads(NumProc);

    maxrows = matr1.GetMaxCols();
    maxcols = matr1.GetMaxRows();
    length = matr1.GetMaxCols() * matr1.GetMaxRows();
    delete[]data;
    data = new T[maxrows * maxcols]();
    size_t i, j;
 
   
    #pragma omp parallel for
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
//T& Matrix<T>::operator()(size_t row, size_t col){
//    if (row >= maxrows || col >= maxcols) {
//        throw std::out_of_range("operator ()indices is out of range");
//    }
//    else {
//        return data[row * maxcols + col];
//    }
//}

template<class T>
T& Matrix<T>::operator[](size_t num){
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
    int NumProc = omp_get_num_procs();
    omp_set_num_threads(NumProc);
    #pragma omp parallel for
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            B[j * ldb + i] = A[i * lda + j];
        }
    }
}
template <class T>
inline void transpose_block(T* A, T* B, const size_t n, const size_t m, const size_t lda, const size_t ldb, const size_t block_size) {
    int NumProc = omp_get_num_procs();
    omp_set_num_threads(NumProc);
#pragma omp parallel for
    for (int i = 0; i < n; i += block_size) {
        for (int j = 0; j < m; j += block_size) {
            transpose_scalar_block(&A[i * lda + j], &B[j * ldb + i], lda, ldb, block_size);
        }
    }
}
// n - is current dim of A
void getCofactor(Matrix<int>& A, Matrix<int>& temp, int p, int q, int n) {
    int i = 0, j = 0;

    // Looping for each element of the matrix
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            //  Copying into temporary matrix only those element
            //  which are not in given row and column
            if (row != p && col != q) {
//                temp(i,j++) = A(row,col);
                temp.AddData(A(row, col), i, j++);
                // Row is filled, so increase row index and
                // reset col index
                if (j == n - 1){
                    j = 0;
                    i++;
                }
            }
        }
    }
}

void getCofactorOMP(Matrix<int>& A, Matrix<int>& temp, int p, int q, int n) {
    int i = 0, j = 0;
    int NumProc = omp_get_num_procs();
    omp_set_num_threads(NumProc);
#pragma omp parallel for
    // Looping for each element of the matrix
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            //  Copying into temporary matrix only those element
            //  which are not in given row and column
            if (row != p && col != q) {
//                temp(i,j++) = A(row,col);
                temp.AddData(A(row, col), i, j++);
                // Row is filled, so increase row index and
                // reset col index
                if (j == n - 1){
                    j = 0;
                    i++;
                }
            }
        }
    }
}

/* Recursive function for finding determinant of matrix. 
   n is current dimension of A[][]. */
int determinant(Matrix<int>& A, int n){ 
    int D = 0; // Initialize result 
    int N = A.GetMaxCols();
    //  Base case : if matrix contains single element 
    if (n == 1) 
        return A(0,0); 
  
    //int temp[N][N]; // To store cofactors 
    Matrix<int>temp(N,N);
    int sign = 1;  // To store sign multiplier 
  
     // Iterate for each element of first row 
    for (int f = 0; f < n; f++){ 
        // Getting Cofactor of A[0][f] 
        getCofactor(A, temp, 0, f, n); 
        D += sign * A(0,f) * determinant(temp, n - 1); 
  
        // terms are to be added with alternate sign 
        sign = -sign; 
    } 
  
    return D; 
}
/* Recursive function for finding determinant of matrix.
 n is current dimension of A[][]. */
int determinantOMP(Matrix<int>& A, int n){
    int D = 0; // Initialize result
    int N = A.GetMaxCols();
    //  Base case : if matrix contains single element
    if (n == 1)
        return A(0,0);

    //int temp[N][N]; // To store cofactors
    Matrix<int>temp(N,N);
    int sign = 1;  // To store sign multiplier
    int NumProc = omp_get_num_procs();
    omp_set_num_threads(NumProc);
#pragma omp parallel for
    // Iterate for each element of first row
    for (int f = 0; f < n; f++){
        // Getting Cofactor of A[0][f]
        getCofactorOMP(A, temp, 0, f, n);
        D += sign * A(0,f) * determinant(temp, n - 1);

        // terms are to be added with alternate sign
        sign = -sign;
    }

    return D;
}
// Function to get adjoint of A[N][N] in adj[N][N]. 
void adjoint(Matrix<int>& A, Matrix<int>& adj){
    size_t N = A.GetMaxCols();
    if (N == 1){ 
        adj[0] = 1;
        return; 
    } 
  
    // temp is used to store cofactors of A[][] 
    int sign = 1;//, temp[N][N]; 
    Matrix<int> temp(N,N);
    for (int i=0; i<N; i++){ 
        for (int j=0; j<N; j++){ 
            // Get cofactor of A[i][j] 
            getCofactor(A, temp, i, j, N); 
  
            // sign of adj[j][i] positive if sum of row 
            // and column indexes is even. 
            sign = ((i+j)%2==0)? 1: -1; 
  
            // Interchanging rows and columns to get the 
            // transpose of the cofactor matrix 
//            adj[j][i] = (sign)*(determinant(temp, N-1));
            adj.AddData((sign)*(determinant(temp, N-1)), j, i);
        } 
    } 
}
// Function to get adjoint of A[N][N] in adj[N][N].
void adjointOMP(Matrix<int>& A, Matrix<int>& adj){
    size_t N = A.GetMaxCols();
    if (N == 1){
        adj[0] = 1;
        return;
    }

    // temp is used to store cofactors of A[][]
    int sign = 1;//, temp[N][N];
    Matrix<int> temp(N,N);
    int NumProc = omp_get_num_procs();
    omp_set_num_threads(NumProc);
#pragma omp parallel for
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            // Get cofactor of A[i][j]
            getCofactorOMP(A, temp, i, j, N);

            // sign of adj[j][i] positive if sum of row
            // and column indexes is even.
            sign = ((i+j)%2==0)? 1: -1;

            // Interchanging rows and columns to get the
            // transpose of the cofactor matrix
//            adj[j][i] = (sign)*(determinant(temp, N-1));
            adj.AddData((sign)*(determinant(temp, N-1)), j, i);
        }
    }
}
// Function to calculate and store inverse, returns false if 
// matrix is singular 
bool inverse(Matrix<int>& A, Matrix<float>& inverse){ 
    // Find determinant of A[][]
    size_t N = A.GetMaxCols();
    int det = determinant(A, N); 
    if (det == 0){ 
        std::cout << "Singular matrix, can't find its inverse";
        return false; 
    } 
  
    // Find adjoint 
    //int adj[N][N];
    Matrix<int> adj(N,N); 
    adjoint(A, adj); 
  
    // Find Inverse using formula "inverse(A) = adj(A)/det(A)" 
    for (int i=0; i<N; i++) 
        for (int j=0; j<N; j++) 
//            inverse(i,j) = adj(i,j)/float(det);
            inverse.AddData(adj(i,j)/float(det), i, j);
  
    return true; 
}

bool inverseOMP(Matrix<int>& A, Matrix<float>& inverse){
    // Find determinant of A[][]
    size_t N = A.GetMaxCols();
    int det = determinantOMP(A, N);
    if (det == 0){
        std::cout << "Singular matrix, can't find its inverse";
        return false;
    }

    // Find adjoint
    //int adj[N][N];
    Matrix<int> adj(N,N);
    adjointOMP(A, adj);

    // Find Inverse using formula "inverse(A) = adj(A)/det(A)"
    for (int i=0; i<N; i++)
        for (int j=0; j<N; j++)
            inverse.AddData(adj(i,j)/float(det), i, j);

    return true;
}