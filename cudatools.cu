#define BLOCK_SIZE 128
#include "Matrix.cpp" 
#include <stdlib.h>
#include <stdio.h>
template<class T>
class FastMatrix: public Matrix<T>
{
public:
    FastMatrix() : Matrix<T>() {}
    FastMatrix(size_t N, size_t M) : Matrix<T>(N, M) { std::cerr << "create fast matrix!\n"; }
    virtual ~FastMatrix() { std::cerr << "Destructor fastmatrix!\n"; };

    FastMatrix& CUDAMultiply(const FastMatrix& matr1, const FastMatrix& matr2);

};

template<class T>
FastMatrix<T>& FastMatrix<T>::CUDAMultiply(const FastMatrix& matr1, const FastMatrix& matr2){
    if (matr1.GetMaxCols() != matr2.GetMaxRows())
        throw std::exception("thease matricies is cannot be multiplied!");
    //TODO - Make setters in parent class
    this->SetMaxRows(matr1.GetMaxRows());
    this->SetMaxCols(matr2.GetMaxCols());
    this->SetLenth(matr1.GetMaxRows(), matr2.GetMaxCols());
    this->AllocateMatr(matr1.GetMaxRows() * matr2.GetMaxCols());
    T dat;
    size_t i, j, k;
    for (i = 0; i < matr1.GetMaxRows(); i++) {
        for (j = 0; j < matr2.GetMaxCols(); j++) {
            dat = 0;
            for (k = 0; k < matr1.GetMaxCols(); k++) {
                dat+= matr1[i * matr1.GetMaxCols() + k] * matr2[k * matr2.GetMaxCols() + j];
            }
            this->AddData(dat, i ,j);
        }
    }
    return *this;
}


__global__ void kernel_global(float* a, float* b, int n, float* c, int k)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    //for ( int j = 0; j < n; j++ ) sum += a [j + t*n] * b [k + j*n]; 
    //c[k+t*n]=sum;

  //   Transp a
    for (int j = 0; j < n; j++) sum += a[t + j * n] * b[k + j * n];
    c[t + k * n] = sum;

}

// Matrix multiplication



FILE* out, * out1, * out2;
void testglob()
{
    int N = 1024;
    int m, n, k;
    float timerValueGPU, timerValueCPU;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);



    int numBytes = N * N * sizeof(float);
    float* adev, * bdev, * cdev, * a, * b, * c, * cc, * bT, * aT;

    a = (float*)malloc(numBytes);
    b = (float*)malloc(numBytes);
    bT = (float*)malloc(numBytes);
    aT = (float*)malloc(numBytes);
    c = (float*)malloc(numBytes);
    cc = (float*)malloc(numBytes);

    for (n = 0; n < N; n++)
    {
        for (m = 0; m < N; m++)
        {
            a[m + n * N] = sinf(m) + cosf(n);
            b[m + n * N] = cosf(m) - sinf(n);
            aT[n + m * N] = a[m + n * N];
            bT[n + m * N] = b[m + n * N];
        }
    }

    dim3 threads(BLOCK_SIZE);
    dim3 blocks(N / threads.x);

    cudaMalloc((void**)&adev, numBytes);	// allocate DRAM
    cudaMalloc((void**)&bdev, numBytes); // allocate DRAM
    cudaMalloc((void**)&cdev, numBytes); // allocate DRAM

    cudaEventRecord(start, 0);
    // copy from CPU to DRAM
    cudaMemcpy(adev, aT, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(bdev, b, numBytes, cudaMemcpyHostToDevice);

    // cudaEventRecord(start, 0);
    for (k = 0; k < N; k++)
    {
        kernel_global << <blocks, threads >> > (adev, bdev, N, cdev, k);
        cudaThreadSynchronize();
    }
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop); 

    cudaMemcpy(c, cdev, numBytes, cudaMemcpyDeviceToHost);


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timerValueGPU, start, stop);

    printf("\n GPU calculation time %f msec\n", timerValueGPU);

    // CPU ---------------------------------------------------------------
    cudaEventRecord(start, 0);

    for (n = 0; n < N; n++)
    {
        for (m = 0; m < N; m++)
        {
            cc[m + n * N] = 0.f;
            //for(k=0;k<N;k++) cc[m+n*N]+=a[k+n*N]*bT[k+m*N]; //best CPU performance
            for (k = 0; k < N; k++) cc[m + n * N] += a[k + n * N] * b[m + k * N]; // poor variant
        }
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timerValueCPU, start, stop);

    printf("\n CPU calculation time %f msec\n", timerValueCPU);
    printf("\n Rate %f x\n", timerValueCPU / timerValueGPU);

    /*
    out=fopen("c_gpu.dat","w"); out1=fopen("c_cpu.dat","w");// out2=fopen("b.dat","w");
    for(n=0;n<N;n++)
    {for(m=0;m<N;m++)
     {fprintf(out,"%e ",c[m+n*N]);
     // fprintf(out,"%e ",c[n+m*N]); // transp
      fprintf(out1,"%e ",cc[m+n*N]);
    //  fprintf(out2,"%e ",b[m+n*N]);
     }
     fprintf(out,"\n"); fprintf(out1,"\n"); //fprintf(out2,"\n");
    }
    fclose(out);  fclose(out1); //fclose(out2);
   */

    cudaFree(adev);
    cudaFree(bdev);
    cudaFree(cdev);
    free(a);
    free(b);
    free(bT);
    free(aT);
    free(c);
    free(cc);

}




// device code
#define BLOCK_SIZE 16

__global__ void kernel_shared(float* a, float* b,
    int n, float* c)
{
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int aBegin = n * BLOCK_SIZE * by;
    int aEnd = aBegin + n - 1;
    int bBegin = BLOCK_SIZE * bx;
    int aStep = BLOCK_SIZE, bStep = BLOCK_SIZE * n;
    float sum = 0.0f;
    __shared__ float as[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float bs[BLOCK_SIZE][BLOCK_SIZE + 1];

    for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep)
    {
        as[tx][ty] = a[ia + n * ty + tx];
        bs[tx][ty] = b[ib + n * ty + tx];

        __syncthreads(); 	// Synchronize to make sure the matrices are loaded 
        for (int k = 0; k < BLOCK_SIZE; k++) sum += as[k][ty] * bs[tx][k];

        __syncthreads(); 	// Synchronize to make sure submatrices not needed
    }
    c[n * BLOCK_SIZE * by + BLOCK_SIZE * bx + n * ty + tx] = sum;
}

__global__ void kernel_shared_1(float* a, float* b,
    int n, float* c)
{
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int aBegin = n * BLOCK_SIZE * by;
    int aEnd = aBegin + n - 1;
    int bBegin = BLOCK_SIZE * bx;
    int aStep = BLOCK_SIZE, bStep = BLOCK_SIZE * n;
    float sum = 0.0f;
    for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep)
    {
        __shared__ float as[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float bs[BLOCK_SIZE][BLOCK_SIZE];
        as[ty][tx] = a[ia + n * ty + tx];
        bs[ty][tx] = b[ib + n * ty + tx];

        __syncthreads(); 	// Synchronize to make sure the matrices are loaded 
        for (int k = 0; k < BLOCK_SIZE; k++) sum += as[ty][k] * bs[k][tx];

        __syncthreads(); 	// Synchronize to make sure submatrices not needed
    }
    c[n * BLOCK_SIZE * by + BLOCK_SIZE * bx + n * ty + tx] = sum;
}

__global__ void kernel_global(float* a, float* b,
    int n, float* c)
{
    int   bx = blockIdx.x;
    int   by = blockIdx.y;
    int   tx = threadIdx.x;
    int   ty = threadIdx.y;
    float sum = 0.0f;
    int   ia = n * BLOCK_SIZE * by + n * ty;
    int   ib = BLOCK_SIZE * bx + tx;

    int   ic = n * BLOCK_SIZE * by + BLOCK_SIZE * bx;

    for (int k = 0; k < n; k++) sum += a[ia + k] * b[ib + k * n];

    c[ic + n * ty + tx] = sum;
}


// host code


FILE* outsh, * out1sh;
void testshared()
{
    int N = 1024;
    int m, n, k;
    float timerValueGPU, timerValueCPU;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int numBytes = N * N * sizeof(float);
    float* adev, * bdev, * cdev, * a, * b, * c, * cc, * bT, * aT;

    a = (float*)malloc(numBytes);
    b = (float*)malloc(numBytes);
    bT = (float*)malloc(numBytes);
    aT = (float*)malloc(numBytes);
    c = (float*)malloc(numBytes);
    cc = (float*)malloc(numBytes);

    for (n = 0; n < N; n++)
    {
        for (m = 0; m < N; m++)
        {
            a[m + n * N] = 2.0f * m + n;
            b[m + n * N] = m - n;
            aT[m + n * N] = m + n * 2.0f;
            bT[m + n * N] = n - m;
        }
    }

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(N / threads.x, N / threads.y);

    cudaMalloc((void**)&adev, numBytes);	// allocate DRAM
    cudaMalloc((void**)&bdev, numBytes); // allocate DRAM
    cudaMalloc((void**)&cdev, numBytes); // allocate DRAM
    //----------------- GPU
    cudaEventRecord(start, 0);
    // copy from CPU to DRAM
    cudaMemcpy(adev, a, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(bdev, b, numBytes, cudaMemcpyHostToDevice);

    kernel_shared_1 << <blocks, threads >> > (adev, bdev, N, cdev);
    //kernel_global<<<blocks, threads>>> ( adev, bdev, N, cdev ); 
    cudaMemcpy(c, cdev, numBytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timerValueGPU, start, stop);
    printf("\n GPU calculation time %f msec\n", timerValueGPU);

    // CPU ---------------------------------------------------------------
    cudaEventRecord(start, 0);

    for (n = 0; n < N; n++)
    {
        for (m = 0; m < N; m++)
        {
            cc[m + n * N] = 0.f;
            for (k = 0; k < N; k++) cc[m + n * N] += a[k + n * N] * bT[k + m * N]; //best CPU performance
          //  for(k=0;k<N;k++) cc[m+n*N]+=a[k+n*N]*b[m+k*N]; // poor variant
        }
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timerValueCPU, start, stop);
    printf("\n CPU calculation time %f msec\n", timerValueCPU);
    printf("\n Rate %f x\n", timerValueCPU / timerValueGPU);

    /*
    out=fopen("c_gpu.dat","w"); out1=fopen("c_cpu.dat","w");
    for(n=0;n<N;n++)
    {for(m=0;m<N;m++)
     {fprintf(out,"%e ",c[m+n*N]);
      fprintf(out1,"%e ",cc[m+n*N]);
     }
     fprintf(out,"\n"); fprintf(out1,"\n");
    }
    fclose(out);  fclose(out1);
   */

    cudaFree(adev);
    cudaFree(bdev);
    cudaFree(cdev);
    free(a);
    free(b);
    free(bT);
    free(aT);
    free(c);
    free(cc);
    // уничтожение переменных-событий
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}
