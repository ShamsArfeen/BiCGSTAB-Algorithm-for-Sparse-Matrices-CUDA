#include "stdio.h"
#include "time.h"
#include "stdlib.h"
#include "omp.h"

/* nvcc heptaMatCorrectness.cu -Xcompiler -fopenmp */

#define SPACING 7
#define BLOCKSIZE 1024
#define N (BLOCKSIZE * 1000) 
#define CACHE (2 * SPACING + BLOCKSIZE + 1)

/* SPACING : Offset b/w main and distant diagonal, greater than 2 */
/* BLOCKSIZE : Threads per block, multiple of 32 */
/* N : Row/Column size */
/* CACHE : Size of shared memory per block */

#define HEPT(a) ((a)+3)
int BLOCK;

struct sparseMatrix {
    float* DIA[7];
} A;

float *X;
float *B;
int n;

float Ah[7][N], Xh[(N + 2*SPACING)], Bh[N];


__global__ void heptaMatrixMul( struct sparseMatrix A, float *X, float *B) {

    int thrId = blockIdx.x * blockDim.x + threadIdx.x;

    int middle = SPACING + thrId;

    float Bvalue;
    Bvalue =    A.DIA[HEPT(3)][thrId]   *X[thrId]
            +   A.DIA[HEPT(2)][thrId]   *X[middle - 2]
            +   A.DIA[HEPT(1)][thrId]   *X[middle - 1]
            +   A.DIA[HEPT(0)][thrId]   *X[middle]
            +   A.DIA[HEPT(-1)][thrId]  *X[middle + 1]
            +   A.DIA[HEPT(-2)][thrId]  *X[middle + 2]
            +   A.DIA[HEPT(-3)][thrId]  *X[middle + SPACING];

    B[thrId] = Bvalue;
}

__global__ void heptaMatrixMulSharedMem( struct sparseMatrix A, float *X, float *B) {

    int thrId = blockIdx.x * blockDim.x + threadIdx.x;

    int i;
    __shared__ float Xc[CACHE];

    for ( i = threadIdx.x; i < (2 * SPACING + blockDim.x + 1); i+= blockDim.x )
        Xc[i] = X[blockIdx.x * blockDim.x + i];

    __syncthreads();

    int middle = SPACING + threadIdx.x;

    float Bvalue;
    Bvalue =    A.DIA[HEPT(3)][thrId]   *Xc[threadIdx.x]
            +   A.DIA[HEPT(2)][thrId]   *Xc[middle - 2]
            +   A.DIA[HEPT(1)][thrId]   *Xc[middle - 1]
            +   A.DIA[HEPT(0)][thrId]   *Xc[middle]
            +   A.DIA[HEPT(-1)][thrId]  *Xc[middle + 1]
            +   A.DIA[HEPT(-2)][thrId]  *Xc[middle + 2]
            +   A.DIA[HEPT(-3)][thrId]  *Xc[middle + SPACING];

    B[thrId] = Bvalue;
}

void heptaMatSerial( float (*A)[N], float *X, float *B) {

    int i, middle;
    #pragma omp parallel for private(middle)
    for ( i = 0; i < n; ++i) {
        middle = i + SPACING;
        B[i] =      A[HEPT(3)][i]   *X[i]
                +   A[HEPT(2)][i]   *X[middle - 2]
                +   A[HEPT(1)][i]   *X[middle - 1]
                +   A[HEPT(0)][i]   *X[middle]
                +   A[HEPT(-1)][i]  *X[middle + 1]
                +   A[HEPT(-2)][i]  *X[middle + 2]
                +   A[HEPT(-3)][i]  *X[middle + SPACING];
    }
}

void run() {
    
    float Time;
    clock_t begin, end;

    for ( BLOCK = 32; BLOCK <= BLOCKSIZE; BLOCK *= 2 ) {
        begin = clock();
        heptaMatrixMul <<<n/BLOCK, BLOCK>>> ( A, X, B);
        cudaDeviceSynchronize();
        end = clock();
        Time = (float)(end - begin) / CLOCKS_PER_SEC;
        printf("%f\t", Time);
    }
    

    for ( BLOCK = 32; BLOCK <= BLOCKSIZE; BLOCK *= 2 ) {
        begin = clock();
        heptaMatrixMulSharedMem <<<n/BLOCK, BLOCK>>> ( A, X, B);
        cudaDeviceSynchronize();
        end = clock();
        Time = (float)(end - begin) / CLOCKS_PER_SEC;
        printf("%f\t", Time);
    }

    begin = clock();
    heptaMatSerial( Ah, Xh, Bh);
    end = clock();
    Time = (float)(end - begin) / CLOCKS_PER_SEC;
    printf("%f\n", Time);
}

void initialize ();
void freeResources ();

int main ( int argc, char *argv[]) {

    printf(" N\t");
    for ( BLOCK = 32; BLOCK <= BLOCKSIZE; BLOCK *= 2 ) {
        printf("%d-GLOBL\t", BLOCK);
    }
    for ( BLOCK = 32; BLOCK <= BLOCKSIZE; BLOCK *= 2 ) {
        printf("%d-SHARE\t", BLOCK);
    }
    printf("OPENMP\n");

    for ( n = BLOCKSIZE * 100; n <= N; n += BLOCKSIZE * 100) {
        printf("%d\t", n);
        initialize ();
        run();
        freeResources ();
    }
    return 0;
}


void initialize() {

    cudaMalloc (&X,     (N + 2*SPACING)* sizeof(float));
    cudaMalloc (&B,     N* sizeof(float));

    cudaMalloc (&A.DIA[HEPT(0)],     N* sizeof(float));
    cudaMalloc (&A.DIA[HEPT(-1)],    N* sizeof(float));
    cudaMalloc (&A.DIA[HEPT(1)],     N* sizeof(float));
    cudaMalloc (&A.DIA[HEPT(-2)],    N* sizeof(float));
    cudaMalloc (&A.DIA[HEPT(2)],     N* sizeof(float));
    cudaMalloc (&A.DIA[HEPT(-3)],    N* sizeof(float));
    cudaMalloc (&A.DIA[HEPT(3)],     N* sizeof(float));
}

void freeResources () {
    cudaFree(X);
    cudaFree(B);
    cudaFree( A.DIA[HEPT(0)]);
    cudaFree( A.DIA[HEPT(-1)]);
    cudaFree( A.DIA[HEPT(1)]);
    cudaFree( A.DIA[HEPT(-2)]);
    cudaFree( A.DIA[HEPT(2)]);
    cudaFree( A.DIA[HEPT(-3)]);
    cudaFree( A.DIA[HEPT(3)]);
}