#include "stdio.h"
#include "time.h"
#include "stdlib.h"
#include "omp.h"

/* nvcc heptaMatCorrectness.cu -Xcompiler -fopenmp */

#define SPACING 7
#define BLOCKSIZE 64
#define N (BLOCKSIZE * 100) 
#define CACHE (2 * SPACING + BLOCKSIZE + 1)

/* SPACING : Offset b/w main and distant diagonal, greater than 2 */
/* BLOCKSIZE : Threads per block, multiple of 32 */
/* N : Row/Column size */
/* CACHE : Size of shared memory per block */

#define HEPT(a) ((a)+3)

struct sparseMatrix {
    float* DIA[7];
} A;

float *X;
float *B;

 /* Host Arrays for Correctness check */
float As[N][N], Xs[N], Bs[N];

__global__ void heptaMatrixMul( struct sparseMatrix A, float *X, float *B) {

    int thrId = blockIdx.x * blockDim.x + threadIdx.x;

    int i;
    __shared__ float Xc[CACHE];

    for ( i = threadIdx.x; i < CACHE; i+= blockDim.x )
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
    for ( i = 0; i < N; ++i) {
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

void verify() {
    float Ah[7][N], Xh[(N + 2*SPACING)], Bh[N], B2[N], Time;
    int i, j;

    for ( i = 0; i < N; ++i)
        Ah[HEPT(0)][i] = As[i][i];

    for ( i = 0; i < (N + 2*SPACING); ++i)
        Xh[i + SPACING] = Xs[i];

    for ( i = 0; i < SPACING; ++i) {
        Xh[i] = 0;
        Xh[(N + 2*SPACING) - i] = 0;
    }
    
    cudaMemcpy(A.DIA[HEPT(0)], Ah[HEPT(0)], 
        sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(X, Xh, 
        sizeof(float) * (N + 2*SPACING), cudaMemcpyHostToDevice);

    for ( i = 0; i < N; ++i) {
        if ( i < 1) {
            Ah[(HEPT(-1))][N - i - 1] = 0;
            Ah[(HEPT(1))][i] = 0;
        }
        else {
            Ah[(HEPT(-1))][i - 1] = As[i - 1][i];
            Ah[(HEPT(1))][i] = As[i][i - 1];
        }
    }
    cudaMemcpy(A.DIA[HEPT(-1)], Ah[(HEPT(-1))], 
        sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(A.DIA[HEPT(1)], Ah[(HEPT(1))], 
        sizeof(float) * N, cudaMemcpyHostToDevice);

    for ( i = 0; i < N; ++i) {
        if ( i < 2) {
            Ah[(HEPT(-2))][N - i - 1] = 0;
            Ah[(HEPT(2))][i] = 0;
        }
        else {
            Ah[(HEPT(-2))][i - 2] = As[i - 2][i];
            Ah[(HEPT(2))][i] = As[i][i - 2];
        }
    }
    cudaMemcpy(A.DIA[HEPT(-2)], Ah[(HEPT(-2))], 
        sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(A.DIA[HEPT(2)], Ah[(HEPT(2))], 
        sizeof(float) * N, cudaMemcpyHostToDevice);

    for ( i = 0; i < N; ++i) {
        if ( i < SPACING) {
            Ah[(HEPT(-3))][N - i - 1] = 0;
            Ah[(HEPT(3))][i] = 0;
        }
        else {
            Ah[(HEPT(-3))][i - SPACING] = As[i - SPACING][i];
            Ah[(HEPT(3))][i] = As[i][i - SPACING];
        }
    }
    cudaMemcpy(A.DIA[HEPT(-3)], Ah[(HEPT(-3))], 
        sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(A.DIA[HEPT(3)], Ah[(HEPT(3))], 
        sizeof(float) * N, cudaMemcpyHostToDevice);

    /* Matrix-Vector Normal Multiplication O(N^2) */
    
    for ( i = 0; i < N; ++i) {
        float Bvalue = 0;
        for ( j = 0; j < N; ++j)
            Bvalue += As[i][j] * Xs[j];
        Bs[i] = Bvalue;
    }

    clock_t begin, end;

    begin = clock();
    heptaMatrixMul <<<N/BLOCKSIZE, BLOCKSIZE>>> ( A, X, B);
    cudaDeviceSynchronize();

    end = clock();
    Time = (float)(end - begin) / CLOCKS_PER_SEC;
    printf("GPU TIME: %f\n", Time);
    cudaMemcpy(B2, B, 
        sizeof(float) * N, cudaMemcpyDeviceToHost);

    begin = clock();
    heptaMatSerial( Ah, Xh, Bh);

    end = clock();
    Time = (float)(end - begin) / CLOCKS_PER_SEC;
    printf("CPU TIME: %f\n", Time);

    printf("\n CUDA   \tOPENMP  \tSERIAL (MV-SAXPY)\n");
    for ( i = 0; i < 2 * SPACING; ++i)
        printf(" %f\t%f\t%f\n", B2[i], Bh[i], Bs[i]);
        
    for ( i = 0; i < N; ++i)
        if (abs(B2[i] - Bh[i]) > 1e-04 && abs(Bs[i] - Bh[i]) > 1e-04) printf("Error %f %f %f  :(\n", B2[i], Bh[i], Bs[i]);
    printf("Results are Equivalent\n");
}

void initialize ();
void freeResources ();

int main ( int argc, char *argv[]) {
    initialize ();
    verify ();
    freeResources ();
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

    srand(  time(0));
    int i, j;

    for ( i = 0; i < N; ++i)
    for ( j = 0; j < N; ++j)
        As[i][j] = 0;

    for ( i = 0; i < N; ++i) {
        As[i][i] = rand() / (float) RAND_MAX;
        Xs[i] = rand() / (float) RAND_MAX;
    }
    for ( i = 1; i < N; ++i) {
        As[i - 1][i] = rand() / (float) RAND_MAX;
        As[i][i - 1] = rand() / (float) RAND_MAX;
    }
    for ( i = 2; i < N; ++i) {
        As[i - 2][i] = rand() / (float) RAND_MAX;
        As[i][i - 2] = rand() / (float) RAND_MAX;
    }
    for ( i = SPACING; i < N; ++i) {
        As[i - SPACING][i] = rand() / (float) RAND_MAX;
        As[i][i - SPACING] = rand() / (float) RAND_MAX;
    }
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