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

struct sparseMatrix {
    float* DIA[7];
} A;

float *X;
float *B;

float Ah[7][N], Xh[(N + 2*SPACING)], Bh[N], B2[N];

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

void launch() {
    float Time;
    int i;

    for ( i = 0; i < N; ++i)
        Ah[HEPT(0)][i] = rand() / (float) RAND_MAX;

    for ( i = 0; i < (N + 2*SPACING); ++i)
        Xh[i + SPACING] = rand() / (float) RAND_MAX;

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
            Ah[(HEPT(-1))][i - 1] = rand() / (float) RAND_MAX;
            Ah[(HEPT(1))][i] = rand() / (float) RAND_MAX;
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
            Ah[(HEPT(-2))][i - 2] = rand() / (float) RAND_MAX;
            Ah[(HEPT(2))][i] = rand() / (float) RAND_MAX;
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
            Ah[(HEPT(-3))][i - SPACING] = rand() / (float) RAND_MAX;
            Ah[(HEPT(3))][i] = rand() / (float) RAND_MAX;
        }
    }
    cudaMemcpy(A.DIA[HEPT(-3)], Ah[(HEPT(-3))], 
        sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(A.DIA[HEPT(3)], Ah[(HEPT(3))], 
        sizeof(float) * N, cudaMemcpyHostToDevice);

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

    printf("\n CUDA   \tOPENMP\n");
    for ( i = 0; i < 2 * SPACING; ++i)
        printf(" %f\t%f\n", B2[i], Bh[i]);

    for ( i = 0; i < N; ++i)
        if (abs(B2[i] - Bh[i]) > 1e-04) printf("Error %f %f :(\n", B2[i], Bh[i]);
    printf("Results are Equivalent\n");
}

void initialize ();
void freeResources ();

int main ( int argc, char *argv[]) {
    initialize ();
    launch ();
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