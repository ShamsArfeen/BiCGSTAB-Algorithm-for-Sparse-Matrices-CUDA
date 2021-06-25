#include "stdio.h"
#include "time.h"
#include "curand.h"

#define SPACING 7 
#define BLOCKSIZE 128 
#define N (1024 * 1000) 
#define CACHE (2 * SPACING + BLOCKSIZE + 1)

/* SPACING : Offset b/w main and distant diagonal */
/* BLOCKSIZE : Threads per block, multiple of 32 */
/* N : Row/Column size */
/* CACHE : Size of shared memory per block */

#define HEPT(a) ((a)+3)

struct sparseMatrix {
    float* DIA[7];
} A;

float *X;
float *B;


void freeResources ();
void initialize ();

__global__ void heptaMatrixMul ( struct sparseMatrix A, float *X, float *B) {

    /* Using only Global Memory */
    int thrId = blockIdx.x * blockDim.x + threadIdx.x;

    int middle = SPACING + thrId;

    float Bvalue;
    Bvalue = A.DIA[HEPT(3)][thrId] * X[thrId]
            + A.DIA[HEPT(2)][thrId] *  X[middle - 2]
            + A.DIA[HEPT(1)][thrId] *  X[middle - 1]
            + A.DIA[HEPT(0)][thrId] *  X[middle]
            + A.DIA[HEPT(-1)][thrId] * X[middle + 1]
            + A.DIA[HEPT(-2)][thrId] * X[middle + 2]
            + A.DIA[HEPT(-3)][thrId] * X[middle + SPACING];

    B[thrId] = Bvalue;
}


__global__ void heptaMatrixMulSharedMem ( struct sparseMatrix A, float *X, float *B) {

    /* Caching vector X in Shared Memory 
    to avoid accessing each element 7 times 
    from Global Memory */

    int thrId = blockIdx.x * blockDim.x + threadIdx.x;

    int i, OFFSET = blockIdx.x * blockDim.x;
    __shared__ float Xc[CACHE];

    for ( i = threadIdx.x; i < (2 * SPACING + blockDim.x + 1); i+= blockDim.x )
        Xc[i] = X[OFFSET + i];

    __syncthreads();

    int middle = SPACING + threadIdx.x;

    float Bvalue;
    Bvalue = A.DIA[HEPT(3)][thrId] * Xc[threadIdx.x]
            + A.DIA[HEPT(2)][thrId] *  Xc[middle - 2]
            + A.DIA[HEPT(1)][thrId] *  Xc[middle - 1]
            + A.DIA[HEPT(0)][thrId] *  Xc[middle]
            + A.DIA[HEPT(-1)][thrId] * Xc[middle + 1]
            + A.DIA[HEPT(-2)][thrId] * Xc[middle + 2]
            + A.DIA[HEPT(-3)][thrId] * Xc[middle + SPACING];

    B[thrId] = Bvalue;
}


void performance () {

    /* Performance measure for Shared Memory Vs. only Global Memory */

    int i;
    float resultSharedMem[N], resultGlobalMem[N];



    float elapsed=0;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    heptaMatrixMul <<<N/BLOCKSIZE, BLOCKSIZE>>> ( A, X, B);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize (stop);

    cudaEventElapsedTime(&elapsed, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("GLOBAL MEMORY TIME: %f\n", elapsed);




    cudaMemcpy(resultGlobalMem, B, 
        sizeof(float) * N, cudaMemcpyDeviceToHost);

    freeResources (); /* Flushing and Refilling */
    initialize (); /* to refresh GPU caches */


    elapsed=0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    heptaMatrixMulSharedMem <<<N/BLOCKSIZE, BLOCKSIZE>>> ( A, X, B);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize (stop);

    cudaEventElapsedTime(&elapsed, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("SHARED MEMORY TIME: %f\n", elapsed);

    
    cudaMemcpy(resultSharedMem, B, 
        sizeof(float) * N, cudaMemcpyDeviceToHost);
    
    printf("\n SHARED  \tGLOBAL\n");
    for ( i = 0; i < 2 * SPACING; ++i)
        printf(" %f\t%f\n", 
            resultSharedMem[i], resultGlobalMem[i]);

    for ( i = 0; i < N; ++i)
        if (resultSharedMem[i] != resultGlobalMem[i]) 
            printf("Error %f %f  :(\n", resultSharedMem[i], resultGlobalMem[i]);
    printf("Results are Equivalent\n");
}

int main ( int argc, char *argv[]) {
    initialize ();
    performance ();
    freeResources ();
    return 0;
}

 /* Random floats generation of Hepta Matrix using lib cuRAND */

void freeResources() {
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

void initialize() {

    /* AX = B */

    /* Extended vector X */
    cudaMalloc(&X, (N + 2*SPACING)* sizeof(float));

    /* Resultant vector B */
    cudaMalloc(&B, N* sizeof(float));

    /* Extended diagonals of Hepta-Mat A */
    cudaMalloc(&A.DIA[HEPT(0)],     N* sizeof(float));
    cudaMalloc(&A.DIA[HEPT(-1)],    N* sizeof(float));
    cudaMalloc(&A.DIA[HEPT(1)],     N* sizeof(float));
    cudaMalloc(&A.DIA[HEPT(-2)],    N* sizeof(float));
    cudaMalloc(&A.DIA[HEPT(2)],     N* sizeof(float));
    cudaMalloc(&A.DIA[HEPT(-3)],    N* sizeof(float));
    cudaMalloc(&A.DIA[HEPT(3)],     N* sizeof(float));

    
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    /* Filling random floats in A and X */
    curandGenerateUniform(gen, X, N);
    curandGenerateUniform(gen, A.DIA[HEPT(0)],  N);
    curandGenerateUniform(gen, A.DIA[HEPT(-1)], N);
    curandGenerateUniform(gen, A.DIA[HEPT(1)],  N);
    curandGenerateUniform(gen, A.DIA[HEPT(-2)], N);
    curandGenerateUniform(gen, A.DIA[HEPT(2)],  N);
    curandGenerateUniform(gen, A.DIA[HEPT(-3)], N);
    curandGenerateUniform(gen, A.DIA[HEPT(3)],  N);

    curandDestroyGenerator(gen);
}