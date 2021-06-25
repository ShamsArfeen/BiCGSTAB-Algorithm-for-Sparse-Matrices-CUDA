#include "stdio.h"
#include "time.h"
#include "stdlib.h"
#include "omp.h"
#include "curand.h"

/* nvcc heptaMatCorrectness.cu -Xcompiler -fopenmp */

#define SPACING 7
#define BLOCKSIZE 128
#define N (1024 * 1000) 
#define CACHE (2 * SPACING + BLOCKSIZE + 1)
#define ITER 32

/* SPACING : Offset b/w main and distant diagonal, greater than 2 */
/* BLOCKSIZE : Threads per iter, multiple of 32 */
/* N : Row/Column size */
/* CACHE : Size of shared memory per iter */

#define HEPT(a) ((a)+3)
int iter;

struct sparseMatrix {
    float* DIA[7];
} A;

float *X;
float *B;
int n;

float Ah[7][N], Xh[(N + 2*SPACING)], Bh[N];


__global__ void heptaMatrixMul( struct sparseMatrix A, float *X, float *B, int n) {

    int thrId = blockIdx.x * blockDim.x + threadIdx.x;
    int t;
    for ( t = thrId; t < n; t += gridDim.x * blockDim.x) {
        
        int middle = SPACING + t;

        float Bvalue;
        Bvalue =    A.DIA[HEPT(3)][t]   *X[t]
                +   A.DIA[HEPT(2)][t]   *X[middle - 2]
                +   A.DIA[HEPT(1)][t]   *X[middle - 1]
                +   A.DIA[HEPT(0)][t]   *X[middle]
                +   A.DIA[HEPT(-1)][t]  *X[middle + 1]
                +   A.DIA[HEPT(-2)][t]  *X[middle + 2]
                +   A.DIA[HEPT(-3)][t]  *X[middle + SPACING];

        B[t] = Bvalue;
    }
}

__global__ void heptaMatrixMulSharedMem( struct sparseMatrix A, float *X, float *B, int n) {

    int thrId = blockIdx.x * blockDim.x + threadIdx.x;

    int i, t, k = 0;
    __shared__ float Xc[CACHE];

    for ( t = thrId; t < n; t += gridDim.x * blockDim.x) {

        for ( i = threadIdx.x; i < (2 * SPACING + blockDim.x + 1); i+= blockDim.x )
            Xc[i] = X[(blockIdx.x + k*gridDim.x) * blockDim.x + i];
    
        __syncthreads();
    
        int middle = SPACING + threadIdx.x;
    
        float Bvalue;
        Bvalue =    A.DIA[HEPT(3)][t]   *Xc[threadIdx.x]
                +   A.DIA[HEPT(2)][t]   *Xc[middle - 2]
                +   A.DIA[HEPT(1)][t]   *Xc[middle - 1]
                +   A.DIA[HEPT(0)][t]   *Xc[middle]
                +   A.DIA[HEPT(-1)][t]  *Xc[middle + 1]
                +   A.DIA[HEPT(-2)][t]  *Xc[middle + 2]
                +   A.DIA[HEPT(-3)][t]  *Xc[middle + SPACING];
    
        B[t] = Bvalue;
        ++k;
    }
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

    
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

    for ( iter = 1; iter <= BLOCKSIZE; iter *= 4 ) {
        float elapsed=0;
        cudaEvent_t start, stop;

        
        curandSetPseudoRandomGeneratorSeed(gen, iter);

        /* Filling random floats in A and X to flush GPU caches */
        curandGenerateUniform(gen, X, N);
        curandGenerateUniform(gen, A.DIA[HEPT(0)],  N);
        curandGenerateUniform(gen, A.DIA[HEPT(-1)], N);
        curandGenerateUniform(gen, A.DIA[HEPT(1)],  N);
        curandGenerateUniform(gen, A.DIA[HEPT(-2)], N);
        curandGenerateUniform(gen, A.DIA[HEPT(2)],  N);
        curandGenerateUniform(gen, A.DIA[HEPT(-3)], N);
        curandGenerateUniform(gen, A.DIA[HEPT(3)],  N);

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);

        heptaMatrixMul <<<n/BLOCKSIZE/iter, BLOCKSIZE>>> ( A, X, B, n);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize (stop);

        cudaEventElapsedTime(&elapsed, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        printf("%f\t", elapsed);
    }
    

    for ( iter = 1; iter <= BLOCKSIZE; iter *= 4 ) {
        float elapsed=0;
        cudaEvent_t start, stop;

        curandSetPseudoRandomGeneratorSeed(gen, iter);

        /* Filling random floats in A and X to flush GPU caches */
        curandGenerateUniform(gen, X, N);
        curandGenerateUniform(gen, A.DIA[HEPT(0)],  N);
        curandGenerateUniform(gen, A.DIA[HEPT(-1)], N);
        curandGenerateUniform(gen, A.DIA[HEPT(1)],  N);
        curandGenerateUniform(gen, A.DIA[HEPT(-2)], N);
        curandGenerateUniform(gen, A.DIA[HEPT(2)],  N);
        curandGenerateUniform(gen, A.DIA[HEPT(-3)], N);
        curandGenerateUniform(gen, A.DIA[HEPT(3)],  N);

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);

        heptaMatrixMulSharedMem <<<n/BLOCKSIZE/iter, BLOCKSIZE>>> ( A, X, B, n);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize (stop);

        cudaEventElapsedTime(&elapsed, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        printf("%f\t", elapsed);
    }

    curandDestroyGenerator(gen);

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
    for ( iter = 1; iter <= BLOCKSIZE; iter *= 4 ) {
        printf("%d-GLOBL\t", iter);
    }
    for ( iter = 1; iter <= BLOCKSIZE; iter *= 4 ) {
        printf("%d-SHARE\t", iter);
    }
    printf("OPENMP\n");

    for ( n = BLOCKSIZE * 6000; n <= N; n += BLOCKSIZE * 500) {
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