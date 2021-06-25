#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include "curand.h"

#define SPACING 7 
#define BLOCKSIZE 1024 
#define N (BLOCKSIZE * 1024) 
#define CACHE (2 * SPACING + BLOCKSIZE + 1)
#define HEPT(a) ((a)+3)

/* SPACING : Offset b/w main and distant diagonalgonal */
/* BLOCKSIZE : Threads per block, multiple of 32 */
/* N : Row/Column size */
/* CACHE : Size of shared memory per block */

float *X, *B, *V, *P, *R, *AX, *R0, *D, *D2, *H, *S, *T;
float *Omega, *Rho, *Beta, *Alpha, *Temp, *Error;

struct twofloat {
    float a, b;
};

struct sparseMatrix {
    float* diagonal[7];
} A;

__global__ void heptaMatrixMul ( 
    struct sparseMatrix A, 
    float *X, 
    float *B) {

    /* B = A.X */
    int thrId = blockIdx.x * blockDim.x + threadIdx.x;

    // taking unextended vector X
    int middle = thrId; // SPACING + thrId;

    int ind1 = middle - SPACING;
    int ind2 = middle - 2;
    int ind3 = middle - 1;
    int ind4 = middle;
    int ind5 = middle + 1;
    int ind6 = middle + 2;
    int ind7 = middle + SPACING;

    int zero1 = (ind1 >= 0) && (thrId >= SPACING);
    int zero2 = (ind2 >= 0) && (thrId >= 2);
    int zero3 = (ind3 >= 0) && (thrId >= 1);
    int zero4 = (ind4 >= 0);
    int zero5 = (ind5 >= 0) && (thrId < (N-1));
    int zero6 = (ind6 >= 0) && (thrId < (N-2));
    int zero7 = (ind7 >= 0) && (thrId < (N-SPACING));

    float Bvalue;
    Bvalue = A.diagonal[HEPT(3)][thrId] * zero1 * X[ind1 * zero1]
            + A.diagonal[HEPT(2)][thrId] * zero2 *  X[ind2 * zero2]
            + A.diagonal[HEPT(1)][thrId] * zero3 *  X[ind3 * zero3]
            + A.diagonal[HEPT(0)][thrId] * zero4 *  X[ind4 * zero4]
            + A.diagonal[HEPT(-1)][thrId] * zero5 * X[ind5 * zero5]
            + A.diagonal[HEPT(-2)][thrId] * zero6 * X[ind6 * zero6]
            + A.diagonal[HEPT(-3)][thrId] * zero7 * X[ind7 * zero7];

    B[thrId] = Bvalue;
}

__global__ void vecCopy ( 
    float *V1, 
    float *V2) {

    int thrId = blockIdx.x * blockDim.x + threadIdx.x;
    V1[thrId] = V2[thrId];
}

__global__ void vecAdd2 ( 
    float *R, 
    float *V1, 
    float *V2, 
    float *C, 
    float *V3, 
    float *D) {

    int thrId = blockIdx.x * blockDim.x + threadIdx.x;
    R[thrId] = V1[thrId] + (*C) * (V2[thrId] - (*D) * V3[thrId]);
}

__global__ void vecAdd ( 
    float *R, 
    float *V1, 
    float *V2, 
    float *C, 
    int sign) {

    int thrId = blockIdx.x * blockDim.x + threadIdx.x;
    R[thrId] = V1[thrId] + sign * (*C) * V2[thrId];
}

__global__ void vecMultiply ( 
    float *D, 
    float *V1, 
    float *V2) {

    int thrId = blockIdx.x * blockDim.x + threadIdx.x;
    D[thrId] = V1[thrId] * V2[thrId];
}

__global__ void getArraySum(
    float *A, 
    float *B, 
    int num, 
    float *res) {

    int thrId = blockIdx.x * blockDim.x + threadIdx.x;
    float* pair;
    if ( num == 2) {
        pair = B;
        A[thrId] = pair[0] + pair[1];
        *res = pair[0] + pair[1];
    }
    else if ( 2 * thrId < num) {
        if ( 2 * thrId + 1 == num) {
            A[thrId] = B[2 * thrId];
        }
        else {
            pair = (B + 2 * thrId);
            A[thrId] = pair[0] + pair[1];
        }
    }
}

__global__ void getBetaRho( 
    float *Omega, 
    float *Rho, 
    float *Beta, 
    float *Alpha, 
    float *Temp) {

    *Beta = (*Temp / *Rho) *
        ( *Alpha / *Omega);
    *Rho = *Temp;
}

__global__ void initOmegaRhoAlpha( 
    float *Omega, 
    float *Rho, 
    float *Alpha) {

    *Rho = 1;
    *Omega = 1;
    *Alpha = 1;
}

__global__ void getAlpha( 
    float *Alpha, 
    float *Rho, 
    float *Temp) {

    *Alpha = *Rho / *Temp;
}

__global__ void getOmega( 
    float *Omega, 
    float *Temp) {

    *Omega = *Omega / *Temp;
}

void getDotProduct( float *res) {
    int num = N;
    while (num > 1) {
        getArraySum <<<num/BLOCKSIZE+1, BLOCKSIZE>>>(D2, D, num, res);
        float *Ptr = D2;
        D2 = D;
        D = Ptr;
        num = num/2 + (num%2);
    }
}


void freeResources ();
void initialize ();

int main ( int argc, char *argv[]) {
    initialize ();
    
    float Err, elapsed;  
    int i; 

    heptaMatrixMul <<<N/BLOCKSIZE, BLOCKSIZE>>> ( A, X, AX);
    initOmegaRhoAlpha <<<1, 1>>>( Omega, Rho, Alpha);
    vecAdd <<<N/BLOCKSIZE, BLOCKSIZE>>>( R, B, AX, Omega, -1);
    vecMultiply <<<N/BLOCKSIZE, BLOCKSIZE>>>( D, R, R);
    vecCopy <<<N/BLOCKSIZE, BLOCKSIZE>>>( R0, R);

    cudaEvent_t start, stop;
    
    for ( i = 0; i < 100; ++i) {
    
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        vecMultiply <<<N/BLOCKSIZE, BLOCKSIZE>>>( D, R0, R);
        getDotProduct( Temp);
        getBetaRho <<<1, 1>>>( Omega, Rho, Beta, Alpha, Temp);
        vecAdd2 <<<N/BLOCKSIZE, BLOCKSIZE>>>( P, R, P, Beta, V, Omega);
        heptaMatrixMul <<<N/BLOCKSIZE, BLOCKSIZE>>> ( A, P, V);
        vecMultiply <<<N/BLOCKSIZE, BLOCKSIZE>>>( D, R0, V);
        getDotProduct( Temp);
        getAlpha <<<1, 1>>>( Alpha, Rho, Temp);
        vecAdd <<<N/BLOCKSIZE, BLOCKSIZE>>>( H, X, P, Alpha, 1);
        vecAdd <<<N/BLOCKSIZE, BLOCKSIZE>>>( S, R, V, Alpha, -1);
        heptaMatrixMul <<<N/BLOCKSIZE, BLOCKSIZE>>> ( A, S, T);
        vecMultiply <<<N/BLOCKSIZE, BLOCKSIZE>>>( D, T, S);
        getDotProduct( Omega);
        vecMultiply <<<N/BLOCKSIZE, BLOCKSIZE>>>( D, T, T);
        getDotProduct( Temp);
        getOmega <<<1, 1>>>( Omega, Temp);
        vecAdd <<<N/BLOCKSIZE, BLOCKSIZE>>>( X, H, S, Omega, 1);
        vecAdd <<<N/BLOCKSIZE, BLOCKSIZE>>>( R, S, T, Omega, -1);
        vecMultiply <<<N/BLOCKSIZE, BLOCKSIZE>>>( D, R, R);
        getDotProduct( Error);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize (stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);


        if (i % 10 == 0) {
            cudaDeviceSynchronize();
            cudaMemcpy(&Err, Error, sizeof(float), cudaMemcpyDeviceToHost);
            printf("time : %f\terr : %f\n", elapsed, Err);
        }
    }

    freeResources ();
    return 0;
}

 /* Random floats generation of Hepta Matrix using lib cuRAND */

void freeResources() {
    cudaFree(T);
    cudaFree(S);
    cudaFree(P);
    cudaFree(V);
    cudaFree(H);
    cudaFree(D);
    cudaFree(D2);
    cudaFree(R);
    cudaFree(R0);
    cudaFree(AX);

    cudaFree(X);
    cudaFree(B);
    cudaFree( A.diagonal[HEPT(0)]);
    cudaFree( A.diagonal[HEPT(-1)]);
    cudaFree( A.diagonal[HEPT(1)]);
    cudaFree( A.diagonal[HEPT(-2)]);
    cudaFree( A.diagonal[HEPT(2)]);
    cudaFree( A.diagonal[HEPT(-3)]);
    cudaFree( A.diagonal[HEPT(3)]);
}

void initialize() {

    /* AX = B */

    /* Extended vector X */
    cudaMalloc(&X, N*sizeof(float));

    /* Resultant vector B */
    cudaMalloc(&B, N* sizeof(float));

    
    cudaMalloc(&Omega, sizeof(float));
    cudaMalloc(&Rho, sizeof(float));
    cudaMalloc(&Beta, sizeof(float));
    cudaMalloc(&Alpha, sizeof(float));
    cudaMalloc(&Temp, sizeof(float));
    cudaMalloc(&Error, sizeof(float));

    cudaMalloc(&V, N* sizeof(float));
    cudaMalloc(&R, N* sizeof(float));
    cudaMalloc(&P, N* sizeof(float));
    cudaMalloc(&AX, N* sizeof(float));
    cudaMalloc(&R0, N* sizeof(float));
    cudaMalloc(&D, N* sizeof(float));
    cudaMalloc(&D2, N* sizeof(float));
    cudaMalloc(&H, N* sizeof(float));
    cudaMalloc(&S, N* sizeof(float));
    cudaMalloc(&T, N* sizeof(float));

    /* Extended diagonalgonals of Hepta-Mat A */
    cudaMalloc(&A.diagonal[HEPT(0)],     N* sizeof(float));
    cudaMalloc(&A.diagonal[HEPT(-1)],    N* sizeof(float));
    cudaMalloc(&A.diagonal[HEPT(1)],     N* sizeof(float));
    cudaMalloc(&A.diagonal[HEPT(-2)],    N* sizeof(float));
    cudaMalloc(&A.diagonal[HEPT(2)],     N* sizeof(float));
    cudaMalloc(&A.diagonal[HEPT(-3)],    N* sizeof(float));
    cudaMalloc(&A.diagonal[HEPT(3)],     N* sizeof(float));

    /* Filling random floats in A and X */
    float *Array = (float *) malloc(N* sizeof(float));
    for ( int i = 0; i < N; ++i) Array[i] = rand() / (float)RAND_MAX / (float)N;
    cudaMemcpy(A.diagonal[HEPT(1)], Array, N*sizeof(float), cudaMemcpyHostToDevice);
    for ( int i = 0; i < N; ++i) Array[i] = rand() / (float)RAND_MAX / (float)N;
    cudaMemcpy(A.diagonal[HEPT(2)], Array, N*sizeof(float), cudaMemcpyHostToDevice);
    for ( int i = 0; i < N; ++i) Array[i] = rand() / (float)RAND_MAX / (float)N;
    cudaMemcpy(A.diagonal[HEPT(3)], Array, N*sizeof(float), cudaMemcpyHostToDevice);
    for ( int i = 0; i < N; ++i) Array[i] = rand() / (float)RAND_MAX / (float)N;
    cudaMemcpy(A.diagonal[HEPT(-3)], Array, N*sizeof(float), cudaMemcpyHostToDevice);
    for ( int i = 0; i < N; ++i) Array[i] = rand() / (float)RAND_MAX / (float)N;
    cudaMemcpy(A.diagonal[HEPT(-2)], Array, N*sizeof(float), cudaMemcpyHostToDevice);
    for ( int i = 0; i < N; ++i) Array[i] = rand() / (float)RAND_MAX / (float)N;
    cudaMemcpy(A.diagonal[HEPT(-1)], Array, N*sizeof(float), cudaMemcpyHostToDevice);
    for ( int i = 0; i < N; ++i) Array[i] = rand() / (float)RAND_MAX / (float)N;
    cudaMemcpy(A.diagonal[HEPT(0)], Array, N*sizeof(float), cudaMemcpyHostToDevice);
    
    for ( int i = 0; i < N; ++i) Array[i] = rand() / (float)RAND_MAX / (float)N;
    cudaMemcpy(B, Array, N*sizeof(float), cudaMemcpyHostToDevice);
    for ( int i = 0; i < N; ++i) Array[i] = rand() / (float)RAND_MAX / (float)N;
    cudaMemcpy(X, Array, N*sizeof(float), cudaMemcpyHostToDevice);
    
    
    for ( int i = 0; i < N; ++i) Array[i] = 0;
    cudaMemcpy(V, Array, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(P, Array, sizeof(float) * N, cudaMemcpyHostToDevice);
    
    free(Array);

    /*curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    curandGenerateUniform(gen, X, N);
    curandGenerateUniform(gen, B, N);
    curandGenerateUniform(gen, A.diagonal[HEPT(0)],  N);
    curandGenerateUniform(gen, A.diagonal[HEPT(-1)], N);
    curandGenerateUniform(gen, A.diagonal[HEPT(1)],  N);
    curandGenerateUniform(gen, A.diagonal[HEPT(-2)], N);
    curandGenerateUniform(gen, A.diagonal[HEPT(2)],  N);
    curandGenerateUniform(gen, A.diagonal[HEPT(-3)], N);
    curandGenerateUniform(gen, A.diagonal[HEPT(3)],  N);
    curandDestroyGenerator(gen);*/
}