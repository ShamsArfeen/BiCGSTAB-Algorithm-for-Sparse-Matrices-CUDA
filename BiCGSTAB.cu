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

double *X, *B, *V, *P, *R, *AX, *R0, *D, *D2, *H, *S, *T;
double *omega, *rho, *beta, *alpha, *temp, *Error;

struct twodouble {
    double a, b;
};

struct sparseMatrix {
    double* diagonal[7];
} A;

__global__ void heptaMatrixMul ( 
    struct sparseMatrix A, 
    double *X, 
    double *B) {

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

    double Bvalue;
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
    double *V1, 
    double *V2) {

    int thrId = blockIdx.x * blockDim.x + threadIdx.x;
    V1[thrId] = V2[thrId];
}

__global__ void vecAdd2 ( 
    double *R, 
    double *V1, 
    double *V2, 
    double *C, 
    double *V3, 
    double *D) {

    int thrId = blockIdx.x * blockDim.x + threadIdx.x;
    R[thrId] = V1[thrId] + (*C) * (V2[thrId] - (*D) * V3[thrId]);
}

__global__ void vecAdd ( 
    double *R, 
    double *V1, 
    double *V2, 
    double *C, 
    int sign) {

    int thrId = blockIdx.x * blockDim.x + threadIdx.x;
    R[thrId] = V1[thrId] + sign * (*C) * V2[thrId];
}

__global__ void vecMultiply ( 
    double *D, 
    double *V1, 
    double *V2) {

    int thrId = blockIdx.x * blockDim.x + threadIdx.x;
    D[thrId] = V1[thrId] * V2[thrId];
}

__global__ void getArraySum(
    double *A, 
    double *B, 
    int num, 
    double *res) {

    int thrId = blockIdx.x * blockDim.x + threadIdx.x;
    double* pair;
    if ( num == 2) {
        pair = B;
        A[thrId] =  pair[0] + pair[1];
        *res =      pair[0] + pair[1];
    }
    else if ( 2*thrId < num) {
        if ( 2*thrId + 1 == num) {
            A[thrId] =      B[2 * thrId];
        }
        else {
            pair = (B + 2*thrId);
            A[thrId] =      pair[0] + pair[1];
        }
    }
}

__global__ void getbetarho( 
    double *omega, 
    double *rho, 
    double *beta, 
    double *alpha, 
    double *temp) {

    *beta = (*temp / *rho) *
        ( *alpha / *omega);
    *rho = *temp;
}

__global__ void initomegarhoalpha( 
    double *omega, 
    double *rho, 
    double *alpha) {

    *rho = 1;
    *omega = 1;
    *alpha = 1;
}

__global__ void getalpha( 
    double *alpha, 
    double *rho, 
    double *temp) {

    *alpha = *rho / *temp;
}

__global__ void getomega( 
    double *omega, 
    double *temp) {

    *omega = *omega / *temp;
}

__global__ void gettemp( 
    double *temp, 
    double *omega) {

    *temp *= -(*omega);
}

void getDotProduct( double *res) {
    int num = N;
    while (num > 1) {
        getArraySum <<<num/BLOCKSIZE+1, BLOCKSIZE>>>(D2, D, num, res);
        double *Ptr = D2;
        D2 = D;
        D = Ptr;
        num = num/2 + (num%2);
    }
}


void freeResources ();
void initialize ();

int main ( int argc, char *argv[]) {
    initialize ();
    
    double Err;
    float elapsed;  
    int i; 

    heptaMatrixMul <<<N/BLOCKSIZE, BLOCKSIZE>>> ( A, X, AX);
    initomegarhoalpha <<<1, 1>>>( omega, rho, alpha);
    vecAdd <<<N/BLOCKSIZE, BLOCKSIZE>>>( R, B, AX, omega, -1);
    vecCopy <<<N/BLOCKSIZE, BLOCKSIZE>>>( R0, R);
    vecMultiply <<<N/BLOCKSIZE, BLOCKSIZE>>>( D, R, R);
    getDotProduct( temp);

    cudaEvent_t start, stop;
    
    for ( i = 0; i < 500; ++i) {
    
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        getbetarho <<<1, 1>>>( omega, rho, beta, alpha, temp);
        vecAdd2 <<<N/BLOCKSIZE, BLOCKSIZE>>>( P, R, P, beta, V, omega);
        heptaMatrixMul <<<N/BLOCKSIZE, BLOCKSIZE>>> ( A, P, V);
        vecMultiply <<<N/BLOCKSIZE, BLOCKSIZE>>>( D, R0, V);
        getDotProduct( temp);
        getalpha <<<1, 1>>>( alpha, rho, temp);
        vecAdd <<<N/BLOCKSIZE, BLOCKSIZE>>>( H, X, P, alpha, 1);
        vecAdd <<<N/BLOCKSIZE, BLOCKSIZE>>>( S, R, V, alpha, -1);
        heptaMatrixMul <<<N/BLOCKSIZE, BLOCKSIZE>>> ( A, S, T);
        vecMultiply <<<N/BLOCKSIZE, BLOCKSIZE>>>( D, T, S);
        getDotProduct( omega);
        vecMultiply <<<N/BLOCKSIZE, BLOCKSIZE>>>( D, T, T);
        getDotProduct( temp);
        getomega <<<1, 1>>>( omega, temp);
        vecAdd <<<N/BLOCKSIZE, BLOCKSIZE>>>( X, H, S, omega, 1);
        vecAdd <<<N/BLOCKSIZE, BLOCKSIZE>>>( R, S, T, omega, -1);
        vecMultiply <<<N/BLOCKSIZE, BLOCKSIZE>>>( D, R0, T);
        getDotProduct( temp);
        gettemp <<<1, 1>>>( temp, omega);
        vecMultiply <<<N/BLOCKSIZE, BLOCKSIZE>>>( D, R, R);
        getDotProduct( Error);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize (stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);


        if (i % 10 == 0) {
            cudaDeviceSynchronize();
            cudaMemcpy(&Err, Error, sizeof(double), cudaMemcpyDeviceToHost);
            printf("time : %f ms\terr : %f\n", elapsed, Err);
        }
    }

    freeResources ();
    return 0;
}

 /* Random doubles generation of Hepta Matrix using lib cuRAND */

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
    cudaMalloc(&X, N*sizeof(double));

    /* Resultant vector B */
    cudaMalloc(&B, N* sizeof(double));

    
    cudaMalloc(&omega, sizeof(double));
    cudaMalloc(&rho, sizeof(double));
    cudaMalloc(&beta, sizeof(double));
    cudaMalloc(&alpha, sizeof(double));
    cudaMalloc(&temp, sizeof(double));
    cudaMalloc(&Error, sizeof(double));

    cudaMalloc(&V, N* sizeof(double));
    cudaMalloc(&R, N* sizeof(double));
    cudaMalloc(&P, N* sizeof(double));
    cudaMalloc(&AX, N* sizeof(double));
    cudaMalloc(&R0, N* sizeof(double));
    cudaMalloc(&D, N* sizeof(double));
    cudaMalloc(&D2, N* sizeof(double));
    cudaMalloc(&H, N* sizeof(double));
    cudaMalloc(&S, N* sizeof(double));
    cudaMalloc(&T, N* sizeof(double));

    /* Extended diagonalgonals of Hepta-Mat A */
    cudaMalloc(&A.diagonal[HEPT(0)],     N* sizeof(double));
    cudaMalloc(&A.diagonal[HEPT(-1)],    N* sizeof(double));
    cudaMalloc(&A.diagonal[HEPT(1)],     N* sizeof(double));
    cudaMalloc(&A.diagonal[HEPT(-2)],    N* sizeof(double));
    cudaMalloc(&A.diagonal[HEPT(2)],     N* sizeof(double));
    cudaMalloc(&A.diagonal[HEPT(-3)],    N* sizeof(double));
    cudaMalloc(&A.diagonal[HEPT(3)],     N* sizeof(double));


    srand( time(0));
    /* Filling random doubles in A and X */
    double *Array = (double *) malloc(N* sizeof(double));
    for ( int i = 0; i < N; ++i) Array[i] = rand() / (double)RAND_MAX ;
    cudaMemcpy(A.diagonal[HEPT(1)], Array, N*sizeof(double), cudaMemcpyHostToDevice);
    for ( int i = 0; i < N; ++i) Array[i] = rand() / (double)RAND_MAX ;
    cudaMemcpy(A.diagonal[HEPT(2)], Array, N*sizeof(double), cudaMemcpyHostToDevice);
    for ( int i = 0; i < N; ++i) Array[i] = rand() / (double)RAND_MAX ;
    cudaMemcpy(A.diagonal[HEPT(3)], Array, N*sizeof(double), cudaMemcpyHostToDevice);
    for ( int i = 0; i < N; ++i) Array[i] = rand() / (double)RAND_MAX ;
    cudaMemcpy(A.diagonal[HEPT(-3)], Array, N*sizeof(double), cudaMemcpyHostToDevice);
    for ( int i = 0; i < N; ++i) Array[i] = rand() / (double)RAND_MAX ;
    cudaMemcpy(A.diagonal[HEPT(-2)], Array, N*sizeof(double), cudaMemcpyHostToDevice);
    for ( int i = 0; i < N; ++i) Array[i] = rand() / (double)RAND_MAX ;
    cudaMemcpy(A.diagonal[HEPT(-1)], Array, N*sizeof(double), cudaMemcpyHostToDevice);
    for ( int i = 0; i < N; ++i) Array[i] = rand() / (double)RAND_MAX ;
    cudaMemcpy(A.diagonal[HEPT(0)], Array, N*sizeof(double), cudaMemcpyHostToDevice);
    
    for ( int i = 0; i < N; ++i) Array[i] = rand() / (double)RAND_MAX ;
    cudaMemcpy(B, Array, N*sizeof(double), cudaMemcpyHostToDevice);
    for ( int i = 0; i < N; ++i) Array[i] = rand() / (double)RAND_MAX ;
    cudaMemcpy(X, Array, N*sizeof(double), cudaMemcpyHostToDevice);
    
    
    for ( int i = 0; i < N; ++i) Array[i] = 0;
    cudaMemcpy(V, Array, sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(P, Array, sizeof(double) * N, cudaMemcpyHostToDevice);
    
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