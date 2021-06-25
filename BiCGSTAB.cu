#include "stdio.h"
#include "time.h"
#include "curand.h"

#define SPACING 7 
#define BLOCKSIZE 5 
#define N (BLOCKSIZE * 2) 
#define CACHE (2 * SPACING + BLOCKSIZE + 1)

/* SPACING : Offset b/w main and distant diagonal */
/* BLOCKSIZE : Threads per block, multiple of 32 */
/* N : Row/Column size */
/* CACHE : Size of shared memory per block */

#define HEPT(a) ((a)+3)

struct sparseMatrix {
    float* DIA[7];
} A;

float *X; // (N + 2*SPACING)
float *B, *V, *P, *R, *AX, *R0, *D, *D2, *H, *S, *T;

struct twofloat {
    float a, b;
};

__global__ void Assignment ( struct sparseMatrix A, float *X, float *B) {

    /* V1 = V2 */
    int thrId = blockIdx.x * blockDim.x + threadIdx.x;

    A.DIA[HEPT(0)][thrId] = thrId;
    
    A.DIA[HEPT(1)][thrId] = 0;
    A.DIA[HEPT(2)][thrId] = 0;
    A.DIA[HEPT(3)][thrId] = 0;
    A.DIA[HEPT(-1)][thrId] = 0;
    A.DIA[HEPT(-2)][thrId] = 0;
    A.DIA[HEPT(-3)][thrId] = 0;

    X[thrId] = 1;
    B[thrId] = 2;
}

__global__ void heptaMatrixMul ( struct sparseMatrix A, float *X, float *B) { // taking unextended vector X

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
    Bvalue = A.DIA[HEPT(3)][thrId] * zero1 * X[ind1 * zero1]
            + A.DIA[HEPT(2)][thrId] * zero2 *  X[ind2 * zero2]
            + A.DIA[HEPT(1)][thrId] * zero3 *  X[ind3 * zero3]
            + A.DIA[HEPT(0)][thrId] * zero4 *  X[ind4 * zero4]
            + A.DIA[HEPT(-1)][thrId] * zero5 * X[ind5 * zero5]
            + A.DIA[HEPT(-2)][thrId] * zero6 * X[ind6 * zero6]
            + A.DIA[HEPT(-3)][thrId] * zero7 * X[ind7 * zero7];

    B[thrId] = Bvalue;
}

__global__ void vecEqual ( float *V1, float *V2) {

    /* V1 = V2 */
    int thrId = blockIdx.x * blockDim.x + threadIdx.x;

    V1[thrId] = V2[thrId];
}

__global__ void vecScaledSum2 ( float *R, float *V1, float *V2, float C, float *V3, float D) {

    /* R = V1 + C.V2 + D.V3 */
    int thrId = blockIdx.x * blockDim.x + threadIdx.x;

    R[thrId] = V1[thrId] + C * V2[thrId] + D * V3[thrId];
}

__global__ void vecScaledSum ( float *R, float *V1, float *V2, float C) {

    /* R = V1 + C.V2 */
    int thrId = blockIdx.x * blockDim.x + threadIdx.x;

    R[thrId] = V1[thrId] + C * V2[thrId];
}

__global__ void vecDotArray ( float *D, float *V1, float *V2) {

    /* D[i] = V1[i] * V2[i] */
    int thrId = blockIdx.x * blockDim.x + threadIdx.x;

    D[thrId] = V1[thrId] * V2[thrId];
}

__global__ void vecDotArray2 ( float *D, float *V) {

    /* D[i] = V[i] * V[i] */
    int thrId = blockIdx.x * blockDim.x + threadIdx.x;

    D[thrId] = V[thrId] * V[thrId];
}

__global__ void vecAddReduc(float *a, float *b, int n) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( 2*tid < n) {
        if ( 2*tid+1 == n)
            a[tid] = b[2*tid];
        else {
            struct twofloat B;
            B = *(struct twofloat *)(b + 2*tid);
            a[tid] = B.a + B.b;
        }
    }
}

void freeResources ();
void initialize ();

int main ( int argc, char *argv[]) {
    initialize ();
    
    float *temp;
    float val;
    float *ZERO =(float*) malloc(sizeof(float) * N);
    int n, i = 0;

    for ( int j = 0; j < N; ++j)
        ZERO[j] = 0;
    cudaMemcpy(V, ZERO, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(P, ZERO, sizeof(float) * N, cudaMemcpyHostToDevice);
    

    heptaMatrixMul <<<N/BLOCKSIZE+1, BLOCKSIZE>>> ( A, X, AX);

    vecScaledSum <<<N/BLOCKSIZE, BLOCKSIZE>>>( R, B, AX, -1);
    
    vecDotArray <<<N/BLOCKSIZE, BLOCKSIZE>>>( D, R, R);

    n = N;
    AGAIN:
        vecAddReduc <<<n/BLOCKSIZE+1, BLOCKSIZE>>>(D2, D, n);
        temp = D2;
        D2 = D;
        D = temp;
        n = n/2 + (n%2);
    if (n > 1) goto AGAIN;
    cudaMemcpy(&val, D, sizeof(float), cudaMemcpyDeviceToHost);

    vecEqual <<<N/BLOCKSIZE, BLOCKSIZE>>>( R0, R);

    float phi = 1, alpha = 1, omega = 1;

    
    ITERATE:
        float phi2;
        vecDotArray <<<N/BLOCKSIZE, BLOCKSIZE>>>( D, R0, R);

        n = N;
        AGAIN1:
            vecAddReduc <<<n/BLOCKSIZE+1, BLOCKSIZE>>>(D2, D, n);
            temp = D2;
            D2 = D;
            D = temp;
            n = n/2 + (n%2);
        if (n > 1) goto AGAIN1;
        cudaDeviceSynchronize();
        cudaMemcpy(&phi2, D, sizeof(float), cudaMemcpyDeviceToHost);

        float beta = (phi2 / (float)phi) * (alpha / (float)omega);
        phi = phi2;

        vecScaledSum2 <<<N/BLOCKSIZE, BLOCKSIZE>>>( P, R, P, beta, V, -beta * omega);
        
        heptaMatrixMul <<<N/BLOCKSIZE, BLOCKSIZE>>> ( A, P, V);
        
        vecDotArray <<<N/BLOCKSIZE, BLOCKSIZE>>>( D, R0, V);
        n = N;
        AGAIN2:
            vecAddReduc <<<n/BLOCKSIZE+1, BLOCKSIZE>>>(D2, D, n);
            temp = D2;
            D2 = D;
            D = temp;
            n = n/2 + (n%2);
        if (n > 1) goto AGAIN2;
        cudaDeviceSynchronize();
        cudaMemcpy(&alpha, D, sizeof(float), cudaMemcpyDeviceToHost);
        alpha = phi / (float)alpha;

        vecScaledSum <<<N/BLOCKSIZE, BLOCKSIZE>>>( H, X, P, alpha);
        
        vecScaledSum <<<N/BLOCKSIZE, BLOCKSIZE>>>( S, R, V, -alpha);

        heptaMatrixMul <<<N/BLOCKSIZE, BLOCKSIZE>>> ( A, S, T);

        vecDotArray <<<N/BLOCKSIZE, BLOCKSIZE>>>( D, T, S);
        n = N;
        AGAIN3:
            vecAddReduc <<<n/BLOCKSIZE+1, BLOCKSIZE>>>(D2, D, n);
            temp = D2;
            D2 = D;
            D = temp;
            n = n/2 + (n%2);
        if (n > 1) goto AGAIN3;
        cudaDeviceSynchronize();
        cudaMemcpy(&omega, D, sizeof(float), cudaMemcpyDeviceToHost);
        
        vecDotArray2 <<<N/BLOCKSIZE, BLOCKSIZE>>>( D, T);

        float denom;
        n = N;
        AGAIN4:
            vecAddReduc <<<n/BLOCKSIZE+1,BLOCKSIZE>>>(D2, D, n);
            temp = D2;
            D2 = D;
            D = temp;
            n = n/2 + (n%2);
        if (n > 1) goto AGAIN4;
        cudaDeviceSynchronize();
        cudaMemcpy(&denom, D, sizeof(float), cudaMemcpyDeviceToHost);
        omega = omega / (float)denom;
        
        vecScaledSum <<<N/BLOCKSIZE, BLOCKSIZE>>>( X, H, S, omega);

        vecScaledSum <<<N/BLOCKSIZE, BLOCKSIZE>>>( R, S, T, -omega);

        vecDotArray <<<N/BLOCKSIZE, BLOCKSIZE>>>( D, R, R);
        float Err;
        n = N;
        AGAIN5:
            vecAddReduc <<<n/BLOCKSIZE+1, BLOCKSIZE>>>(D2, D, n);
            temp = D2;
            D2 = D;
            D = temp;
            n = n/2 + (n%2);
        if (n > 1) goto AGAIN5;
        cudaDeviceSynchronize();
        cudaMemcpy(&Err, D, sizeof(float), cudaMemcpyDeviceToHost);

        if (i %10 ==0) printf("err : %f\n", Err);
        ++i;
    if (i < 100) goto ITERATE;
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
    cudaMalloc(&X, N*sizeof(float));

    /* Resultant vector B */
    cudaMalloc(&B, N* sizeof(float));

    
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
    curandGenerateUniform(gen, B, N);
    curandGenerateUniform(gen, A.DIA[HEPT(0)],  N);
    curandGenerateUniform(gen, A.DIA[HEPT(-1)], N);
    curandGenerateUniform(gen, A.DIA[HEPT(1)],  N);
    curandGenerateUniform(gen, A.DIA[HEPT(-2)], N);
    curandGenerateUniform(gen, A.DIA[HEPT(2)],  N);
    curandGenerateUniform(gen, A.DIA[HEPT(-3)], N);
    curandGenerateUniform(gen, A.DIA[HEPT(3)],  N);

}