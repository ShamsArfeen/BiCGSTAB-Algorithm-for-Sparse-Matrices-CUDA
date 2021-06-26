#include "stdio.h"
#include "stdlib.h"
#include "time.h"

#define SPACING 7
#define HEPT(a) ((a)+SPACING)
#define N 102400

long double A[N][2*SPACING+1], B[N], X[N];

void debug() {
    int i,j;
    for ( i = 0; i < N; ++i) {
        for ( j = 0; j <= 2*SPACING; ++j)
            printf("%f, ", A[i][j]);
        printf("\n");
    }
    printf("\n");
}

void solve() {
    int i, j, r;
    for ( i = 0; i < N-1; ++i) {
        int maxJ = (i+SPACING) < (N-1) ? (i+SPACING) : (N-1);
        for ( j = i+1; j <= maxJ; ++j) {
            long double constant = A[j][HEPT(i-j)] / A[i][HEPT(0)];
            for ( r = 0; r <= SPACING; ++r) {
                A[j][HEPT(i-j+r)] -= constant * A[i][HEPT(r)];
            }
            B[j] -= constant * B[i];
        }
    }
    for ( i = N-1; i >= 0; --i) {
        X[i] = B[i];
        int maxR = (N-i-1) > SPACING ? SPACING : (N-i-1);
        for ( r = 1; r <= maxR; ++r) {
            X[i] -= A[i][HEPT(r)] * X[i+r];
        }
        X[i] /= A[i][HEPT(0)];
    }
}

int verify() {
    int i, r;
    long double Bval;
    for ( i = 0; i < N; ++i) {
        Bval = 0;
        int minR = i > SPACING ? -SPACING : -i;
        int maxR = i < N-1-SPACING ? SPACING : N-1-i;
        for ( r = minR; r <= maxR; ++r) {
            Bval += A[i][HEPT(r)] * X[i+r];
        }
        if ( abs(Bval - B[i]) > 1e-03)
            printf("Error [%d] %f %f\n", i, Bval, B[i]);
    }
    return 1;
}

void initialize() {
    int i,j;
    srand( time(0));
    for ( i = 0 ; i < N ; ++i) {
        for ( j = 0; j <= 2*SPACING; ++j)
            A[i][j] = 0;
        A[i][HEPT(0)] = rand() / (long double)RAND_MAX;
        A[i][HEPT(1)] = rand() / (long double)RAND_MAX;
        A[i][HEPT(2)] = rand() / (long double)RAND_MAX;
        A[i][HEPT(-1)] = rand() / (long double)RAND_MAX;
        A[i][HEPT(-2)] = rand() / (long double)RAND_MAX;
        A[i][HEPT(SPACING)] = rand() / (long double)RAND_MAX;
        A[i][HEPT(-SPACING)] = rand() / (long double)RAND_MAX;

        B[i] = rand() / (long double)RAND_MAX;
    }
}

int main() {
    initialize() ;
    solve();

    initialize() ;
    verify();
    printf("return\n");

    return 0;
}