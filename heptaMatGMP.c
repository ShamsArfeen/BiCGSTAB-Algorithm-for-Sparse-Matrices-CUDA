#include "stdio.h"
#include "stdlib.h"
#include "time.h"

#include "gmp.h"

#define SPACING 7
#define HEPT(a) ((a)+SPACING)
#define N (1024 * 1024)

mpf_t A[N][2*SPACING+1], B[N], X[N];

void debug() {
    int i,j;
    for ( i = 0; i < N; ++i) {
        for ( j = 0; j <= 2*SPACING; ++j)
            gmp_printf("%Ff, ", A[i][j]);
        printf("\n");
    }
    printf("\n");
}

void solve() {
    int i, j, r;
    mpf_t constant, product;
    mpf_init(constant);
    mpf_init(product);
    for ( i = 0; i < N-1; ++i) {
        int maxJ = (i+SPACING) < (N-1) ? (i+SPACING) : (N-1);
        for ( j = i+1; j <= maxJ; ++j) {
            mpf_div( constant, A[j][HEPT(i-j)], A[i][HEPT(0)]);
            for ( r = 0; r <= SPACING; ++r) {
                mpf_mul( product, constant, A[i][HEPT(r)]);
                mpf_sub( A[j][HEPT(i-j+r)], A[j][HEPT(i-j+r)], product);
            }
            mpf_mul( product, constant, B[i]);
            mpf_sub( B[j], B[j], product);
        }
    }
    for ( i = N-1; i >= 0; --i) {
        mpf_set(X[i], B[i]);
        int maxR = (N-i-1) > SPACING ? SPACING : (N-i-1);
        for ( r = 1; r <= maxR; ++r) {
            mpf_mul( product, A[i][HEPT(r)], X[i+r]);
            mpf_sub( X[i], X[i], product);
        }
        mpf_div( X[i], X[i], A[i][HEPT(0)]);
    }
    mpf_clear(constant);
    mpf_clear(product);
}

int verify() {
    int i, r;
    mpf_t bval, temp, zero, err;
    mpf_init(bval);
    mpf_init(temp);
    mpf_init(zero);
    mpf_init(err);
    mpf_set_str(zero, "0.0", 10);
    for ( i = 0; i < N; ++i) {
        mpf_set(bval, zero);
        int minR = i > SPACING ? -SPACING : -i;
        int maxR = i < N-1-SPACING ? SPACING : N-1-i;
        for ( r = minR; r <= maxR; ++r) {
            mpf_mul( temp, A[i][HEPT(r)], X[i+r]);
            mpf_add( bval, bval, temp);
        }
        mpf_sub( temp, bval, B[i]);
        mpf_mul( temp, temp, temp);
        mpf_add( err, err, temp);
        //if ( abs(Bval - B[i]) > 1e-03)
        //gmp_printf("Error [%d] %Ff %Ff\n", i, bval, B[i]);
    }
    gmp_printf("Error %Ff \n", err);
    return 1;
}

void initialize() {
    int i,j;
    srand( 3); // dont change this
    for ( i = 0 ; i < N ; ++i) {
        for ( j = 0; j <= 2*SPACING; ++j)
            mpf_init(A[i][j]) ;
        mpf_set_d(A[i][HEPT(0)], rand() / (double)RAND_MAX);
        mpf_set_d(A[i][HEPT(1)], rand() / (double)RAND_MAX);
        mpf_set_d(A[i][HEPT(2)], rand() / (double)RAND_MAX);
        mpf_set_d(A[i][HEPT(-1)], rand() / (double)RAND_MAX);
        mpf_set_d(A[i][HEPT(-2)], rand() / (double)RAND_MAX);
        mpf_set_d(A[i][HEPT(SPACING)], rand() / (double)RAND_MAX);
        mpf_set_d(A[i][HEPT(-SPACING)], rand() / (double)RAND_MAX);

	mpf_init(X[i]);
	mpf_init(B[i]);
        mpf_set_d(B[i], rand() / (double)RAND_MAX);
    }
}

void recover() {
    int i,j;
    srand( 3); // dont change this
    mpf_t zero;
    mpf_init( zero);
    for ( i = 0 ; i < N ; ++i) {
        for ( j = 0; j <= 2*SPACING; ++j)
            mpf_set(A[i][j], zero) ;
        mpf_set_d(A[i][HEPT(0)], rand() / (double)RAND_MAX);
        mpf_set_d(A[i][HEPT(1)], rand() / (double)RAND_MAX);
        mpf_set_d(A[i][HEPT(2)], rand() / (double)RAND_MAX);
        mpf_set_d(A[i][HEPT(-1)], rand() / (double)RAND_MAX);
        mpf_set_d(A[i][HEPT(-2)], rand() / (double)RAND_MAX);
        mpf_set_d(A[i][HEPT(SPACING)], rand() / (double)RAND_MAX);
        mpf_set_d(A[i][HEPT(-SPACING)], rand() / (double)RAND_MAX);

        mpf_set_d(B[i], rand() / (double)RAND_MAX);
    }
}

int main() {
    //mpf_set_default_prec(64);
    initialize() ;
    solve();

    recover() ;
    verify();
    printf("return\n");

    return 0;
}