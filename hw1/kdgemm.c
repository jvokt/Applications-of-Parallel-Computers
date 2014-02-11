#include <nmmintrin.h>

/*
 * Dimensions for a "kernel" multiply.  We use define statements in
 * order to make sure these are treated as compile-time constants
 * (which the optimizer likes)
 */
#define M 4
#define N 4
#define P 4

/*
 * The ktimer driver expects these variables to be set to whatever
 * the dimensions of a kernel multiply are.  It uses them both for
 * space allocation and for flop rate computations.
 */
int DIM_M=M;
int DIM_N=N;
int DIM_P=P;

/*
 * Block matrix multiply kernel (simple fixed-size case).
 * Use restrict to tell the compiler there is no aliasing,
 * and inform the compiler of alignment constraints.
 */
void kdgemm(const double * restrict A,
            const double * restrict B,
            double * restrict C)
{
    __assume_aligned(A, 16);
    __assume_aligned(B, 16);
    __assume_aligned(C, 16);

    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < P; ++k) {
            double bkj = B[k+j*P];
            for (int i = 0; i < M; ++i) {
                C[i+j*M] += A[i+k*M]*bkj;
            }
        }
    }
}

/*
 * Conversion routines that take a matrix block in column-major form
 * and put it into whatever form the kdgemm routine likes.
 */

void to_kdgemm_A(int ldA, const double* restrict A, double * restrict Ak)
{
    for (int j = 0; j < N; ++j)
       for (int i = 0; i < M; ++i)
           Ak[i+j*M] = A[i+j*ldA];
}

void to_kdgemm_B(int ldB, const double* restrict B, double * restrict Bk)
{
    for (int j = 0; j < N; ++j)
       for (int i = 0; i < M; ++i)
           Bk[i+j*P] = B[i+j*ldB];
}

void from_kdgemm_C(int ldC, const double* restrict Ck, double * restrict C)
{
    for (int j = 0; j < N; ++j)
       for (int i = 0; i < M; ++i)
           C[i+j*ldC] = Ck[i+j*M];
}
