#include <nmmintrin.h>
#include "kdgemm.h"
#include <stdio.h>

#define N KERNEL_N
#define M KERNEL_M
#define P KERNEL_P


/*
 * On the Nehalem architecture, shufpd and multiplication use the same port.
 * 32-bit integer shuffle is a different matter.  If we want to try to make
 * it as easy as possible for the compiler to schedule multiplies along
 * with adds, it therefore makes sense to abuse the integer shuffle
 * instruction.  See also
 *   http://locklessinc.com/articles/interval_arithmetic/
 */
#ifdef USE_SHUFPD
#  define swap_sse_doubles(a) _mm_shuffle_pd(a, a, 1)
#else
#  define swap_sse_doubles(a) (__m128d) _mm_shuffle_epi32((__m128i) a, 0x4e)
#endif

/*
 * The ktimer driver expects these variables to be set to whatever
 * the dimensions of a kernel multiply are.  It uses them both for
 * space allocation and for flop rate computations.
 */
int DIM_M=KERNEL_M;
int DIM_N=KERNEL_N;
int DIM_P=KERNEL_P;

/*
 * Block matrix multiply kernel (simple fixed-size case).
 * Use restrict to tell the compiler there is no aliasing,
 * and inform the compiler of alignment constraints.
 */
void kdgemm(const double * restrict A,
            const double * restrict B,
            double * restrict C)
{
    // This is really implicit in using the aligned ops...
    __assume_aligned(A, 16);
    __assume_aligned(B, 16);
    __assume_aligned(C, 16);

    double temp = C[1];
	C[1] = C[3];
	C[3] = temp;

    // Load diagonal and off-diagonals
    __m128d cd = _mm_load_pd(C+0);
    __m128d co = _mm_load_pd(C+2);

    /*
    printf("Print A:\n");
    for (int i = 0; i < DIM_M; i++)
    {
      for (int j = 0; j < DIM_P; j++)
	printf("%g, ",A[i+j*DIM_M]);
      printf("\n");
    }

    printf("Print B:\n");
    for (int i = 0; i < DIM_P; i++)
    {
      for (int j = 0; j < DIM_N; j++)
	printf("%g, ",B[i+j*DIM_P]);
      printf("\n");
    }
    */
    /*
     * Do block dot product.  Each iteration adds the result of a two-by-two
     * matrix multiply into the accumulated 2-by-2 product matrix, which is
     * stored in the registers cd (diagonal part) and co (off-diagonal part).
     */
    for (int k = 0; k < P; k += 2) {

      __m128d a0 = _mm_load_pd(A+2*k+0);
      __m128d b0 = _mm_load_pd(B+2*k+0);
      __m128d td0 = _mm_mul_pd(a0, b0);
      __m128d bs0 = swap_sse_doubles(b0);
      __m128d to0 = _mm_mul_pd(a0, bs0);

      __m128d a1 = _mm_load_pd(A+2*k+2);
      __m128d b1 = _mm_load_pd(B+2*k+2);
      __m128d td1 = _mm_mul_pd(a1, b1);
      __m128d bs1 = swap_sse_doubles(b1);
      __m128d to1 = _mm_mul_pd(a1, bs1);

      __m128d td_sum = _mm_add_pd(td0, td1);
      __m128d to_sum = _mm_add_pd(to0, to1);

      cd = _mm_add_pd(cd, td_sum);
      co = _mm_add_pd(co, to_sum);
    }

    // Write back sum
    _mm_store_pd(C+0, cd);
    _mm_store_pd(C+2, co);

    temp = C[1];
    C[1] = C[3];
    C[3] = temp;

}

/*
 * Conversion routines that take a matrix block in column-major form
 * and put it into whatever form the kdgemm routine likes.
 */

void to_kdgemm_A(int ldA, const double* restrict A, double * restrict Ak)
{
    for (int j = 0; j < P; ++j)
       for (int i = 0; i < M; ++i)
           Ak[i+j*M] = A[i+j*ldA];
}

void to_kdgemm_B(int ldB, const double* restrict B, double * restrict Bk)
{
  for (int i = 0; i < P; ++i)
    for (int j = 0; j < N; ++j)
           Bk[j+i*N] = B[i+j*ldB];
}

void to_kdgemm_C(int ldC, const double* restrict C, double * restrict Ck)
{
    for (int j = 0; j < N; ++j)
       for (int i = 0; i < M; ++i)
           Ck[i+j*M] = C[i+j*ldC];
}

void from_kdgemm_C(int ldC, const double* restrict Ck, double * restrict C)
{
    for (int j = 0; j < N; ++j)
       for (int i = 0; i < M; ++i)
           C[i+j*ldC] = Ck[i+j*M];
}
