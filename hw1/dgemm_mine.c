#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "kdgemm.h"

const char* dgemm_desc = "My awesome dgemm.";

//#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 4)
//#endif

// Memory for kernel operations
double* kernel_A = 0;
double* kernel_B = 0;
double* kernel_C = 0;

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double *A, const double *B, double *C)
{
	int i, j, k;

	if (M != KERNEL_M || N != KERNEL_N || K != KERNEL_P)
	{
		// Do basic method
		for (i = 0; i < M; ++i) {
			for (j = 0; j < N; ++j) {
				double cij = C[j*lda+i];
				for (k = 0; k < K; ++k) {
					cij += A[k*lda+i] * B[j*lda+k];
				}
				C[j*lda+i] = cij;
			}
		}
	}
	else
	{
		if (kernel_A == 0)
		{
			kernel_A = _mm_malloc(DIM_M * DIM_P * sizeof(double), 16);
			kernel_B = _mm_malloc(DIM_P * DIM_N * sizeof(double), 16);
			kernel_C = _mm_malloc(DIM_M * DIM_N * sizeof(double), 16);
		}

		// Copy optimization to kernel memory
		to_kdgemm_A(lda, A, kernel_A);
		to_kdgemm_B(lda, B, kernel_B);
		// Clear matrix C for accumulation
		memset(kernel_C, 0, KERNEL_M * KERNEL_N * sizeof(double));

		// Execute kernel
		kdgemm(kernel_A, kernel_B, kernel_C);

		// Copy back from kernel memory
		from_kdgemm_C(lda, kernel_C, C);
	}
}

void do_block(const int lda,
              const double *A, const double *B, double *C,
              const int i, const int j, const int k)
{
    const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
    const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
    const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);
    basic_dgemm(lda, M, N, K,
                A + i + k*lda, B + k + j*lda, C + i + j*lda);
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi) {
        const int i = bi * BLOCK_SIZE;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int j = bj * BLOCK_SIZE;
            for (bk = 0; bk < n_blocks; ++bk) {
                const int k = bk * BLOCK_SIZE;
                do_block(M, A, B, C, i, j, k);
            }
        }
    }
}

