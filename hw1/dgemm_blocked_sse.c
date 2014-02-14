const char* dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 16)
#endif

double kernel_A[2 * BLOCK_SIZE] __attribute__ ((aligned (BYTE_ALIGNMENT)));
double kernel_B[2 * BLOCK_SIZE] __attribute__ ((aligned (BYTE_ALIGNMENT)));
double kernel_C[2 * 2] __attribute__ ((aligned (BYTE_ALIGNMENT)));

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
    for (i = 0; i < M; i += 2) {
        for (j = 0; j < N; j += 2) {
			to_kdgemm_A_sized(lda, A + i, kernel_A, min(2, M-i), K);
			to_kdgemm_B_sized(lda, B + lda * j, kernel_B, K, min(2, N-j));
			to_kdgemm_C_sized(lda, C + lda * j + i, kernel_C, min(2, M-i), min(2, N-j));

			// Perform kernel operations
			kdgemm(kernel_A, kernel_B, kernel_C);

				// Copy results out of kernel memory buffers
			from_kdgemm_C_sized(M, kernel_C, C + lda * j + i, min(2, M-i), min(2, N-j));
        }
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

