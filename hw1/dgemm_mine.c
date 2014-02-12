const char* dgemm_desc = "My awesome dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 16)
#endif

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
void inline basic_dgemm(const int lda, const int M, const int N, const int K,
                 	    const double *A, const double *B, double *C)
{
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N * lda; j += lda) {
            double cij = C[j+i];
#pragma unroll(4)
            for (k = 0; k < K * lda; k += lda) {
                cij += A[k+i] * B[j+k];
            }
            C[j+i] = cij;
        }
    }
}

void inline do_block(const int lda,
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
    int i, j, k;
    for (i = 0; i < n_blocks * BLOCK_SIZE; i += BLOCK_SIZE) {
        for (j = 0; j < n_blocks * BLOCK_SIZE; j += BLOCK_SIZE) {
            for (k = 0; k < n_blocks * BLOCK_SIZE; k += BLOCK_SIZE) {
                do_block(M, A, B, C, i, j, k);
            }
        }
    }
}

