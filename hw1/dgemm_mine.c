#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "kdgemm.h"

const char* dgemm_desc = "My awesome dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 16)
#endif

/*******************************************************************************
 * Program constants
 */
#ifndef A_BLOCK_LEN
#define A_BLOCK_LEN 16
#endif

#ifndef C_BLOCK_LEN
#define C_BLOCK_LEN 16
#endif

#ifndef P_BLOCK_LEN
#define P_BLOCK_LEN 2
#endif

#ifndef K_BLOCK_LEN
#define K_BLOCK_LEN 2
#endif

/*
 #ifndef L1_BLOCK_SIZE
 #define L1_BLOCK_SIZE KERNEL_P
 #endif

 #ifndef L2_BLOCK_MULT
 #define L2_BLOCK_MULT 2
 #endif

 #ifndef L2_BLOCK_SIZE
 #define L2_BLOCK_SIZE (L1_BLOCK_SIZE * L2_BLOCK_MULT)
 #endif

 #ifndef L3_BLOCK_MULT
 #define L3_BLOCK_MULT 4
 #endif

 #ifndef L3_BLOCK_SIZE
 #define L3_BLOCK_SIZE (L2_BLOCK_SIZE * L3_BLOCK_MULT)
 #endif
 */
/*******************************************************************************
 * Standard calculations
 */

// Calculate the number of blocks in a data segment for a given block width
#define CALC_NUM_BLOCKS(DATA_WIDTH, BLOCK_WIDTH) (DATA_WIDTH / BLOCK_WIDTH + (DATA_WIDTH % BLOCK_WIDTH ? 1 : 0))

// Calculate the width of the current block
#define CALC_CUR_BLOCK_WIDTH(CUR_START, BLOCK_WIDTH, DATA_WIDTH) (CUR_START + BLOCK_WIDTH > DATA_WIDTH ? DATA_WIDTH - CUR_START : BLOCK_WIDTH)

/*******************************************************************************
 * Memory segments
 */

double A_pack[C_BLOCK_LEN * A_BLOCK_LEN] __attribute__ ((aligned (MEM_ALIGN)));
//double B_pack[] ((aligned (MEM_ALIGN)));//[A_BLOCK_LEN*M] __attribute__ ((aligned (MEM_ALIGN)));
double B_pack[A_BLOCK_LEN * 32] __attribute__ ((aligned (MEM_ALIGN)));
double C_pack[C_BLOCK_LEN * 2] __attribute__ ((aligned (MEM_ALIGN)));
double A_kernel[A_BLOCK_LEN * 2] __attribute__ ((aligned (MEM_ALIGN)));
double B_kernel[2 * A_BLOCK_LEN] __attribute__ ((aligned (MEM_ALIGN)));
double C_kernel[2 * 2] __attribute__ ((aligned (MEM_ALIGN)));

/*
 // Memory for the 3 buffers for L3 level blocking, converted for zero padding and transposed
 double l3mem_A[L3_BLOCK_SIZE * L3_BLOCK_SIZE] __attribute__ ((aligned (MEM_ALIGN)));
 double l3mem_B[L3_BLOCK_SIZE * L3_BLOCK_SIZE] __attribute__ ((aligned (MEM_ALIGN)));
 double l3mem_C[L3_BLOCK_SIZE * L3_BLOCK_SIZE] __attribute__ ((aligned (MEM_ALIGN)));

 // Memory for the 3 buffers for L2 level blocking, already padded and transposed
 double l2mem_A[L2_BLOCK_SIZE * L2_BLOCK_SIZE] __attribute__ ((aligned (MEM_ALIGN)));
 double l2mem_B[L2_BLOCK_SIZE * L2_BLOCK_SIZE] __attribute__ ((aligned (MEM_ALIGN)));
 double l2mem_C[L2_BLOCK_SIZE * L2_BLOCK_SIZE] __attribute__ ((aligned (MEM_ALIGN)));

 // Memory for the 3 buffers for L1 level blocking, already padded and transposed
 double l1mem_A[L1_BLOCK_SIZE * L1_BLOCK_SIZE] __attribute__ ((aligned (MEM_ALIGN)));
 double l1mem_B[L1_BLOCK_SIZE * L1_BLOCK_SIZE] __attribute__ ((aligned (MEM_ALIGN)));
 double l1mem_C[L1_BLOCK_SIZE * L1_BLOCK_SIZE] __attribute__ ((aligned (MEM_ALIGN)));

 // Memory for the 3 buffers for kernel operations, already padded and transposed
 double kernel_A[KERNEL_M * KERNEL_P] __attribute__ ((aligned (MEM_ALIGN)));
 double kernel_B[KERNEL_N * KERNEL_P] __attribute__ ((aligned (MEM_ALIGN)));
 double kernel_C[KERNEL_M * KERNEL_N] __attribute__ ((aligned (MEM_ALIGN)));
 */
/*
 void square_dgemm(const int lda, const int M, const int N, const int K,
 const double *A, const double *B, double *C)
 {
 int i, j, k;
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
 */
void square_dgemm3(const int M, const double *A, const double *B, double *C) {
	int i, j, k;
	for (i = 0; i < M; ++i) {
		for (j = 0; j < M; ++j) {
			double cij = C[j * M + i];
			for (k = 0; k < M; ++k)
				cij += A[k * M + i] * B[j * M + k];
			C[j * M + i] = cij;
		}
	}
}

/*
 void do_block(const int lda,
 const double *A, const double *B, double *C,
 const int i, const int j, const int k)
 {
 const int M = K_BLOCK_LEN;
 const int N = P_BLOCK_LEN;
 const int K = A_BLOCK_LEN;
 basic_dgemm(lda, M, N, K,
 A + i + k*lda,
 B + k + j*lda,
 C + i + j*lda);
 const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
 const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
 const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);
 basic_dgemm(lda, M, N, K,
 A + i + k*lda, B + k + j*lda, C + i + j*lda);
 }
 */

void pmat(double* restrict A, int M, int N) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++)
		{
			//printf("%g, ", A[i + j * M]);
		}
		//printf("\n");
	}
}

void square_dgemm(const int M, const double *A, const double *B, double *C) {
	//double* B_pack = _mm_malloc(A_BLOCK_LEN*M*sizeof(double), MEM_ALIGN);

	const int num_A_blocks = CALC_NUM_BLOCKS(M, A_BLOCK_LEN);
	const int num_C_blocks = CALC_NUM_BLOCKS(M, C_BLOCK_LEN);
	const int num_P_blocks = CALC_NUM_BLOCKS(M, P_BLOCK_LEN);
	int panel, C_block, P_block, K_block;
	// iterate over panels of A/B
	for (panel = 0; panel < num_A_blocks; ++panel) {
		const int panel_pos = panel * A_BLOCK_LEN;
		const int panel_width = CALC_CUR_BLOCK_WIDTH(panel_pos, A_BLOCK_LEN, M);
		// pack the current panel of B into contiguous memory as B_pack
		// to_B_pack(B_pack, B + panel_pos, panel_width, M);
		for (int j = 0; j < M; ++j) {
			/*
			 memcpy(B_pack + A_BLOCK_LEN * col,
			 B + panel_pos + M * col,
			 panel_width * sizeof(double));
			 */
			for (int i = 0; i < panel_width; ++i) {
				B_pack[i + j * A_BLOCK_LEN] = B[panel_pos + i + j * M];
			}
		}
		// iterate over panels of C
		for (C_block = 0; C_block < num_C_blocks; ++C_block) {
			// assume B packed into contigous memory as B_pack
			const int C_block_pos = C_block * C_BLOCK_LEN;
			const int C_block_width = CALC_CUR_BLOCK_WIDTH(C_block_pos, C_BLOCK_LEN, M);
			const int num_K_blocks = CALC_NUM_BLOCKS(C_block_width, K_BLOCK_LEN);
			// pack block of A into contiguous memory as A_pack
			// to_A_pack(A_pack, A + C_block_pos + panel_pos*C_BLOCK_LEN,
			//   C_block_width, panel_width);
			for (int j = 0; j < panel_width; ++j) {
				/*
				 memcpy(A_pack + C_BLOCK_LEN * col,
				 A + C_block_pos + C_BLOCK_LEN * (col + panel_pos),
				 C_block_width * sizeof(double));
				 */
				for (int i = 0; i < C_block_width; ++i) {
					A_pack[i + j * C_BLOCK_LEN] = A[(C_block_pos + i)
							+ (panel_pos + j) * M];
				}
			}
			// iterate over panels of C_pack/B_pack
			for (P_block = 0; P_block < num_P_blocks; ++P_block) {
				// compute the P_block of C_pack = A_pack * B_pack
				const int P_block_pos = P_block * P_BLOCK_LEN;
				const int P_block_width = CALC_CUR_BLOCK_WIDTH(P_block_pos,
						P_BLOCK_LEN,
						M);
				to_kdgemm_B_sized(A_BLOCK_LEN,
						B_pack + P_block_pos * A_BLOCK_LEN, B_kernel,
						panel_width, P_block_width);
				for (K_block = 0; K_block < num_K_blocks; ++K_block) {
					// call 2xP times Px2 kernel to populate C_pack
					const int K_block_pos = K_block * K_BLOCK_LEN;
					const int K_block_width = CALC_CUR_BLOCK_WIDTH(K_block_pos,
							K_BLOCK_LEN,
							C_block_width);
					to_kdgemm_A_sized(C_BLOCK_LEN, A_pack + K_block_pos,
							A_kernel, C_block_width, panel_width);
					//printf("panel=%d,cblock=%d,pblock=%d,kblock=%d,panelwidth=%d,cblockwidth=%d,pblockwidth=%d,kblockwidth=%d,cblockpos=%d\n",panel,C_block,P_block,K_block,panel_width,C_block_width,P_block_width,K_block_width,C_block_pos);
					printf("Print A_kernel:\n");
					pmat(A_kernel, 2, A_BLOCK_LEN);
					printf("Print B_kernel:\n");
					pmat(B_kernel, A_BLOCK_LEN, 2);
					clear_kdgemm_C_sized(C_kernel);
					kdgemm(A_kernel, B_kernel, C_kernel);
					printf("Print C_kernel:\n");
					pmat(C_kernel, 2, 2);
					from_kdgemm_C_sized(K_BLOCK_LEN, C_kernel,
							C_pack + K_block_pos, K_block_width, P_block_width);
				}
				/*
				 from_C_pack(C + C_block_pos + P_block_pos * M,
				 C_pack,
				 C_block_width,
				 P_block_width,
				 M);
				 */
				for (int col = 0; col < P_block_width; ++col)
					for (int row = 0; row < C_block_width; ++row) {
						int i = row + C_block_pos;
						int j = col + P_block_pos;
						int result = C_pack[row + col * C_BLOCK_LEN];
						C[i + M * j] += result;
						printf("i=%d,j=%d,result=%d\n", i, j, result);
					}
			}
		}
	}
	//_mm_free(B_pack);
}

void to_B_pack(double* dest, double* source, const int num_rows,
		const int num_cols) {
	for (int col = 0; col < num_cols; ++col) {
		memcpy(dest + A_BLOCK_LEN * col, source + A_BLOCK_LEN * col,
				num_rows * sizeof(double));
	}
}

void to_A_pack(double* dest, double* source, const int num_rows,
		const int num_cols) {
	for (int col = 0; col < num_cols; ++col) {
		memcpy(dest + C_BLOCK_LEN * col, source + C_BLOCK_LEN * col,
				num_rows * sizeof(double));
	}
}

void from_C_pack(double* C, double* C_pack, const int num_rows,
		const int num_cols, const int M) {
	for (int col = 0; col < num_cols; ++col)
		for (int row = 0; row < num_rows; ++row)
			C[row + col * M] += C_pack[row + col * C_BLOCK_LEN];
}
/*
 to_kdgemm_C_sized(A_BLOCK_LEN,
 C_pack + iter_kernel_col *
 A_BLOCK_LEN + iter_kernel_row,
 C_kernel,
 num_kernel_rows,
 num_kernel_cols);
 */

// const int i = C_block_pos + K_block_pos;
// const int j = P_block_pos;
// const int k = panel_pos;
//pack(A_kernel, A_pack + ..., 2, A_BLOCK_LEN);
//pack(B_kernel, B_pack + ..., A_BLOCK_LEN, 2);
//pack(C_kernel
/*
 void pack(const double* A,
 const double* B,
 const double* C,
 const int M,
 const int mem_row,
 const int mem_num_rows,
 const int mem_col,
 const int mem_num_cols,
 const int mem_acc,
 const int mem_num_accs)
 {
 for(int iter_col = 0; iter_col < lmem_num_cols; ++iter_col)
 {
 memcpy(lmem_sub_B + iter_col * lmem_sub_size,
 lmem_B + (iter_col + lmem_col) * lmem_size + lmem_acc,
 lmem_num_accs * sizeof(double));
 }
 }
 */
// Caux(K_block_pos:K_block_pos+1, 1:2) =
//   Ak(K_block_pos:K_block_pos+1, 1:A_BLOCK_LEN) *
//   Bk(:,P_block_pos:P_block_pos+1)
//
//do_block(M, A, B, C, i, j, k);
/*
 int ii, jj, kk;
 for (ii = i; ii < i+2; ++ii) {
 for (jj = j; jj < j+2; ++jj) {
 double cij = C[jj*M+ii];
 for (kk = k; kk < k+A_BLOCK_LEN; ++kk) {
 fprintf(f, "ii=%d,jj=%d,kk=%d\n",ii,jj,kk);
 cij += A[kk*M+ii] * B[jj*M+kk];
 }
 C[jj*M+ii] = cij;
 }
 //            C[j*lda+i] = cij;
 //	      for (kk = k; kk < k+A_BLOCK_LEN; ++kk) {
 //		C[ii+jj*M] = C[ii+jj*M] + A[ii+kk*M]*B[kk+jj*M];
 //}
 //}
 }
 */
//fclose(f);
/*
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
 // Calculate the number of sub blocks
 const int num_l3_blocks = CALC_NUM_BLOCKS(M, L3_BLOCK_SIZE);

 // Perform blocked operations
 for(int iter_row_block = 0; iter_row_block < num_l3_blocks; ++iter_row_block)
 {
 const int cur_row = iter_row_block * L3_BLOCK_SIZE;
 const int cur_num_rows = CALC_CUR_BLOCK_WIDTH(cur_row, L3_BLOCK_SIZE, M);

 for(int iter_col_block = 0; iter_col_block < num_l3_blocks; ++iter_col_block)
 {
 const int cur_col = iter_col_block * L3_BLOCK_SIZE;
 const int cur_num_cols = CALC_CUR_BLOCK_WIDTH(cur_col, L3_BLOCK_SIZE, M);

 for(int iter_acc_block = 0; iter_acc_block < num_l3_blocks; ++iter_acc_block)
 {
 const int cur_acc = iter_acc_block * L3_BLOCK_SIZE;
 const int cur_num_accs = CALC_CUR_BLOCK_WIDTH(cur_acc, L3_BLOCK_SIZE, M);
 }
 }
 }
 */

/*******************************************************************************
 * Copies data from main memory to the first L3 segment
 * @param A The memory segment to copy from
 * @param B The memory segment to copy from
 * @param C The memory segment to copy from
 * @param M The size of the memory segment
 * @param mem_row The row to copy from
 * @param mem_num_rows The number of rows to copy
 * @param mem_col The column to copy from
 * @param mem_num_cols The number of cols to copy
 * @param mem_acc The accumulator to copy from
 * @param mem_num_accs The number of accs to run
 */
/*
 void copy_main_to_l3(const double* A, const double* B, const double* C, const int M,
 const int mem_row, const int mem_num_rows,
 const int mem_col, const int mem_num_cols,
 const int mem_acc, const int mem_num_accs)
 {
 // Copy all of C
 for(int iter_col = 0; iter_col < mem_num_cols; ++iter_col)
 {
 memcpy(l3mem_C + iter_col * L3_BLOCK_SIZE,
 C + (iter_col + mem_col) * M + mem_row,
 mem_num_rows * sizeof(double));
 }
 // Copy all of A
 for(int iter_col = 0; iter_col < mem_num_accs; ++iter_col)
 {
 memcpy(l3mem_A + iter_col * L3_BLOCK_SIZE,
 A + (iter_col + mem_acc) * M + mem_row,
 mem_num_rows * sizeof(double));
 }
 // Copy all of B
 for(int iter_col = 0; iter_col < mem_num_cols; ++iter_col)
 {
 memcpy(l3mem_B + iter_col * L3_BLOCK_SIZE,
 B + (iter_col + mem_col) * M + mem_acc,
 mem_num_accs * sizeof(double));
 }
 }
 */
/*******************************************************************************
 * Copies data to main memory from the L3 segment
 * @param C The memory segment to copy to
 * @param M The size of the memory segment
 * @param mem_row The row to copy from
 * @param mem_num_rows The number of rows to copy
 * @param mem_col The column to copy from
 * @param mem_num_cols The number of cols to copy
 */
/*
 void copy_main_from_l3(double* C, const int M,
 const int mem_row, const int mem_num_rows,
 const int mem_col, const int mem_num_cols)
 {
 // Copy all of C
 for(int iter_col = 0; iter_col < mem_num_cols; ++iter_col)
 {
 memcpy(C + (iter_col + mem_col) * M + mem_row,
 l3mem_C + iter_col * L3_BLOCK_SIZE,
 mem_num_rows * sizeof(double));
 }
 }
 */
/*******************************************************************************
 * Copies data from one cache memory size to another. Direction cur -> sub
 * @param lmem_A The current memory buffer for A
 * @param lmem_B The current memory buffer for B
 * @param lmem_C The current memory buffer for C
 * @param lmem_size The size of the current memory buffer
 * @param lmem_row The row number to copy from
 * @param lmem_num_rows The number of rows to copy
 * @param lmem_col The col number to copy from
 * @param lmem_num_cols The number of cols to copy
 * @param lmem_acc The accumulator number to copy from
 * @param lmem_num_acc The number of acc rows to copy
 * @param lmem_sub_A The sub memory buffer to copy to
 * @param lmem_sub_B The sub memory buffer to copy to
 * @param lmem_sub_C The sub memory buffer to copy to
 * @param lmem_sub_size The size of the sub memory buffer
 */
/*
 void copy_lmem_to_sublmem(const double* restrict lmem_A,
 const double* restrict lmem_B,
 const double* restrict lmem_C,
 const int lmem_size,
 const int lmem_row,
 const int lmem_num_rows,
 const int lmem_col,
 const int lmem_num_cols,
 const int lmem_acc,
 const int lmem_num_accs,
 double* restrict lmem_sub_A,
 double* restrict lmem_sub_B,
 double* restrict lmem_sub_C,
 const int lmem_sub_size)
 {
 // Copy all of C
 for(int iter_col = 0; iter_col < lmem_num_cols; ++iter_col)
 {
 memcpy(lmem_sub_C + iter_col * lmem_sub_size,
 lmem_C + (iter_col + lmem_col) * lmem_size + lmem_row,
 lmem_num_rows * sizeof(double));
 }
 // Copy all of A
 for(int iter_col = 0; iter_col < lmem_num_accs; ++iter_col)
 {
 memcpy(lmem_sub_A + iter_col * lmem_sub_size,
 lmem_A + (iter_col + lmem_acc) * lmem_size + lmem_row,
 lmem_num_rows * sizeof(double));
 }
 // Copy all of B
 for(int iter_col = 0; iter_col < lmem_num_cols; ++iter_col)
 {
 memcpy(lmem_sub_B + iter_col * lmem_sub_size,
 lmem_B + (iter_col + lmem_col) * lmem_size + lmem_acc,
 lmem_num_accs * sizeof(double));
 }
 }
 */
/*******************************************************************************
 * Copies data from one cache memory size to another. Direction sub -> current
 * @param lmem_C The current memory buffer for C
 * @param lmem_size The size of the current memory buffer
 * @param lmem_row The row number to copy from
 * @param lmem_num_rows The number of rows to copy
 * @param lmem_col The col number to copy from
 * @param lmem_num_cols The number of cols to copy
 * @param lmem_sub_C The sub memory buffer to copy to
 * @param lmem_sub_size The size of the sub memory buffer
 */
/*
 void copy_lmem_from_sublmem(double* restrict lmem_C,
 const int lmem_size,
 const int lmem_row,
 const int lmem_num_rows,
 const int lmem_col,
 const int lmem_num_cols,
 double* restrict lmem_sub_C,
 const int lmem_sub_size)
 {
 // Copy all of C
 for(int iter_col = 0; iter_col < lmem_num_cols; ++iter_col)
 {
 memcpy(lmem_C + (iter_col + lmem_col) * lmem_size + lmem_row,				lmem_sub_C + iter_col * lmem_sub_size,
 lmem_num_rows * sizeof(double));
 }
 }
 */
/*******************************************************************************
 * Performs the blocked recursive matrix multiply at the given cache level
 * @param lmem_A The A memory block of size lmem_size^2
 * @param lmem_B The B memory block of size lmem_size^2
 * @param lmem_C The C memory block of size lmem_size^2
 * @param lmem_size The size of the lmem blocks is lmem_size^2
 * @param lmem_level The cache level of lmem
 * @param lmem_num_fill_row The num of rows in the lmem blocks that are relevant
 * @param lmem_num_fill_col The num of cols in the lmem blocks that are relevant
 * @param lmem_num_fill_acc The num of accumulator row/cols in the lmem blocks that are relevant
 * @param lmem_sub_A The A sub block to use for further blocking
 * @param lmem_sub_B The B sub block to use for further blocking
 * @param lmem_sub_C The C sub block to use for further blocking
 * @param lmem_sub_size The size of the lmem_sub blocks is lmem_sub_size^2
 * @param lmem_sub_level The cache level of lmem_sub
 */
/*
 void square_dgemm_recursive_cache_level(double* restrict lmem_A,
 double* restrict lmem_B,
 double* restrict lmem_C,
 const int lmem_size,
 const int lmem_level,
 const int lmem_num_fill_row,
 const int lmem_num_fill_col,
 const int lmem_num_fill_acc,
 double* restrict lmem_sub_A,
 double* restrict lmem_sub_B,
 double* restrict lmem_sub_C,
 const int lmem_sub_size,
 const int lmem_sub_level)
 {
 // The lmem blocks are aligned in memory, use for caching improvements
 __assume_aligned(lmem_A, MEM_ALIGN);
 __assume_aligned(lmem_B, MEM_ALIGN);
 __assume_aligned(lmem_C, MEM_ALIGN);
 __assume_aligned(lmem_sub_A, MEM_ALIGN);
 __assume_aligned(lmem_sub_B, MEM_ALIGN);
 __assume_aligned(lmem_sub_C, MEM_ALIGN);

 // Check the level, if it is 1 then it is time to run the kernel, otherwise
 // it requires recursion
 if(lmem_level != 1)
 {
 // Not yet on the L1 sized blocks, must recurse

 // Calculate the number of sub blocks
 const int num_sub_row_blocks = CALC_NUM_BLOCKS(lmem_num_fill_row, lmem_sub_size);
 const int num_sub_col_blocks = CALC_NUM_BLOCKS(lmem_num_fill_col, lmem_sub_size);
 const int num_sub_acc_blocks = CALC_NUM_BLOCKS(lmem_num_fill_acc, lmem_sub_size);

 // Perform blocked operations
 for(int iter_row_block = 0; iter_row_block < num_sub_row_blocks; ++iter_row_block)
 {
 const int cur_row = iter_row_block * lmem_sub_size;
 const int cur_num_rows = CALC_CUR_BLOCK_WIDTH(cur_row, lmem_sub_size, lmem_num_fill_row);

 for(int iter_col_block = 0; iter_col_block < num_sub_col_blocks; ++iter_col_block)
 {
 const int cur_col = iter_col_block * lmem_sub_size;
 const int cur_num_cols = CALC_CUR_BLOCK_WIDTH(cur_col, lmem_sub_size, lmem_num_fill_col);


 for(int iter_acc_block = 0; iter_acc_block < num_sub_acc_blocks; ++iter_acc_block)
 {
 const int cur_acc = iter_acc_block * lmem_sub_size;
 const int cur_num_accs = CALC_CUR_BLOCK_WIDTH(cur_acc, lmem_sub_size, lmem_num_fill_acc);

 // Now have the position and size within lmem for which to
 // copy to lmem_sub and recurse

 // Copy from lmem to lmem_sub, make sure to copy padding if
 // the size is odd
 copy_lmem_to_sublmem(lmem_A, lmem_B, lmem_C, lmem_size,
 cur_row, cur_num_rows, cur_col, cur_num_cols, cur_acc, cur_num_accs,
 lmem_sub_A, lmem_sub_B, lmem_sub_C, lmem_sub_size);

 // Recurse
 // Get the next level
 const int lmem_rec_level = lmem_sub_level -1;
 // Based on next level, determine the recurse blocks and size
 double* lmem_rec_A = 0;
 double* lmem_rec_B = 0;
 double* lmem_rec_C = 0;
 int lmem_rec_size = 0;
 switch(lmem_rec_level)
 {
 case 1:
 lmem_rec_A = l1mem_A;
 lmem_rec_B = l1mem_B;
 lmem_rec_C = l1mem_C;
 lmem_rec_size = L1_BLOCK_SIZE;
 break;
 case 2:
 lmem_rec_A = l2mem_A;
 lmem_rec_B = l2mem_B;
 lmem_rec_C = l2mem_C;
 lmem_rec_size = L2_BLOCK_SIZE;
 break;
 }
 // Call recursion
 square_dgemm_recursive_cache_level(lmem_sub_A,
 lmem_sub_B,
 lmem_sub_C,
 lmem_sub_size,
 lmem_sub_level,
 cur_num_rows,
 cur_num_cols,
 cur_num_accs,
 lmem_rec_A,
 lmem_rec_B,
 lmem_rec_C,
 lmem_rec_size,
 lmem_rec_level);

 // Copy from lmem_sub to lmem
 copy_lmem_from_sublmem(lmem_C, lmem_size,
 cur_row, cur_num_rows, cur_col, cur_num_cols,
 lmem_sub_C, lmem_sub_size);
 }
 }
 }
 }
 else
 {
 // Now on the L1 sized blocks, time to run the kernel
 for(int iter_kernel_row = 0; iter_kernel_row < lmem_num_fill_row; iter_kernel_row += KERNEL_M)
 {
 const int num_kernel_rows = CALC_CUR_BLOCK_WIDTH(iter_kernel_row, KERNEL_M, lmem_num_fill_row);
 to_kdgemm_A_sized(L1_BLOCK_SIZE, l1mem_A + iter_kernel_row, kernel_A, num_kernel_rows, lmem_num_fill_acc);

 for(int iter_kernel_col = 0; iter_kernel_col < lmem_num_fill_col; iter_kernel_col += KERNEL_N)
 {
 // Calculate the number of row,cols to process. The kernel will
 // be filled with 0s to handle missing places

 const int num_kernel_cols = CALC_CUR_BLOCK_WIDTH(iter_kernel_col, KERNEL_N, lmem_num_fill_col);

 // Copy current kernel section from L1 to kernel memory
 to_kdgemm_B_sized(L1_BLOCK_SIZE, l1mem_B + iter_kernel_col * L1_BLOCK_SIZE, kernel_B, lmem_num_fill_acc, num_kernel_cols);
 to_kdgemm_C_sized(L1_BLOCK_SIZE, l1mem_C + iter_kernel_col * L1_BLOCK_SIZE + iter_kernel_row, kernel_C, num_kernel_rows, num_kernel_cols);

 // Execute kernel
 kdgemm(kernel_A, kernel_B, kernel_C);

 // Copy back from kernel memory
 from_kdgemm_C_sized(L1_BLOCK_SIZE, kernel_C, l1mem_C + iter_kernel_col * L1_BLOCK_SIZE + iter_kernel_row, num_kernel_rows, num_kernel_cols);
 }
 }
 }

 }
 */
/*******************************************************************************
 * Definition of the squared matrix multiply. Performs recusrively blocked
 * matrix multiply using multiple copy operations. It performs the transpose and
 * the zero padding the kernel requires once at the highest level.
 */
/*
 void square_dgemm_phil(const int M, const double *A, const double *B, double *C)
 {

 // Calculate the number of sub blocks
 const int num_l3_blocks = CALC_NUM_BLOCKS(M, L3_BLOCK_SIZE);

 // Perform blocked operations
 for(int iter_row_block = 0; iter_row_block < num_l3_blocks; ++iter_row_block)
 {
 const int cur_row = iter_row_block * L3_BLOCK_SIZE;
 const int cur_num_rows = CALC_CUR_BLOCK_WIDTH(cur_row, L3_BLOCK_SIZE, M);

 for(int iter_col_block = 0; iter_col_block < num_l3_blocks; ++iter_col_block)
 {
 const int cur_col = iter_col_block * L3_BLOCK_SIZE;
 const int cur_num_cols = CALC_CUR_BLOCK_WIDTH(cur_col, L3_BLOCK_SIZE, M);

 for(int iter_acc_block = 0; iter_acc_block < num_l3_blocks; ++iter_acc_block)
 {
 const int cur_acc = iter_acc_block * L3_BLOCK_SIZE;
 const int cur_num_accs = CALC_CUR_BLOCK_WIDTH(cur_acc, L3_BLOCK_SIZE, M);

 // Copy data to L3 memory to begin recrusive process
 copy_main_to_l3(A, B, C, M, cur_row, cur_num_rows, cur_col, cur_num_cols, cur_acc, cur_num_accs);

 // Begin recursive blocked matrix multiply on L3 size block
 square_dgemm_recursive_cache_level(l3mem_A, l3mem_B, l3mem_C, L3_BLOCK_SIZE, 3,
 cur_num_rows, cur_num_cols, cur_num_accs,
 l2mem_A, l2mem_B, l2mem_C, L2_BLOCK_SIZE, 2);

 // Copy data back to main memory from L3
 copy_main_from_l3(C, M, cur_row, cur_num_rows, cur_col, cur_num_cols);
 }
 }
 }
 }
 */
