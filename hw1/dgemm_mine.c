#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "kdgemm.h"

const char* dgemm_desc = "My awesome dgemm.";

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
double* B_pack; // Allocated dynamically at runtime
double C_aux[C_BLOCK_LEN * 2] __attribute__ ((aligned (MEM_ALIGN)));

double A_kernel[A_BLOCK_LEN * 2] __attribute__ ((aligned (MEM_ALIGN)));
double B_kernel[2 * A_BLOCK_LEN] __attribute__ ((aligned (MEM_ALIGN)));
double C_kernel[2 * 2] __attribute__ ((aligned (MEM_ALIGN)));


void pmat(double* restrict A, int M, int N) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++)
		{
			//printf("%g, ", A[i + j * M]);
		}
		//printf("\n");
	}
}

/**
 * Block panel product
 * @param The original size M
 * @param Pointer to A at the current row,column
 * @param num_rows_AC The number of valid rows in A,C
 * @param num_acc The number of valid cols in A, and rows in B
 * @param Pointer to C at the current row
 */
void gebp_opt1(const int M, const double* A, const int num_rows_AC, const int num_acc, double* C)
{
	// Pack A into memory
	for(int iter_col = 0; iter_col < num_acc; ++iter_col)
	{
		for(int iter_row = 0; iter_row < num_rows_AC; ++iter_row)
		{
			A_pack[iter_row + iter_col * C_BLOCK_LEN] = A[iter_row + iter_col * M];
		}
	}

	// For each slice in B,C
	const int num_slice_blocks = CALC_NUM_BLOCKS(M, 2);
	for(int iter_slice_block = 0; iter_slice_block < num_slice_blocks; ++iter_slice_block)
	{
		const int cur_slice_pos = iter_slice_block * 2;
		const int num_slice = CALC_CUR_BLOCK_WIDTH(cur_slice_pos, 2, M);

		// For each row in A,C_aux
		const int num_a_aux_blocks = CALC_NUM_BLOCKS(num_rows_AC, 2);
		for(int iter_a_aux_block = 0; iter_a_aux_block < num_a_aux_blocks; ++iter_a_aux_block)
		{
			const int cur_a_aux_pos = iter_a_aux_block * 2;
			const int num_a_aux = CALC_CUR_BLOCK_WIDTH(cur_a_aux_pos, 2, num_rows_AC);

			// Copy into kernel memory
			to_kdgemm_A_sized(C_BLOCK_LEN, A_pack + cur_a_aux_pos, A_kernel, num_a_aux, num_acc);
			to_kdgemm_B_sized(A_BLOCK_LEN, B_pack + cur_slice_pos * A_BLOCK_LEN, B_kernel, num_acc, num_slice);
			clear_kdgemm_C_sized(C_kernel);

			// Run kernel
			kdgemm(A_kernel, B_kernel, C_kernel);

			// Store results into C_aux
			from_kdgemm_C_sized(C_BLOCK_LEN, C_kernel, C_aux + cur_a_aux_pos, num_a_aux, num_slice);
		}

		// Accumulate results from C_aux to C
		for(int iter_slice_part = 0; iter_slice_part < num_slice; ++iter_slice_part)
		{
			for(int iter_row = 0; iter_row < num_rows_AC; ++iter_row)
			{
				C[iter_row + (cur_slice_pos + iter_slice_part) * M] += C_aux[iter_row + iter_slice_part * C_BLOCK_LEN];
			}
		}
	}

}

/**
 * Panel panel product
 * @param M The number of rows in A,C. The number of columns in B,C
 * @param A Pointer to the current column A
 * @param num_accum The number of columns in A, and rows in B
 * @param B Pointer to the current row B
 * @param C Pointer to the entire C matrix
 */
void gepp_blk_var1(const int M, const double* A, const int num_acc, const double* B, double *C)
{
	// Pack B into B_pack
	for(int iter_col = 0; iter_col < M; ++iter_col)
	{
		for(int iter_row = 0; iter_row < num_acc; ++iter_row)
		{
			B_pack[iter_row + iter_col * A_BLOCK_LEN] = B[iter_row + iter_col * M];
		}
	}

	// For each block of A
	int num_AC_blocks = CALC_NUM_BLOCKS(M, A_BLOCK_LEN);
	for(int iter_AC_block = 0; iter_AC_block < num_AC_blocks; ++iter_AC_block)
	{
		const int cur_AC_pos = iter_AC_block * A_BLOCK_LEN;
		const int num_rows_AC = CALC_CUR_BLOCK_WIDTH(cur_AC_pos, A_BLOCK_LEN, M);

		// Perform block panel multiplication
		const double* cur_A = A + cur_AC_pos;
		double* cur_C = C + cur_AC_pos;
		gebp_opt1(M, cur_A, num_rows_AC, num_acc, cur_C);
	}
}

void square_dgemm(const int M, const double *A, const double *B, double *C) {

	// Allocate the B packed memory
	B_pack = _mm_malloc(A_BLOCK_LEN*M*sizeof(double), MEM_ALIGN);

	// Calculate the number of accum blocks in the master loop
	const int num_acc_blocks = CALC_NUM_BLOCKS(M, A_BLOCK_LEN);

	// For each accumulation block (row of B, column of A)
	for(int acc_block = 0; acc_block < num_acc_blocks; ++acc_block)
	{
		const int acc_pos = acc_block * A_BLOCK_LEN;
		const int num_acc = CALC_CUR_BLOCK_WIDTH(acc_pos, A_BLOCK_LEN, M);

		// Get the current column of A and row of B
		const double* cur_A = A + acc_pos * M;
		const double* cur_B = B + acc_pos;

		// Do panel panel product
		gepp_blk_var1(M, cur_A, num_acc, cur_B, C);
	}

	// Free the dynamically sized blcok
	_mm_free(B_pack);
}
