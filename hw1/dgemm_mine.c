#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "kdgemm.h"

const char* dgemm_desc = "My awesome dgemm.";

// Define the kernels as the closest value rounded to be the smallest multiple
// of the lower block such that the block is larger

#define BYTE_ALIGNMENT 16

#ifndef L1_KERNEL_P
#define L1_KERNEL_P 16 // (KERNEL_SIZE_ALIGNED(L1_CACHE_SIZE, L1_CACHE_UTILIZATION))
#endif

#if L1_KERNEL_P != KERNEL_P
#error Cannot have differing kernel sizes
#endif

#if KERNEL_M != 2
#error Cannot have kernel size that is not 2
#endif

#if KERNEL_N != 2
#error Cannot have kernel size that is not 2
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

// Memory for the 3 buffers for kernel operations using the L1 cache
double kernel_A[2 * L1_KERNEL_P] __attribute__ ((aligned (BYTE_ALIGNMENT)));
double kernel_B[2 * L1_KERNEL_P] __attribute__ ((aligned (BYTE_ALIGNMENT)));
double kernel_C[2 * 2] __attribute__ ((aligned (BYTE_ALIGNMENT)));

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
	// Get the number of blocks l3 blocks in M
	const int num_blocks = CALC_NUM_BLOCKS(M, L1_KERNEL_P);

	// Perform blocked multiplication with L1_KERNEL_P sized blocks
	for(int iter_col_block = 0; iter_col_block < num_blocks; ++iter_col_block)
	{
		const int cur_col = iter_col_block * L1_KERNEL_P;
		const int cur_col_num = CALC_CUR_BLOCK_WIDTH(cur_col, L1_KERNEL_P, M);
		const int num_blocks_kernel_col = CALC_NUM_BLOCKS(cur_col_num, 2);

		for(int iter_row_block = 0; iter_row_block < num_blocks; ++iter_row_block)
		{
			const int cur_row = iter_row_block * L1_KERNEL_P;
			const int cur_row_num = CALC_CUR_BLOCK_WIDTH(cur_row, L1_KERNEL_P, M);
			const int num_blocks_kernel_row = CALC_NUM_BLOCKS(cur_row_num, 2);

			for(int iter_accum_block = 0; iter_accum_block < num_blocks; ++iter_accum_block)
			{
				const int cur_accum = iter_accum_block * L1_KERNEL_P;
				const int cur_accum_num = CALC_CUR_BLOCK_WIDTH(cur_accum, L1_KERNEL_P, M);

				// Now have sizes of L1_KERNEL_P^2 or less. Use kernel which calculates blocks
				// of size [2x2] = [2xL1_KERNEL_P] [L1_KERNEL_Px2]

				// Perform kernel operations for each [2x2] block within the L1_KERNEL_P sized block
				for(int iter_kernel_col_block = 0; iter_kernel_col_block < num_blocks_kernel_col; ++iter_kernel_col_block)
				{
					const int cur_kernel_col = 2 * iter_kernel_col_block;
					const int cur_kernel_col_num = CALC_CUR_BLOCK_WIDTH(cur_kernel_col, 2, cur_col_num);

					for(int iter_kernel_row_block = 0; iter_kernel_row_block < num_blocks_kernel_row; ++iter_kernel_row_block)
					{
						const int cur_kernel_row = 2 * iter_kernel_row_block;
						const int cur_kernel_row_num = CALC_CUR_BLOCK_WIDTH(cur_kernel_row, 2, cur_row_num);

						// Copy data into kernel memory buffers (copy optimization 2 & memory layout)
						// Because of L3 memory size and the zero-ing operation, this will fit in the
						// kernel space and have 0s where invalid
						to_kdgemm_A_sized(M, A + M * (cur_accum) + (cur_row + cur_kernel_row), kernel_A, cur_kernel_row_num, cur_accum_num);
						to_kdgemm_B_sized(M, B + M * (cur_col + cur_kernel_col) + (cur_accum), kernel_B, cur_accum_num, cur_kernel_col_num);
						to_kdgemm_C_sized(M, C + M * (cur_col + cur_kernel_col) + (cur_row + cur_kernel_row), kernel_C, cur_kernel_row_num, cur_kernel_col_num);

						// Perform kernel operations
						kdgemm(kernel_A, kernel_B, kernel_C);

						// Copy results out of kernel memory buffers
						from_kdgemm_C_sized(M, kernel_C, C + M * (cur_col + cur_kernel_col) + (cur_row + cur_kernel_row), cur_kernel_row_num, cur_kernel_col_num);
					}
				}
			}
		}
	}
}

