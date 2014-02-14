#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "kdgemm.h"

const char* dgemm_desc = "My awesome dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 4)
#endif

/*******************************************************************************
 * Program constants
 */

#ifndef BYTE_ALIGNMENT
#define BYTE_ALIGNMENT 16
#endif

#ifndef NUM_MAT
#define NUM_MAT 3
#endif

/*******************************************************************************
 * Cache size and utilization definitions
 */

#ifndef L1_CACHE_SIZE
#define L1_CACHE_SIZE ((int) 32768) // 32KB
#endif

#ifndef L1_CACHE_UTILIZATION
#define L1_CACHE_UTILIZATION ((double) 0.5)
#endif

#ifndef L2_CACHE_SIZE
#define L2_CACHE_SIZE ((int) 262144) // 256KB
#endif

#ifndef L2_CACHE_UTILIZATION
#define L2_CACHE_UTILIZATION ((double) 0.5)
#endif

#ifndef L3_CACHE_SIZE
#define L3_CACHE_SIZE ((int) 4194304) // 4MB
#endif

#ifndef L3_CACHE_UTILIZATION
#define L3_CACHE_UTILIZATION ((double) 0.75)
#endif


/*******************************************************************************
 * Kernel and block size calculations
 */

// Calculate the size of kernels
// block_size_bytes = (int)(floor(sqrt(CACHE_SIZE - 3*16) * CACHE_UTIL / (3 * sizeof(double))))
// block_size_bytes = block_size_bytes + (16 - (block_size_bytes % 16))
#define KERNEL_SIZE_UNALIGNED(CACHE_SIZE, CACHE_UTIL) ((int)(floor(sqrt(CACHE_SIZE - NUM_MAT * BYTE_ALIGNMENT) * CACHE_UTIL / (NUM_MAT * sizeof(double)))))
#define KERNEL_SIZE_ALIGNED(CACHE_SIZE, CACHE_UTIL) (KERNEL_SIZE_UNALIGNED(CACHE_SIZE, CACHE_UTIL) + BYTE_ALIGNMENT - (KERNEL_SIZE_UNALIGNED(CACHE_SIZE, CACHE_UTIL) % BYTE_ALIGNMENT))

// Define the kernels as the closest value rounded to be the smallest multiple
// of the lower block such that the block is larger

#ifndef L1_KERNEL_P
#define L1_KERNEL_P 32 // (KERNEL_SIZE_ALIGNED(L1_CACHE_SIZE, L1_CACHE_UTILIZATION))
#endif

#ifndef L2_BLOCK_SIZE
#define L2_BLOCK_SIZE 64 // (KERNEL_SIZE_ALIGNED(L2_CACHE_SIZE, L2_CACHE_UTILIZATION))
#endif

#ifndef L3_BLOCK_SIZE
#define L3_BLOCK_SIZE 128 // (KERNEL_SIZE_ALIGNED(L3_CACHE_SIZE, L3_CACHE_UTILIZATION))
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
	// This function works by performing recursively blocked matrix multiply.
	// The highest level works so that the block size causes the 3 blocks of the
	// matrix operation to work entirely in L3 cache. The second level works so
	// that its block size causes all 3 blocks to fall in L2 cache. The final
	// level uses the kernel which operates on PxP matrices in a 2xP blocked
	// fashion. The first step is to calculate the various block sizes and then
	// they will be executed. The various block sizes were calculated so that
	// boundaries are aligned in memory, which means once copied into aligned
	// memory the alignment won't be an issue

	// Get the number of blocks l3 blocks in M
	const int num_l3_blocks = CALC_NUM_BLOCKS(M, L3_BLOCK_SIZE);

	// Perform blocked multiplication with L3_BLOCK_SIZE sized blocks
	for(int iter_l3_row_block = 0; iter_l3_row_block < num_l3_blocks; ++iter_l3_row_block)
	{
		const int cur_main_row_pos = iter_l3_row_block * L3_BLOCK_SIZE;
		const int cur_main_row_width = CALC_CUR_BLOCK_WIDTH(cur_main_row_pos, L3_BLOCK_SIZE, M);
		const int num_l2_blocks_row = CALC_NUM_BLOCKS(cur_main_row_width, L2_BLOCK_SIZE);

		for(int iter_main_col_block = 0; iter_main_col_block < num_l3_blocks; ++iter_main_col_block)
		{
			const int cur_main_col_pos = iter_main_col_block * L3_BLOCK_SIZE;
			const int cur_main_col_width = CALC_CUR_BLOCK_WIDTH(cur_main_col_pos, L3_BLOCK_SIZE, M);
			const int num_l2_blocks_col = CALC_NUM_BLOCKS(cur_main_col_width, L2_BLOCK_SIZE);

			for(int iter_main_accum_block = 0; iter_main_accum_block < num_l3_blocks; ++iter_main_accum_block)
			{
				const int cur_main_accum_pos = iter_main_accum_block * L3_BLOCK_SIZE;
				const int cur_main_accum_width = CALC_CUR_BLOCK_WIDTH(cur_main_accum_pos, L3_BLOCK_SIZE, M);
				const int num_l2_blocks_accum = CALC_NUM_BLOCKS(cur_main_accum_width, L2_BLOCK_SIZE);

				// Perform blocked multiplication with L2_BLOCK_SIZE sized blocks
				for(int iter_l3_row_block = 0; iter_l3_row_block < num_l2_blocks_row; ++iter_l3_row_block)
				{
					const int cur_l3_row_pos = iter_l3_row_block * L2_BLOCK_SIZE;
					const int cur_l3_row_width = CALC_CUR_BLOCK_WIDTH(cur_l3_row_pos, L2_BLOCK_SIZE, cur_main_row_width);
					const int num_l1_blocks_row = CALC_NUM_BLOCKS(cur_l3_row_width, L1_KERNEL_P);

					for(int iter_l3_col_block = 0; iter_l3_col_block < num_l2_blocks_col; ++iter_l3_col_block)
					{
						const int cur_l3_col_pos = iter_l3_col_block * L2_BLOCK_SIZE;
						const int cur_l3_col_width = CALC_CUR_BLOCK_WIDTH(cur_l3_col_pos, L2_BLOCK_SIZE, cur_main_col_width);
						const int num_l1_blocks_col = CALC_NUM_BLOCKS(cur_l3_col_width, L1_KERNEL_P);

						for(int iter_l3_accum_block = 0; iter_l3_accum_block < num_l2_blocks_accum; ++iter_l3_accum_block)
						{
							const int cur_l3_accum_pos = iter_l3_accum_block * L2_BLOCK_SIZE;
							const int cur_l3_accum_width = CALC_CUR_BLOCK_WIDTH(cur_l3_accum_pos, L2_BLOCK_SIZE, cur_main_accum_width);
							const int num_l1_blocks_accum = CALC_NUM_BLOCKS(cur_l3_accum_width, L1_KERNEL_P);

							// Perform blocked multiplication with L1_KERNEL_P sized blocks
							for(int iter_l2_row_block = 0; iter_l2_row_block < num_l1_blocks_row; ++iter_l2_row_block)
							{
								const int cur_l2_row_pos = iter_l2_row_block * L1_KERNEL_P;
								const int cur_l2_row_width = CALC_CUR_BLOCK_WIDTH(cur_l2_row_pos, L1_KERNEL_P, cur_l3_row_width);
								const int num_kernel_blocks_row = CALC_NUM_BLOCKS(cur_l2_row_width, 2);

								for(int iter_l2_col_block = 0; iter_l2_col_block < num_l1_blocks_col; ++iter_l2_col_block)
								{
									const int cur_l2_col_pos = iter_l2_col_block * L1_KERNEL_P;
									const int cur_l2_col_width = CALC_CUR_BLOCK_WIDTH(cur_l2_col_pos, L1_KERNEL_P, cur_l3_col_width);
									const int num_kernel_blocks_col = CALC_NUM_BLOCKS(cur_l2_col_width, 2);

									for(int iter_l2_accum_block = 0; iter_l2_accum_block < num_l1_blocks_accum; ++iter_l2_accum_block)
									{
										const int cur_l2_accum_pos = iter_l2_accum_block * L1_KERNEL_P;
										const int cur_l2_accum_width = CALC_CUR_BLOCK_WIDTH(cur_l2_accum_pos, L1_KERNEL_P, cur_l3_accum_width);

										// Now have sizes of L1_KERNEL_P^2 or less. Use kernel which calculates blocks
										// of size [2x2] = [L1_KERNEL_Px2] [2xL1_KERNEL_P]

										// Perform kernel operations for each [2x2] block within the L1_KERNEL_P sized block
										for(int iter_kernel_row_block = 0; iter_kernel_row_block < num_kernel_blocks_row; ++iter_kernel_row_block)
										{
											const int cur_kernel_row_pos = 2 * iter_kernel_row_block;
											const int cur_kernel_row_width = CALC_CUR_BLOCK_WIDTH(cur_kernel_row_pos, 2, cur_l2_row_width);

											// Copy data to L3
											to_kdgemm_A_sized(M, A + M * (cur_main_accum_pos + cur_l3_accum_pos + cur_l2_accum_pos) + cur_main_row_pos + cur_l3_row_pos + cur_l2_row_pos + cur_kernel_row_pos, kernel_A, cur_kernel_row_width, cur_l2_accum_width);

											for(int iter_kernel_col_block = 0; iter_kernel_col_block < num_kernel_blocks_col; ++iter_kernel_col_block)
											{
												const int cur_kernel_col_pos = 2 * iter_kernel_col_block;
												const int cur_kernel_col_width = CALC_CUR_BLOCK_WIDTH(cur_kernel_col_pos, 2, cur_l2_col_width);

												// Now have complete offset into l3 memory buffer from which to copy data
												// into the kernel memory buffers

												// Copy data into kernel memory buffers (copy optimization 2 & memory layout)
												// Because of L3 memory size and the zero-ing operation, this will fit in the
												// kernel space and have 0s where invalid
												to_kdgemm_B_sized(M, B + M * (cur_main_col_pos + cur_l3_col_pos + cur_l2_col_pos + cur_kernel_col_pos) + cur_main_accum_pos + cur_l3_accum_pos + cur_l2_accum_pos, kernel_B, cur_l2_accum_width, cur_kernel_col_width);
												to_kdgemm_C_sized(M, C + M * (cur_main_col_pos + cur_l3_col_pos + cur_l2_col_pos + cur_kernel_col_pos) + cur_main_row_pos + cur_l3_row_pos + cur_l2_row_pos + cur_kernel_row_pos, kernel_C, cur_kernel_row_width, cur_kernel_col_width);

												// Perform kernel operations
												kdgemm(kernel_A, kernel_B, kernel_C);

												// Copy results out of kernel memory buffers
												from_kdgemm_C_sized(M, kernel_C, C + M * (cur_main_col_pos + cur_l3_col_pos + cur_l2_col_pos + cur_kernel_col_pos) + cur_main_row_pos + cur_l3_row_pos + cur_l2_row_pos + cur_kernel_row_pos, cur_kernel_row_width, cur_kernel_col_width);
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

