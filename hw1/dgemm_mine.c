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

// Define the kernels as the closest value rounded to be the smallest multiple
// of the lower block such that the block is larger

#ifndef L1_BLOCK_SIZE
#define L1_BLOCK_SIZE KERNEL_P
#endif

#ifndef L2_BLOCK_MULT
#define L2_BLOCK_MULT 16
#endif

#ifndef L2_BLOCK_SIZE
#define L2_BLOCK_SIZE (L1_BLOCK_SIZE * L2_BLOCK_MULT)
#endif

#ifndef L3_BLOCK_MULT
#define L3_BLOCK_MULT 16
#endif

#ifndef L3_BLOCK_SIZE
#define L3_BLOCK_SIZE (L2_BLOCK_SIZE * L3_BLOCK_MULT)
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

// Memory for the 3 buffers for L3 level blocking, converted for zero padding and transposed
double l3mem_A[L3_BLOCK_SIZE * L3_BLOCK_SIZE] __attribute__ ((aligned (BYTE_ALIGNMENT)));
double l3mem_B[L3_BLOCK_SIZE * L3_BLOCK_SIZE] __attribute__ ((aligned (BYTE_ALIGNMENT)));
double l3mem_C[L3_BLOCK_SIZE * L3_BLOCK_SIZE] __attribute__ ((aligned (BYTE_ALIGNMENT)));

// Memory for the 3 buffers for L2 level blocking, already padded and transposed
double l2mem_A[L2_BLOCK_SIZE * L2_BLOCK_SIZE] __attribute__ ((aligned (BYTE_ALIGNMENT)));
double l2mem_B[L2_BLOCK_SIZE * L2_BLOCK_SIZE] __attribute__ ((aligned (BYTE_ALIGNMENT)));
double l2mem_C[L2_BLOCK_SIZE * L2_BLOCK_SIZE] __attribute__ ((aligned (BYTE_ALIGNMENT)));

// Memory for the 3 buffers for L1 level blocking, already padded and transposed
double l1mem_A[L1_BLOCK_SIZE * L1_BLOCK_SIZE] __attribute__ ((aligned (BYTE_ALIGNMENT)));
double l1mem_B[L1_BLOCK_SIZE * L1_BLOCK_SIZE] __attribute__ ((aligned (BYTE_ALIGNMENT)));
double l1mem_C[L1_BLOCK_SIZE * L1_BLOCK_SIZE] __attribute__ ((aligned (BYTE_ALIGNMENT)));

// Memory for the 3 buffers for kernel operations, already padded and transposed
double kernel_A[KERNEL_M * KERNEL_P] __attribute__ ((aligned (BYTE_ALIGNMENT)));
double kernel_B[KERNEL_N * KERNEL_P] __attribute__ ((aligned (BYTE_ALIGNMENT)));
double kernel_C[KERNEL_M * KERNEL_N] __attribute__ ((aligned (BYTE_ALIGNMENT)));

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
void copy_main_to_l3(const double* A, const double* B, const double* C, const int M,
					 const int mem_row, const int mem_num_rows,
					 const int mem_col, const int mem_num_cols,
					 const int mem_acc, const int mem_num_accs)
{
	// Copy all of C
//#pragma unroll(4)
	for(int iter_col = 0; iter_col < mem_num_cols; ++iter_col)
	{
		memcpy(l3mem_C + iter_col * L3_BLOCK_SIZE,
				C + (iter_col + mem_col) * M + mem_row,
				mem_num_rows * sizeof(double));
	}
	// Copy all of A
//#pragma unroll(4)
	for(int iter_col = 0; iter_col < mem_num_accs; ++iter_col)
	{
		memcpy(l3mem_A + iter_col * L3_BLOCK_SIZE,
				A + (iter_col + mem_acc) * M + mem_row,
				mem_num_rows * sizeof(double));
	}
	// Copy all of B
//#pragma unroll(4)
	for(int iter_col = 0; iter_col < mem_num_cols; ++iter_col)
	{
		memcpy(l3mem_B + iter_col * L3_BLOCK_SIZE,
				B + (iter_col + mem_col) * M + mem_acc,
				mem_num_accs * sizeof(double));
	}
}

/*******************************************************************************
 * Copies data to main memory from the L3 segment
 * @param C The memory segment to copy to
 * @param M The size of the memory segment
 * @param mem_row The row to copy from
 * @param mem_num_rows The number of rows to copy
 * @param mem_col The column to copy from
 * @param mem_num_cols The number of cols to copy
 */
void copy_main_from_l3(double* C, const int M,
					 const int mem_row, const int mem_num_rows,
					 const int mem_col, const int mem_num_cols)
{
	// Copy all of C
//#pragma unroll(4)
	for(int iter_col = 0; iter_col < mem_num_cols; ++iter_col)
	{
		memcpy(C + (iter_col + mem_col) * M + mem_row,
				l3mem_C + iter_col * L3_BLOCK_SIZE,
				mem_num_rows * sizeof(double));
	}
}

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
//#pragma unroll(4)
	for(int iter_col = 0; iter_col < lmem_num_cols; ++iter_col)
	{
		memcpy(lmem_sub_C + iter_col * lmem_sub_size,
				lmem_C + (iter_col + lmem_col) * lmem_size + lmem_row,
				lmem_num_rows * sizeof(double));
	}
	// Copy all of A
//#pragma unroll(4)
	for(int iter_col = 0; iter_col < lmem_num_accs; ++iter_col)
	{
		memcpy(lmem_sub_A + iter_col * lmem_sub_size,
				lmem_A + (iter_col + lmem_acc) * lmem_size + lmem_row,
				lmem_num_rows * sizeof(double));
	}
	// Copy all of B
//#pragma unroll(4)
	for(int iter_col = 0; iter_col < lmem_num_cols; ++iter_col)
	{
		memcpy(lmem_sub_B + iter_col * lmem_sub_size,
				lmem_B + (iter_col + lmem_col) * lmem_size + lmem_acc,
				lmem_num_accs * sizeof(double));
	}
}

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
//#pragma unroll(4)
	for(int iter_col = 0; iter_col < lmem_num_cols; ++iter_col)
	{
		memcpy(lmem_C + (iter_col + lmem_col) * lmem_size + lmem_row,
				lmem_sub_C + iter_col * lmem_sub_size,
				lmem_num_rows * sizeof(double));
	}
}

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
	__assume_aligned(lmem_A, BYTE_ALIGNMENT);
	__assume_aligned(lmem_B, BYTE_ALIGNMENT);
	__assume_aligned(lmem_C, BYTE_ALIGNMENT);
	__assume_aligned(lmem_sub_A, BYTE_ALIGNMENT);
	__assume_aligned(lmem_sub_B, BYTE_ALIGNMENT);
	__assume_aligned(lmem_sub_C, BYTE_ALIGNMENT);

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

//#pragma unroll(4)
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

//#pragma unroll(4)
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

/*******************************************************************************
 * Definition of the squared matrix multiply. Performs recusrively blocked
 * matrix multiply using multiple copy operations. It performs the transpose and
 * the zero padding the kernel requires once at the highest level.
 */
void square_dgemm(const int M, const double *A, const double *B, double *C)
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

//#pragma unroll(4)
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

