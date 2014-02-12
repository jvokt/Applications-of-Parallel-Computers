#ifndef KDGEMM_H_
#define KDGEMM_H_

/*
 * Dimensions for a "kernel" multiply.  We use define statements in
 * order to make sure these are treated as compile-time constants
 * (which the optimizer likes)
 */
#define KERNEL_M 4
#define KERNEL_N 4
#define KERNEL_P 4

/*
 * Block matrix multiply kernel (simple fixed-size case).
 * Use restrict to tell the compiler there is no aliasing,
 * and inform the compiler of alignment constraints.
 */
void kdgemm(const double * restrict A,
            const double * restrict B,
            double * restrict C);

/*
 * Conversion routines that take a matrix block in column-major form
 * and put it into whatever form the kdgemm routine likes.
 */

void to_kdgemm_A(int ldA, const double* restrict A, double * restrict Ak);

void to_kdgemm_B(int ldB, const double* restrict B, double * restrict Bk);

void to_kdgemm_C(int ldC, const double* restrict C, double * restrict Ck);

void from_kdgemm_C(int ldC, const double* restrict Ck, double * restrict C);

#endif // KDGEMM_H_
