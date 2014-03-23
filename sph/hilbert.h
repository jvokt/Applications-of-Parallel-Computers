/*
 * Computes the hilbert index for increased spatial locality hashing. The
 * implementation is based on the code found at (javascript -> C conversion):
 * http://hilbertsfc.googlecode.com/git/sfc.html
 */

#ifndef HILBERT_H_
#define HILBERT_H_

/**
 * Creates the hilbert curve and stores it in the bindid mapping. It first
 * creates the reverse mapping using code found on the web, then reverses and
 * stores the mapping. The curve is created once and is quick, so the
 * inefficiency is ok
 */
void hilbert_index_create(unsigned* binid_map, int numBinDim, int numBinDim2, int numBinDim3);


#endif /* HILBERT_H_ */
