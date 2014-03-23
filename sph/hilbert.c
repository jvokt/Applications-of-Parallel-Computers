/*
 * Computes the hilbert index for increased spatial locality hashing. The
 * implementation is based on the code found at (javascript -> C conversion):
 * http://hilbertsfc.googlecode.com/git/sfc.html
 */

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "binhash.h"
#include "hilbert.h"

#define lookup(ix, iy, iz) (binid_map[ix + numBinDim * iy + numBinDim2 * iz])

typedef struct sfcstate_t
{
	int vert;
	int X;
	int Y;
	int Z;
} sfcstate;

void zpk(sfcstate* st, unsigned* arr, int n, int block, int xsign, int ysign, int zsign);
unsigned* createSFCArray(int numBinDim, int numBinDim3);

/**
 * Creates the hilbert curve and stores it in the bindid mapping. It first
 * creates the reverse mapping using code found on the web, then reverses and
 * stores the mapping. The curve is created once and is quick, so the
 * inefficiency is ok
 */
void hilbert_index_create(unsigned* binid_map, int numBinDim, int numBinDim2, int numBinDim3)
{
	// First create the reverse map using the obtained code
	unsigned* reverse_map = createSFCArray(numBinDim, numBinDim3);

	// Clear in case
	memset(binid_map, 0, sizeof(unsigned) * numBinDim3);

	// Reverse and store the mapping
	for(int index = 0; index < numBinDim3; ++index)
	{
		// Get the three values from the map
		unsigned start_pos = index * 3;
		unsigned ix = reverse_map[start_pos];
		unsigned iy = reverse_map[start_pos + 1];
		unsigned iz = reverse_map[start_pos + 2];

		// Store the index in the returned map. The index can be greater because
		// the hilbert only works with powers of 2. Locality can be increased by
		// using the min(index,HASH_SIZE), but better load balancing is achieved
		// with modulus
		lookup(ix, iy, iz) = index % HASH_SIZE;
	}

	// Free the created memory
	free(reverse_map);
}

unsigned* createSFCArray(int numBinDim, int numBinDim3)
{
	// Create the buffer to store the map
	unsigned* buffer = malloc(sizeof(unsigned) * 3 * numBinDim3);

	// Create the needed state holder and initialize
	sfcstate state;
	state.vert = 0;
	state.X = 0;
	state.Y = 0;
	state.Z = 0;

	// Fill and return the buffer
	zpk(&state, buffer, numBinDim, 1, 1,1,1);
	return buffer;
}


// Code from online to fill space curve, modified to run in C
void zpk(sfcstate* st, unsigned* arr, int n, int block, int xsign, int ysign, int zsign)
{

  switch(block)

  {

   case 1:

    if (n>2) zpk(st, arr, n/2,6, -xsign, -ysign, zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->Z+=zsign;

    if (n>2) zpk(st, arr, n/2,3, xsign, ysign, zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->X+=xsign;

    if (n>2) zpk(st, arr, n/2,3, xsign, ysign, zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->Z-=zsign;

    if (n>2) zpk(st, arr, n/2,5, -xsign, ysign, -zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->Y+=ysign;

    if (n>2) zpk(st, arr, n/2,5, -xsign, ysign, -zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->Z+=zsign;

    if (n>2) zpk(st, arr, n/2,3, -xsign, -ysign, zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->X-=xsign;

    if (n>2) zpk(st, arr, n/2,3, -xsign, -ysign, zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->Z-=zsign;

    if (n>2) zpk(st, arr, n/2,6, -xsign, ysign, -zsign);

    arr[st->vert] = st->X; arr[st->vert+1] = st->Y; arr[st->vert+2] = st->Z;

    break;



   case 2:

    if (n>2) zpk(st, arr, n/2,5, xsign, ysign, zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->Y+=ysign;

    if (n>2) zpk(st, arr, n/2,4, -xsign, -ysign, zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->X+=xsign;

    if (n>2) zpk(st, arr, n/2,4, -xsign, -ysign, zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->Y-=ysign;

    if (n>2) zpk(st, arr, n/2,6, xsign, ysign, zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->Z+=zsign;

    if (n>2) zpk(st, arr, n/2,6, xsign, ysign, zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->Y+=ysign;

    if (n>2) zpk(st, arr, n/2,4, xsign, -ysign, -zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->X-=xsign;

    if (n>2) zpk(st, arr, n/2,4, xsign, -ysign, -zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->Y-=ysign;

    if (n>2) zpk(st, arr, n/2,5, xsign, -ysign, -zsign);

    arr[st->vert] = st->X; arr[st->vert+1] = st->Y; arr[st->vert+2] = st->Z;

    break;



   case 3:

    if (n>2) zpk(st, arr, n/2,1, xsign, ysign, zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->Y+=ysign;

    if (n>2) zpk(st, arr, n/2,6, -xsign, -ysign, zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->Z+=zsign;

    if (n>2) zpk(st, arr, n/2,6, -xsign, -ysign, zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->Y-=ysign;

    if (n>2) zpk(st, arr, n/2,4, -xsign, ysign, -zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->X+=xsign;

    if (n>2) zpk(st, arr, n/2,4, -xsign, ysign, -zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->Y+=ysign;

    if (n>2) zpk(st, arr, n/2,6, xsign, -ysign, -zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->Z-=zsign;

    if (n>2) zpk(st, arr, n/2,6, xsign, -ysign, -zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->Y-=ysign;

    if (n>2) zpk(st, arr, n/2,1, -xsign, -ysign, zsign);

    arr[st->vert] = st->X; arr[st->vert+1] = st->Y; arr[st->vert+2] = st->Z;

    break;



   case 4:

    if (n>2) zpk(st, arr, n/2,2, -xsign, -ysign, zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->Z+=zsign;

    if (n>2) zpk(st, arr, n/2,5, -xsign, -ysign, zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->Y-=ysign;

    if (n>2) zpk(st, arr, n/2,5, -xsign, -ysign, zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->Z-=zsign;

    if (n>2) zpk(st, arr, n/2,3, -xsign, ysign, -zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->X-=xsign;

    if (n>2) zpk(st, arr, n/2,3, -xsign, ysign, -zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->Z+=zsign;

    if (n>2) zpk(st, arr, n/2,5, xsign, ysign, zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->Y+=ysign;

    if (n>2) zpk(st, arr, n/2,5, xsign, ysign, zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->Z-=zsign;

    if (n>2) zpk(st, arr, n/2,2, xsign, -ysign, -zsign);

    arr[st->vert] = st->X; arr[st->vert+1] = st->Y; arr[st->vert+2] = st->Z;

    break;



   case 5:

    if (n>2) zpk(st, arr, n/2,4, -xsign, -ysign, zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->X+=xsign;

    if (n>2) zpk(st, arr, n/2,2, xsign, ysign, zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->Z+=zsign;

    if (n>2) zpk(st, arr, n/2,2, xsign, ysign, zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->X-=xsign;

    if (n>2) zpk(st, arr, n/2,1, -xsign, ysign, -zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->Y+=ysign;

    if (n>2) zpk(st, arr, n/2,1, -xsign, ysign, -zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->X+=xsign;

    if (n>2) zpk(st, arr, n/2,2, xsign, -ysign, -zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->Z-=zsign;

    if (n>2) zpk(st, arr, n/2,2, xsign, -ysign, -zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->X-=xsign;

    if (n>2) zpk(st, arr, n/2,4, xsign, ysign, zsign);

    arr[st->vert] = st->X; arr[st->vert+1] = st->Y; arr[st->vert+2] = st->Z;

    break;



   case 6:

    if (n>2) zpk(st, arr, n/2,3, -xsign, -ysign, zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->X-=xsign;

    if (n>2) zpk(st, arr, n/2,1, -xsign, -ysign, zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->Y-=ysign;

    if (n>2) zpk(st, arr, n/2,1, -xsign, -ysign, zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->X+=xsign;

    if (n>2) zpk(st, arr, n/2,2, xsign, ysign, zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->Z+=zsign;

    if (n>2) zpk(st, arr, n/2,2, xsign, ysign, zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->X-=xsign;

    if (n>2) zpk(st, arr, n/2,1, -xsign, ysign, -zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->Y+=ysign;

    if (n>2) zpk(st, arr, n/2,1, -xsign, ysign, -zsign);

    arr[st->vert++] = st->X; arr[st->vert++] = st->Y; arr[st->vert++] = st->Z;

    st->X+=xsign;

    if (n>2) zpk(st, arr, n/2,3, xsign, -ysign, -zsign);

    arr[st->vert] = st->X; arr[st->vert+1] = st->Y; arr[st->vert+2] = st->Z;

    break;



   default: break;

  }

}
