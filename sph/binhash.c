#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "zmorton.h"
#include "binhash.h"

#define min(a,b) (a < b ? a : b)
#define max(a,b) (a > b ? a : b)
#define encode(ix, iy, iz) (zm_encode(ix & HASH_MASK, iy & HASH_MASK, iz & HASH_MASK))
#define lookup(ix, iy, iz) (binid_map[ix + numBinDim * iy + numBinDim2 * iz])
/*@q
 * ====================================================================
 */

/*@T
 * \subsection{Spatial hashing implementation}
 * 
 * In the current implementation, we assume [[HASH_DIM]] is $2^b$,
 * so that computing a bitwise of an integer with [[HASH_DIM]] extracts
 * the $b$ lowest-order bits.  We could make [[HASH_DIM]] be something
 * other than a power of two, but we would then need to compute an integer
 * modulus or something of that sort.
 * 
 *@c*/

#define HASH_MASK (HASH_DIM-1)

static unsigned* binid_map;
static int numBinDim;
static int numBinDim2;

void particle_bucket_lookup_init(float h)
{
	numBinDim = (int)(ceil(1.0 / h)*1.5);
	numBinDim2 = numBinDim * numBinDim;
	binid_map = malloc(sizeof(unsigned) * numBinDim * numBinDim * numBinDim);

	for(int iz = 0; iz < numBinDim; ++iz)
	{
		for(int iy = 0; iy < numBinDim; ++iy)
		{
			for(int ix = 0; ix < numBinDim; ++ix)
			{
				lookup(ix,iy,iz) = encode(ix, iy, iz);
			}
		}
	}
}

void particle_bucket_lookup_cleanup()
{
	free(binid_map);
}

unsigned particle_bucket_lookup_spatial(unsigned ix, unsigned iy, unsigned iz)
{
	return lookup(ix,iy,iz);
}

unsigned particle_bucket_lookup(particle_t* p, float h)
{
	unsigned ix = p->x[0]/h;
	unsigned iy = p->x[1]/h;
	unsigned iz = p->x[2]/h;
	p->ix = ix;
	p->iy = iy;
	p->iz = iz;
	p->binId = lookup(ix,iy,iz);
	return p->binId;
}

unsigned particle_bucket(particle_t* p, float h)
{
    unsigned ix = p->x[0]/h;
    unsigned iy = p->x[1]/h;
    unsigned iz = p->x[2]/h;
    p->ix = ix;
    p->iy = iy;
    p->iz = iz;
    p->binId = encode(ix, iy, iz);
    return p->binId;
}

int compare_unsigned (const void *a, const void *b)
{
	const unsigned* da = (const unsigned *) a;
	const unsigned* db = (const unsigned *) b;

	return (*da > *db) - (*da < *db);
}

unsigned particle_neighborhood(unsigned* buckets, particle_t* p, float h, char* usedBinID)
{
    /* BEGIN TASK */

	// Get the position of the particle
	unsigned ix = p->ix;
	unsigned iy = p->iy;
	unsigned iz = p->iz;

	// Get the start and end points for the bucket iteration
	unsigned ix_start = max(1, ix) - 1;
	unsigned ix_end = ix + 1;
	unsigned iy_start = max(1, iy) - 1;
	unsigned iy_end = iy + 1;
	unsigned iz_start = max(1, iz) - 1;
	unsigned iz_end = iz + 1;

	// Add the relevant buckets to the set
	unsigned bucket_count = 0;
	for(unsigned iter_x = ix_start; iter_x <= ix_end; ++iter_x)
	{
		for(unsigned iter_y = iy_start; iter_y <= iy_end; ++iter_y)
		{
			for(unsigned iter_z = iz_start; iter_z <= iz_end; ++iter_z)
			{
				// Get the current bin id
				unsigned cur_bin_id = lookup(iter_x, iter_y, iter_z);

				// If bin has not already been added, then add it
				if(!usedBinID[cur_bin_id])
				{
					buckets[bucket_count] = cur_bin_id;
					++bucket_count;

					usedBinID[cur_bin_id] = 1;
				}
			}
		}
	}

	// Unmark all of the bins for future runs
	for(int iter_bucket = 0; iter_bucket < bucket_count; ++iter_bucket)
	{
		usedBinID[buckets[iter_bucket]] = 0;
	}

	// Return the number of buckets added
	return bucket_count;

    /* END TASK */
}

void hash_particles(sim_state_t* s, float h)
{
    /* BEGIN TASK */

	// Get the hash table and number of particles
	particle_t** hash = s->hash;
	const int num_particles = s->n;

	// Clear the hash table
	memset(hash, 0, sizeof(particle_t*) * HASH_SIZE);

	// Add each particle to the hash table
	for(int iter_particle = 0; iter_particle < num_particles; ++iter_particle)
	{
		// Get the current particle
		particle_t* cur_particle = s->part + iter_particle;

		// Get the bin id for which to add the bucket
		unsigned bin_id = particle_bucket_lookup(cur_particle, h);

		// Add the particle to the bin
		cur_particle->next = hash[bin_id];
		hash[bin_id] = cur_particle;
	}

    /* END TASK */
}
