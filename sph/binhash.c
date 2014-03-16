#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "zmorton.h"
#include "binhash.h"

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

unsigned particle_bucket(particle_t* p, float h)
{
    unsigned ix = p->x[0]/h;
    unsigned iy = p->x[1]/h;
    unsigned iz = p->x[2]/h;
    return zm_encode(ix & HASH_MASK, iy & HASH_MASK, iz & HASH_MASK);
}

unsigned particle_neighborhood(unsigned* buckets, particle_t* p, float h)
{
    /* BEGIN TASK */

	// Get the position of the particle
	unsigned ix = p->x[0]/h;
	unsigned iy = p->x[1]/h;
	unsigned iz = p->x[2]/h;

	// Get the start and end points for the bucket iteration
	unsigned ix_start = max(0, ix - 1);
	unsigned ix_end = min(HASH_DIM-1, ix + 1);
	unsigned iy_start = max(0, iy - 1);
	unsigned iy_end = min(HASH_DIM-1, iy + 1);
	unsigned iz_start = max(0, iz - 1);
	unsigned iz_end = min(HASH_DIM-1, iz + 1);

	// Add the relevant buckets to the set
	unsigned bucket_count = 0;
	for(unsigned iter_x = ix_start; iter_x <= ix_end; ++iter_x)
	{
		for(unsigned iter_y = iy_start; iter_y <= iy_end; ++iter_y)
		{
			for(unsigned iter_z = iz_start; iter_z <= iz_end; ++iter_z)
			{
				buckets[bucket_count] = zm_encode(iter_x & HASH_MASK, iter_y & HASH_MASK, iter_z & HASH_MASK);
				++bucket_count;
			}
		}
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
	for(int iter_bucket = 0; iter_bucket < HASH_SIZE; ++iter_bucket)
	{
		hash[iter_bucket] = 0;
	}

	// Add each particle to the hash table
	for(int iter_particle = 0; iter_particle < num_particles; ++iter_particle)
	{
		// Get the current particle
		particle_t* cur_particle = s->part + iter_particle;

		// Clear its next pointer
		cur_particle->next = NULL;

		// Get the bin id for which to add the bucket
		unsigned bin_id = particle_bucket(cur_particle, h);

		// Add the particle to the bin
		if(hash[bin_id] == NULL)
		{
			// This is the first particle for the bin, set as head in hash
			hash[bin_id] = cur_particle;
		}
		else
		{
			// This is not the first particle for the bin, add to chain
			// by following chain to its end
			particle_t* cur_chain_iter = hash[bin_id];
			while(cur_chain_iter != NULL)
			{
				// Check if this is the last particle in the chain
				if(cur_chain_iter->next == NULL)
				{
					// Last particle in the chain, add to it and finish
					cur_chain_iter->next = cur_particle;
					break;
				}
			}
		}
	}

    /* END TASK */
}
