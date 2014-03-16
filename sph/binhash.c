#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "zmorton.h"
#include "binhash.h"

#define min(a,b) (a < b ? a : b)
#define max(a,b) (a > b ? a : b)
#define encode(ix, iy, iz) (zm_encode(ix & HASH_MASK, iy & HASH_MASK, iz & HASH_MASK))

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
    return encode(ix, iy, iz);
}

int compare_unsigned (const void *a, const void *b)
{
	const unsigned* da = (const unsigned *) a;
	const unsigned* db = (const unsigned *) b;

	return (*da > *db) - (*da < *db);
}

unsigned particle_neighborhood(unsigned* buckets, particle_t* p, float h)
{
    /* BEGIN TASK */

	// Get the position of the particle
	unsigned ix = p->x[0]/h;
	unsigned iy = p->x[1]/h;
	unsigned iz = p->x[2]/h;

	// Get the start and end points for the bucket iteration
	unsigned ix_start = max(1, ix) - 1;
	unsigned ix_end = min(HASH_DIM-2, ix) + 1;
	unsigned iy_start = max(1, iy) - 1;
	unsigned iy_end = min(HASH_DIM-2, iy) + 1;
	unsigned iz_start = max(1, iz) - 1;
	unsigned iz_end = min(HASH_DIM-2, iz) + 1;

	// Add the relevant buckets to the set
	unsigned bucket_count = 0;
	for(unsigned iter_x = ix_start; iter_x <= ix_end; ++iter_x)
	{
		for(unsigned iter_y = iy_start; iter_y <= iy_end; ++iter_y)
		{
			for(unsigned iter_z = iz_start; iter_z <= iz_end; ++iter_z)
			{
				// Get the current bin id
				unsigned cur_bin_id = encode(iter_x, iter_y, iter_z);

				// Check if the bin has already been added (don't double count)
				char bin_already_added = 0;
				for(int check_bin_iter = 0; check_bin_iter < bucket_count; ++check_bin_iter)
				{
					if(buckets[check_bin_iter] == cur_bin_id)
					{
						// Bin has already been added
						bin_already_added = 1;
						break;
					}
				}

				// If bin has not already been added, then add it
				if(!bin_already_added)
				{
					buckets[bucket_count] = cur_bin_id;
					++bucket_count;
				}
			}
		}
	}

	// Sort the bins for locality
//	qsort(buckets, bucket_count, sizeof(unsigned), compare_unsigned);

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

				// Move on to next object
				cur_chain_iter = cur_chain_iter->next;
			}
		}
	}

    /* END TASK */
}
