#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <omp.h>

#include "vec3.h"
#include "zmorton.h"

#include "params.h"
#include "state.h"
#include "interact.h"
#include "binhash.h"

/* Define this to use the bucketing version of the code */
#define USE_BUCKETING

/// Buffer used to store flags
char* used_bin_id_flags;

/// Buffer used for local force accumulation
float* forces_all;

/// Buffer used for local rho accumulation
float* rho_all;

void init_buffers(int num_particles)
{
	int max_threads = omp_get_max_threads();
	int total_mem = sizeof(char) * HASH_SIZE * max_threads;
	used_bin_id_flags = malloc(total_mem);
	memset(used_bin_id_flags, 0, total_mem);

	forces_all = malloc(sizeof(float) * num_particles * 3 * max_threads);
	rho_all = malloc(sizeof(float) * num_particles * max_threads);
}

void cleanup_buffers()
{
	free(used_bin_id_flags);
	free(forces_all);
	free(rho_all);
}

/*@T
 * \subsection{Density computations}
 * 
 * The formula for density is
 * \[
 *   \rho_i = \sum_j m_j W_{p6}(r_i-r_j,h)
 *          = \frac{315 m}{64 \pi h^9} \sum_{j \in N_i} (h^2 - r^2)^3.
 * \]
 * We search for neighbors of node $i$ by checking every particle,
 * which is not very efficient.  We do at least take advange of
 * the symmetry of the update ($i$ contributes to $j$ in the same
 * way that $j$ contributes to $i$).
 *@c*/

inline
void update_density(particle_t* pi, particle_t* pj, float h2, float C, float* i_rho, float* j_rho) {
	float r2 = vec3_dist2(pi->x, pj->x);
	float z = h2 - r2;
	if (z > 0) {
		float rho_ij = C * z * z * z;
		*i_rho += rho_ij;
		*j_rho += rho_ij;
	}
}

void compute_density(sim_state_t* s, sim_param_t* params) {
	int n = s->n;
	particle_t* p = s->part;
	particle_t** hash = s->hash;

	float h = params->h;
	float h2 = params->h2;
//	float h3 = params->h3;
//	float h9 = params->h9;
	float C = params->C;



	// Accumulate density info
#ifdef USE_BUCKETING
	/* BEGIN TASK */

	// Clear densities to the initial value
	float rhoAdditive = params->rhoAdditive;
	for (int i = 0; i < n; ++i)
		p[i].rho = rhoAdditive;

	// Start multi-threading
#pragma omp parallel
	{
		// Get the thread id and total threads for work distribution
		int thread_id =  omp_get_thread_num();
		int total_threads = omp_get_num_threads();

		// Setup a neighbor bin store
		unsigned buckets[MAX_NBR_BINS];
		unsigned numbins;

		// Get the rho accumulator for this thread
		float* rho = rho_all + thread_id * s->n;
		memset(rho, 0, sizeof(float) * s->n);

		// Get the dedupe buffer for this thread
		char* usedBinID = used_bin_id_flags + (thread_id * HASH_SIZE);

		// Iterate over each bucket, and within each bucket each particle
		for (int iter_bucket = thread_id; iter_bucket < HASH_SIZE; iter_bucket += total_threads)
		{
			for (particle_t* pi = hash[iter_bucket]; pi != NULL ; pi = pi->next)
			{
				unsigned diffPosI = pi - s->part;
				float* i_rho = rho + diffPosI;

				// Compute equal and opposite rho additives for the particle,
				// first get neighbors
				numbins = particle_neighborhood(buckets, pi, h, usedBinID);
				for (int j = 0; j < numbins; ++j)
				{
					// Get the neighbor particles
					unsigned bucketid = buckets[j];
					for (particle_t* pj = hash[bucketid]; pj != NULL ; pj = pj->next)
					{
						unsigned diffPosJ = pj - s->part;
						float* j_rho = rho + diffPosJ;

						// Compute density only if appropriate
						if (pi < pj && abs(pi->ix - pj->ix) <= 1
								&& abs(pi->iy - pj->iy) <= 1
								&& abs(pi->iz - pj->iz) <= 1)
						{
							update_density(pi, pj, h2, C, i_rho, j_rho);
						}
					}
				}
			}
		}

		// Accumulate the rho for each particle
		for(int iter_particle = 0; iter_particle < s->n; ++iter_particle)
		{
			particle_t* cur_particle = s->part + iter_particle;
			float rho_add = rho[iter_particle];
			if(rho_add != 0)
			{
				omp_set_lock(&cur_particle->lock);
				cur_particle->rho += rho_add;
				omp_unset_lock(&cur_particle->lock);
			}
		}
	}


	/* END TASK */
#else
	// Clear densities
	for (int i = 0; i < n; ++i)
		p[i].rho = 0;

	for (int i = 0; i < n; ++i) {
		particle_t* pi = s->part+i;
		pi->rho += (315.0/64.0/M_PI) * s->mass / h3;
		for (int j = i+1; j < n; ++j) {
			particle_t* pj = s->part+j;
			update_density(pi, pj, h2, C, &pi->rho, &pj->rho);
		}
	}
#endif
}

/*@T
 * \subsection{Computing forces}
 * 
 * The acceleration is computed by the rule
 * \[
 *   \bfa_i = \frac{1}{\rho_i} \sum_{j \in N_i} 
 *     \bff_{ij}^{\mathrm{interact}} + \bfg,
 * \]
 * where the pair interaction formula is as previously described.
 * Like [[compute_density]], the [[compute_accel]] routine takes
 * advantage of the symmetry of the interaction forces
 * ($\bff_{ij}^{\mathrm{interact}} = -\bff_{ji}^{\mathrm{interact}}$)
 * but it does a very expensive brute force search for neighbors.
 *@c*/

inline
void update_forces(particle_t* pi, particle_t* pj, float h2, float rho0,
		float C0, float Cp, float Cv, float* pia, float* pja) {
	float dx[3];
	vec3_diff(dx, pi->x, pj->x);
	float r2 = vec3_len2(dx);
	if (r2 < h2) {
		const float rhoi = pi->rho;
		const float rhoj = pj->rho;
		float q = sqrt(r2 / h2);
		float u = 1 - q;
		float w0 = C0 * u / rhoi / rhoj;
		float wp = w0 * Cp * (rhoi + rhoj - 2 * rho0) * u / q;
		float wv = w0 * Cv;
		float dv[3];
		vec3_diff(dv, pi->v, pj->v);

		// Equal and opposite pressure forces
		vec3_saxpy(pia, wp, dx);
		vec3_saxpy(pja, -wp, dx);

		// Equal and opposite viscosity forces
		vec3_saxpy(pia, wv, dv);
		vec3_saxpy(pja, -wv, dv);
	}
}

void compute_accel(sim_state_t* state, sim_param_t* params) {
	// Unpack basic parameters
	const float h = params->h;
	const float rho0 = params->rho0;
//	const float k = params->k;
//	const float mu = params->mu;
	const float g = params->g;
//	const float mass = state->mass;
	const float h2 = params->h2;

	// Unpack system state
	particle_t* p = state->part;
	particle_t** hash = state->hash;
	const int n = state->n;

	// Rehash the particles
	hash_particles(state, h);

	// Compute density and color
	compute_density(state, params);

	// Constants for interaction term
	const float C0 = params->C0;
	const float Cp = params->Cp;
	const float Cv = params->Cv;

	// Start with gravity and surface forces
	for (int i = 0; i < n; ++i)
	{
		vec3_set(p[i].a, 0, -g, 0);
	}

	// Accumulate forces
#ifdef USE_BUCKETING
	/* BEGIN TASK */

	// Start multi-threaded

	// Iterate over each bucket, and within each bucket each particle
#pragma omp parallel
	{
		// Get the thread ID and total threads
		int thread_id = omp_get_thread_num();
		int total_threads = omp_get_num_threads();

		// Get the appropriate forces vector
		float* forces = forces_all + thread_id * (state->n * 3);
		memset(forces, 0, sizeof(float) * state->n * 3);

		// Get the dedupe buffer for this thread
		char* usedBinID = used_bin_id_flags + (thread_id * HASH_SIZE);

		// Create storage for neighbor set
		unsigned buckets[MAX_NBR_BINS];
		unsigned numbins;

		// Process all buckets that the thread is responsible for
		for (int iter_bucket = thread_id; iter_bucket < HASH_SIZE; iter_bucket += total_threads)
		{
			for (particle_t* pi = hash[iter_bucket]; pi != NULL ; pi = pi->next)
			{
				// Get the position of the pi force accumulator
				unsigned diffPosI = pi - state->part;
				float* pia = forces + 3 * diffPosI;

				// Compute equal and opposite forces for the particle,
				// first get neighbors
				numbins = particle_neighborhood(buckets, pi, h, usedBinID);
				for (int j = 0; j < numbins; ++j)
				{
					// Get the neighbor particles
					unsigned bucketid = buckets[j];
					for (particle_t* pj = hash[bucketid]; pj != NULL ; pj = pj->next) {
						// Compute forces only if appropriate
						if (pi < pj && abs(pi->ix - pj->ix) <= 1
								&& abs(pi->iy - pj->iy) <= 1
								&& abs(pi->iz - pj->iz) <= 1)
						{
							// Get the position of the pj force accumulator
							unsigned diffPosJ = pj - state->part;
							float* pja = forces + 3 * diffPosJ;

							// Accumulate forces
							update_forces(pi, pj, h2, rho0, C0, Cp, Cv, pia, pja);
						}
					}
				}
			}
		}

		// Accumulate values in vector
		for (int iter_particle = 0; iter_particle < state->n; ++iter_particle)
		{
			particle_t* cur_particle = state->part + iter_particle;
			float* pia = forces + 3 * iter_particle;
			if(pia[0] != 0 || pia[1] != 0 || pia[2] != 0)
			{
				omp_set_lock(&cur_particle->lock);
				vec3_saxpy(cur_particle->a, 1, pia);
				omp_unset_lock(&cur_particle->lock);
			}
		}
	}


	/* END TASK */
#else

	for (int i = 0; i < n; ++i) {
		particle_t* pi = p+i;
		for (int j = i+1; j < n; ++j) {
			particle_t* pj = p+j;
			update_forces(pi, pj, h2, rho0, C0, Cp, Cv, pi->a, pj->a);
		}
	}
#endif
}

