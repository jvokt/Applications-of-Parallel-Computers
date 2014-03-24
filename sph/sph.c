//#define GPERF_DEBUG

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <omp.h>

#ifdef GPERF_DEBUG
#include <gperftools/profiler.h>
#endif

#include "vec3.h"
#include "io.h"
#include "params.h"
#include "state.h"
#include "binhash.h"
#include "interact.h"
#include "leapfrog.h"

/*@q
 * ====================================================================
 */

/*@T
 * \section{Initialization}
 *
 * We've hard coded the computational domain to a unit box, but we'd prefer
 * to do something more flexible for the initial distribution of fluid.
 * In particular, we define the initial geometry of the fluid in terms of an
 * {\em indicator function} that is one for points in the domain occupied
 * by fluid and zero elsewhere.  A [[domain_fun_t]] is a pointer to an
 * indicator for a domain, which is a function that takes two floats and
 * returns 0 or 1.  Two examples of indicator functions are a little box
 * of fluid in the corner of the domain and a circular drop.
 *@c*/
typedef int (*domain_fun_t)(float, float, float);

int box_indicator(float x, float y, float z)
{
    return (x < 0.5) && (y < 0.75) && (z < 0.5);
}

int circ_indicator(float x, float y, float z)
{
    float dx = (x-0.5);
    float dy = (y-0.5);
    float dz = (z-0.5);
    float r2 = dx*dx + dy*dy + dz*dz;
    return (r2 < 0.25*0.25*0.25);
}

/*@T
 *
 * The [[place_particles]] routine fills a region (indicated by the
 * [[indicatef]] argument) with fluid particles.  The fluid particles
 * are placed at points inside the domain that lie on a regular mesh
 * with cell sizes of $h/1.3$.  This is close enough to allow the
 * particles to overlap somewhat, but not too much.
 *@c*/
sim_state_t* place_particles(sim_param_t* param, 
                             domain_fun_t indicatef)
{
    float h  = param->h;
    float hh = h/1.3;

    // Count mesh points that fall in indicated region.
    int count = 0;
    for (float x = 0; x < 1; x += hh)
        for (float y = 0; y < 1; y += hh)
        	for (float z = 0; z < 1; z += hh)
        		count += indicatef(x,y,z);

    // Populate the particle data structure
    sim_state_t* s = alloc_state(count);
    int p = 0;
    for (float x = 0; x < 1; x += hh) {
        for (float y = 0; y < 1; y += hh) {
            for (float z = 0; z < 1; z += hh) {
                if (indicatef(x,y,z)) {
                    vec3_set(s->part[p].x, x, y, z);
                    vec3_set(s->part[p].v, 0, 0, 0);
                    ++p;
                }
            }
        }
    }
    return s;    
}

/*@T
 *
 * The [[place_particle]] routine determines the initial particle
 * placement, but not the desired mass.  We want the fluid in the
 * initial configuration to exist roughly at the reference density.
 * One way to do this is to take the volume in the indicated body of
 * fluid, multiply by the mass density, and divide by the number of
 * particles; but that requires that we be able to compute the volume
 * of the fluid region.  Alternately, we can simply compute the
 * average mass density assuming each particle has mass one, then use
 * that to compute the particle mass necessary in order to achieve the
 * desired reference density.  We do this with [[normalize_mass]].
 * 
 * @c*/
void normalize_mass(sim_state_t* s, sim_param_t* param)
{
    s->mass = 1;

    float C = (315.0 / 64.0 / M_PI) * s->mass / param->h9;
	param->C = C;
	float rhoAdditive = (315.0 / 64.0 / M_PI) * s->mass / param->h3;
	param->rhoAdditive = rhoAdditive;
	float C0 = 45 * s->mass / M_PI / ((param->h2) * (param->h2) * param->h);
	param->C0 = C0;

    hash_particles(s, param->h);
    compute_density(s, param);
    float rho0 = param->rho0;
    float rho2s = 0;
    float rhos  = 0;
    for (int i = 0; i < s->n; ++i) {
        rho2s += (s->part[i].rho)*(s->part[i].rho);
        rhos  += s->part[i].rho;
    }
    s->mass *= ( rho0*rhos / rho2s );
}

sim_state_t* init_particles(sim_param_t* param)
{
    sim_state_t* s = place_particles(param, box_indicator);

    printf("Creating dedupe flag vector for threads...  ");
	fflush(stdout);
    init_buffers(s->n);
    printf("Complete\n");

    normalize_mass(s, param);
    return s;
}

/*@T
 * \section{The [[main]] event}
 *
 * The [[main]] routine actually runs the time step loop, writing
 * out files for visualization every few steps.  For debugging
 * convenience, we use [[check_state]] before writing out frames,
 * just so that we don't spend a lot of time on a simulation that
 * has gone berserk.
 *@c*/

void check_state(sim_state_t* s)
{
    for (int i = 0; i < s->n; ++i) {
        float xi = s->part[i].x[0];
        float yi = s->part[i].x[1];
        float zi = s->part[i].x[2];
        assert( xi >= 0 || xi <= 1 );
        assert( yi >= 0 || yi <= 1 );
        assert( zi >= 0 || zi <= 1 );
    }
}

int main(int argc, char** argv)
{
#ifdef GPERF_DEBUG
	ProfilerStart("sph-gperftools.profile");
#endif

	printf("Running simulation with max threads: %d\n", omp_get_max_threads());

    sim_param_t params;
    if (get_params(argc, argv, &params) != 0)
        exit(-1);

    printf("Creating bucket lookup table...  ");
	fflush(stdout);
	particle_bucket_lookup_init(params.h);
	printf("Complete\n");

	printf("Precomputation for reduced redundancy...  ");
	float h = params.h;
	float h2 = h * h;
	params.h2 = h2;
	float h3 = h2 * h;
	params.h3 = h3;
	float h9 = h3 * h3 * h3;
	params.h9 = h9;
	float Cp = params.k / 2;
	params.Cp = Cp;
	float Cv = -1 * params.mu;
	params.Cv = Cv;
	printf("Complete\n");

    sim_state_t* state = init_particles(&params);
    FILE* fp    = fopen(params.fname, "w");
    int nframes = params.nframes;
    int npframe = params.npframe;
    float dt    = params.dt;
    int n       = state->n;

    printf("Redoing precomputation calculations based on new mass... ");
    fflush(stdout);
    float C = (315.0 / 64.0 / M_PI) * state->mass / h9;
	params.C = C;
	float rhoAdditive = (315.0 / 64.0 / M_PI) * state->mass / h3;
	params.rhoAdditive = rhoAdditive;
	float C0 = 45 * state->mass / M_PI / ((h2) * (h2) * h);
	params.C0 = C0;
	printf("Complete\n");

	printf("Initializing particle locks... ");
	fflush(stdout);
	for(int iter_particle = 0; iter_particle < state->n; ++iter_particle)
	{
		particle_t* cur_particle = state->part + iter_particle;
		omp_init_lock(&cur_particle->lock);
	}
	printf("Complete\n");


    double t_start = omp_get_wtime();

    //write_header(fp, n);
    write_header(fp, n, nframes, params.h);
    write_frame_data(fp, n, state, NULL);

    compute_accel(state, &params);
    leapfrog_start(state, dt);
    check_state(state);

    for (int frame = 1; frame < nframes; ++frame) {
        for (int i = 0; i < npframe; ++i) {
            compute_accel(state, &params);
            leapfrog_step(state, dt);
            check_state(state);
        }
        printf("Frame: %d of %d - %2.1f%%\n",frame, nframes, 
               100*(float)frame/nframes);
        write_frame_data(fp, n, state, NULL);
    }
    double t_end = omp_get_wtime();
    printf("Ran in %g seconds\n", t_end-t_start);

    printf("Runtime,%g,Particles,%d,Threads,%d\n", t_end - t_start, state->n, omp_get_max_threads());

    fclose(fp);
    free_state(state);

    particle_bucket_lookup_cleanup();
    cleanup_buffers();
	for(int iter_particle = 0; iter_particle < state->n; ++iter_particle)
	{
		particle_t* cur_particle = state->part + iter_particle;
		omp_destroy_lock(&cur_particle->lock);
	}

#ifdef GPERF_DEBUG
    ProfilerStop();
#endif
}
