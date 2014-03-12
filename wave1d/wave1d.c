#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "wave1d.h"

//ldoc on
/**
 * # Simulation implementation
 * 
 * ## Simulation data
 * 
 * The main thing we need is storage for the field and ghost cells
 * at three successive times.  We store these in a 3-by-n+2 array,
 * where the local storage for time step k goes into slot k mod 3.  In
 * general, we will refer to steps relative to the current step, saved
 * in `step`.  We wrap up the relevant indexing in the `sim_get_u`
 * function below.
 * 
 * For the parallel version of the code, we want to keep track of which
 * processor is responsible for which part of the array.  We do this
 * with the `pidx` array: processor `k` owns the 
 * range `pidx[k] <= i < pidx[k+1]`.
 * 
 */
struct sim_data_t {

    // -- Simulation parameters
    double  a, b;   // End points of the domain
    double  c;      // Speed of sound
    double  dt;     // Time step
    double  dx;     // Space step

    // -- Parallel work allocation data
    int     proc;   // Index of the current processor
    int     nproc;  // Number of processors
    int*    pidx;   // Indexing for offsets to local data

    // -- Manage state at each time step
    double* ubuf;   // Buffer space for three frames
    int     nlocal; // Number of local points
    int     step;   // Step index

};


/**
 * ## Simulation initialization and teardown
 * 
 * The `sim_init` function sets up the simulation data structure and
 * allocates space for the three successive time steps.  It also
 * sets the initial condition by evaluating `initf(x, ctx)` at each
 * mesh coordinate `x`.
 * 
 */
sim_t sim_init(double a, double b, double c, double dt,
               int n, int proc, int nproc, 
               double (*initf)(double, void*), void* ctx)
{
    // Allocate sim object
    sim_t sim = (sim_t) malloc(sizeof(struct sim_data_t));

    // Initialize simulation parameter fields
    sim->a  = a;
    sim->b  = b;
    sim->c  = c;
    sim->dt = dt;
    sim->dx = (b-a)/(n-1);
    sim->step = 0;

    // Allocate n nodal points across nproc processors
    sim->proc  = proc;
    sim->nproc = nproc;
    sim->pidx = (int*) malloc( (nproc+1) * sizeof(int) );
    for (int i = 0; i <= nproc; ++i)
        sim->pidx[i] = (n*i)/nproc;
    int nlocal = sim->pidx[proc+1]-sim->pidx[proc];
    sim->nlocal = nlocal;

    // Allocate and clear storage
    sim->ubuf = (double*) malloc( 3*(nlocal+2) * sizeof(double) );
    memset(sim->ubuf, 0, 3*(nlocal+2) * sizeof(double));

    // Initial position set + apply BCs
    double* u = sim_get_u(sim, 0);
    for (int i = 1; i <= nlocal; ++i)
    	u[i] = initf(sim_get_x(sim,i), ctx);

    sim_apply_bc(sim);

    // Copy u to uold (i.e. u_t = 0 at t = 0)
    memcpy(sim_get_u(sim,-1), sim_get_u(sim,0), 
           (nlocal+2)*sizeof(double));

    return sim;
}


void sim_free(sim_t sim)
{
    free(sim->ubuf);
    free(sim->pidx);
    free(sim);
}


/**
 * ## Accessing simulation states
 * 
 * The `sim_get_u` function returns the state for time step `step+offset`.
 * Note that only the current step (offset 0) and previous step (offset 1)
 * are supposed to be in storage at any given time.
 * 
 */
double* sim_get_u(sim_t sim, int offset)
{
    return sim->ubuf + ((sim->step + offset + 3) % 3) * (sim->nlocal + 2);
}

/**
 * ## Getting local nodal coordinates
 * 
 * The `sim_get_x` function computes the coordinate of local node i,
 * where nodes 1 through nlocal are "real" nodes and nodes 0 an nlocal+1
 * are ghost nodes.
 * 
 */
double sim_get_x(sim_t sim, int i)
{
    int n = sim->pidx[sim->nproc]-1;
    int k = sim->pidx[sim->proc]+(i-1);
    return (sim->b * k + sim->a * (n-k))/n;
}


/**
 * ## Boundary conditions and ghost point exchange
 * 
 * For an acoustic wave, the boundary condition corresponding to a rigid
 * boundary is a homogeneous Neumann condition; i.e. the derivative of
 * the unknown field is zero at the boundary.  We implement this condition
 * by mirroring the numerical solution about the last real mesh point.
 * For example a real mesh point at index 1, we make the value
 * at the ghost point one to the right equal to the value of the real
 * point one to the left.  That way, the finite difference stencil used
 * to compute spatial derivatives at index 1 sees data consistent with 
 * the Neumann condition.
 * 
 * This is also where we do the ghost data exchange.  Each processor `p`
 * for `0 <= p < nproc-1` sends their right-most real data point to 
 * processor `p+1`, where it goes into the left ghost slot.  Each processor
 * `p` for `0 < p <= nproc-1` sends their left-most real data point to 
 * processor `p-1`, where it goes into the right ghost slot.  This can
 * be implemented by `MPI_Send` and `MPI_Recv` (or `MPI_Sendrecv`) calls.
 * 
 */
void sim_apply_bc(sim_t sim)
{
    double* u = sim_get_u(sim, 0);
    int nlocal = sim->nlocal;
    int proc   = sim->proc;
    int nproc  = sim->nproc;
    int step   = sim->step;

    // Boundary conditions
    if (sim->proc == 0)
        u[0] = u[2];
    if (sim->proc == sim->nproc-1)
        u[nlocal+1] = u[nlocal-1];

#ifdef USE_MPI
    /* 
     * Implement ghost cell exchange.
     * BEGIN TASK
    /* END TASK */
    if (nproc > 1) {
		double sendbuf;
		double recvbuf;
		MPI_Barrier(MPI_COMM_WORLD);
		if (nproc == 2) {
			if (proc == 0) {
				// send u[nlocal] to P1
				// receive u[nlocal+1] from P1
				printf("P%d sending %f to P%d\n", proc, u[nlocal], 1);
//				MPI_Send(&u[nlocal], 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
//				MPI_Recv(&u[nlocal+1], 1, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Sendrecv(&u[nlocal], 1, MPI_DOUBLE, 1, 0,
							&u[nlocal+1], 1, MPI_DOUBLE, 1, 1,
							MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				printf("P%d received %f from P%d\n", proc, sendbuf, 1);
			}
			else {
				// send u[1] to P0
				// receive u[0] from P0
				printf("P%d sending %f to P%d\n", proc, u[1], 0);
//				MPI_Recv(&u[0], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//				MPI_Send(&u[1], 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
				MPI_Sendrecv(&u[1], 1, MPI_DOUBLE, 0, 1,
							&u[0], 1, MPI_DOUBLE, 0, 0,
							MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				printf("P%d received %f from P%d\n", proc, u[0], 0);
			}

		}
/*
		if (proc % 2 == 0 && proc < nproc-1) {
			sendbuf = u[nlocal];
			MPI_Sendrecv(&sendbuf, 1, MPI_DOUBLE, (proc+1) % nproc, 0,
						&recvbuf, 1, MPI_DOUBLE, (proc+1) % nproc, 0,
						MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			u[nlocal+1] = recvbuf;
		} else if (proc > 0) {
			sendbuf = u[1];
			MPI_Sendrecv(&sendbuf, 1, MPI_DOUBLE, (proc+nproc-1) % nproc, 0,
						&recvbuf, 1, MPI_DOUBLE, (proc+nproc-1) % nproc, 0,
						MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			u[0] = recvbuf;
		}

		if (proc % 2 == 1 && proc < nproc-1) {
			sendbuf = u[nlocal];
			MPI_Sendrecv(&sendbuf, 1, MPI_DOUBLE, (proc+1) % nproc, 1,
						&recvbuf, 1, MPI_DOUBLE, (proc+1) % nproc, 1,
						MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			u[nlocal+1] = recvbuf;
		} else  if (proc > 0) {
			sendbuf = u[1];
			MPI_Sendrecv(&sendbuf, 1, MPI_DOUBLE, (proc+nproc-1) % nproc, 1,
						&recvbuf, 1, MPI_DOUBLE, (proc+nproc-1) % nproc, 1,
						MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			u[0] = recvbuf;
		}
*/
/*
		if (proc > 0 && proc < nproc-1)
		{
			// send to right, receive from left
			sendbuf = u[nlocal];
			printf("P%d sending %f to P%d\n", proc, sendbuf, (proc+1) % nproc);
			MPI_Sendrecv(&sendbuf, 1, MPI_DOUBLE, (proc+1) % nproc, 1,
					&recvbuf, 1, MPI_DOUBLE, (proc+nproc-1) % nproc, 1,
					MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			u[0] = recvbuf;
			printf("P%d received %f from P%d\n", proc, recvbuf, (proc+nproc-1) % nproc);
			// send to left, receive from right
			sendbuf = u[1];
			printf("P%d sending %f to P%d\n", sendbuf, (proc+nproc-1) % nproc);
			MPI_Sendrecv(&sendbuf, 1, MPI_DOUBLE, (proc+nproc-1) % nproc, 1,
					&recvbuf, 1, MPI_DOUBLE, (proc+1) % nproc, 1,
					MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			u[nlocal+1] = recvbuf;
			printf("P%d received %f from P%d\n", proc, recvbuf, (proc+1) % nproc);
		} else if (proc == 0) {
			// send to right
			sendbuf = u[nlocal];
			printf("P%d sending %f to P%d\n", proc, sendbuf, 1);
			MPI_Send(&sendbuf, 1, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);
		} else if (proc == nproc-1) {
			// send to left
			sendbuf = u[1];
			printf("P%d sending %f to P%d\n", proc, sendbuf, nproc-2);
			MPI_Send(&sendbuf, 1, MPI_DOUBLE, nproc-2, 1, MPI_COMM_WORLD);
		}
*/
		MPI_Barrier(MPI_COMM_WORLD);
    }
#endif
}


/**
 * ## Advance the time step
 * 
 * The `sim_advance` function computes a new step using the finite
 * difference formula
 * $$
 *   \frac{u(x,t-\Delta t)-2u(x,t)+u(x,t+\Delta t)}{\Delta t^2} 
 *   = c^2 \frac{u(x-\Delta x,t)-2u(x,t)+u(x+\Delta x,t)}{\Delta t^2}.
 * $$
 * With a little algebra, we rearrange this to get
 * $$
 *   u(x,t+\Delta t) = 2u(x,t) - u(x,t-\Delta t) + 
 *     \tau^2 [ u(x+\Delta x,t) - 2u(x,t) + u(x-\Delta x,t) ].
 * $$
 * where $\tau = c \, \Delta t/\Delta x$.
 * Then we apply the Neumann boundary conditions and update the
 * step counter.
 * 
 */
void sim_advance(sim_t sim)
{
    double* uold = sim_get_u(sim, -1);
    double* u    = sim_get_u(sim,  0);
    double* unew = sim_get_u(sim,  1);

    double  tau  = sim->c * sim->dt / sim->dx;
    double  tau2 = tau*tau;

    // Compute new step via finite differences
    int n = sim->nlocal;
    for (int i = 1; i <= n; ++i)
        unew[i] = 2*u[i] - uold[i] + tau2*(u[i-1] - 2*u[i] + u[i+1]);

    // Advance counter and apply BCs at new state
    sim->step++;
    sim_apply_bc(sim);
}


/**
 * ## File writer
 * 
 * In order to make it easier to debug (and in order for you to see what
 * the simulation is actually doing), we have provided a Javascript
 * visualization tool.  You should be able to use this in any modern
 * (WebGL-capable) browser: simply open `glwave1d.html` and tell the
 * resulting application to load the output file generated by the
 * simulation.
 * 
 * The output file format expected by the Javascript viewer is:
 * 
 *     WAVE1D nproc ncells nframes pidx[0] pidx[1] ... pidx[nproc-1]
 *     x coordinates of mesh points
 *     u values at mesh points for frame 1
 *     u values at mesh points for frame 2
 *     ...
 *     u values at mesh points for frame nframes
 * 
 * Note that `ncells` is *not* the same as the number of mesh points!
 * It is actually one less. 
 * 
 * The `sim_write_header` routine writes the first two lines;
 * the `sim_write_frame` routine then writes the current frame in
 * the simulation at different steps.  For `sim_write_frame`,
 * we use the `MPI_Gatherv` command to pull all the data in to
 * processor 0.
 * 
 * Note: I/O is expensive!  Unless there are many steps between frames,
 * writing the output file is likely to take much more time than the
 * actual simulation.
 * 
 */
void sim_write_header(sim_t sim, int tsteps, FILE* fp)
{
    if (sim->proc != 0)
        return;

    int nproc = sim->nproc;
    int* pidx = sim->pidx;
    int n = pidx[nproc];
    fprintf(fp, "WAVE1D %d %d %d", nproc, n-1, tsteps);
    for (int i = 0; i < nproc; ++i)
        fprintf(fp, " %d", pidx[i]);
    fprintf(fp, "\n");

    double a = sim->a;
    double b = sim->b;
    for (int i = 0; i < n; ++i)
        fprintf(fp, "%g ", (b*i+a*(n-1-i))/(n-1));
    fprintf(fp, "\n");
}

/**
 * 
 * We want to do I/O only at processor 0.  This means that we somehow have
 * to get all the relevant data to processor 0.  This task is trivial if
 * there is only one processor; for there is more than one processor, I
 * use the `MPI_Gatherv` primitive to gather all the data into a single
 * array at processor 0.  It's not really great for memory scalability,
 * but it's a simple solution for a toy problem.
 * 
 */
void sim_write_step(sim_t sim, FILE* fp)
{
#ifdef USE_MPI
    /* 
     * Gather data for file I/O at processor 0
     * BEGIN TASK
    /* END TASK */
    int proc = sim->proc;
	int nproc = sim->nproc;
	if (nproc > 1) {
		int nlocal = sim->nlocal;
		int* pidx = sim->pidx;
		int n = pidx[nproc];
		double *u = sim_get_u(sim, 0);
		double *sbuf = &u[1];
		double *rbuf;
		int *rcounts;
		if (proc == 0) {
			rbuf = (double*) malloc(n*sizeof(double));
			rcounts = (int*) malloc((nproc+1)*sizeof(int));
			for (int i=0; i < nproc; ++i) {
				rcounts[i] = pidx[i+1]-pidx[i];
			}
			rcounts[nproc] = 0;
		}
		MPI_Gatherv(sbuf, nlocal, MPI_DOUBLE, rbuf, rcounts, pidx, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		if (proc == 0) {
			for (int i = 0; i < n; ++i)
				fprintf(fp, "%g ", rbuf[i]);
			fprintf(fp, "\n");
			free(rbuf);
			free(rcounts);
		}
	}
#else
    int n = sim->nlocal;
    double* u = sim_get_u(sim, 0);
    for (int i = 1; i <= n; ++i)
        fprintf(fp, "%g ", u[i]);
    fprintf(fp, "\n");
#endif
}
