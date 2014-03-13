//ldoc
/**
 * # Simulation interface
 * 
 * The simulation object encapsulates the simulation state and
 * operations that act on it (advancing by a time step, getting
 * the state at the current step, writing the state to disk).
 * 
 */
#ifndef WAVE1D_H
#define WAVE1D_H

#include <stdio.h>

// Opaque data time for simulation data
typedef struct sim_data_t* sim_t;

// Initialize simulation data.
// - a,b   = end points of the domain
// - c     = speed of sound
// - dt    = time step
// - n     = global number of non-ghost mesh points
// - proc  = current processor
// - nproc = number of processors involved
// - initf = function returning the initial position
// - ctx   = context variable passed into initf
//
sim_t sim_init(double a, double b, double c, double dt,
               int n, int proc, int nproc, 
               double (*initf)(double, void*), void* ctx);

// Free simulation data object
void sim_free(sim_t sim);

// Access x coordinate of ith local node
double sim_get_x(sim_t sim, int i);

// Access current step (+ offset of 0, -1, or 1)
double* sim_get_u(sim_t sim, int offset);

// Apply boundary conditions to the current state
void sim_apply_bc(sim_t sim);

// Advance to the next step
void sim_advance(sim_t sim);

// Write header metadata or current frame to output file
void sim_write_header(sim_t sim, int tsteps, FILE* fp);
void sim_write_step(sim_t sim, FILE* fp);

#endif /* WAVE1D_H */
