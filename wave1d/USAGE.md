# 1D wave equation simulator: 

To run the serial code on the front-end node (for testing):

    ./wave1d.x params.lua

where `params.lua` is a parameter file.  Available parameters are

 * `fname` - output file name (if output is desired)
 * `n` - number of mesh points
 * `nsteps` - number of time steps
 * `fstep` - time steps between output frames
 * `verbose` - flag for verbose output (default)
 * `a`, `b` - end points of the simulation domain
 * `c` - speed of sound
 * `dt` - time step

For timing experiments, we recommend using the `params_time.lua` script.

For timing experiments, please use the C4 compute nodes.  You can
submit an MPI job on the compute nodes using `mpisub`, e.g.

    mpisub -n 2 ./wave1d_mpi.x params_time.lua

If you want to time the serial version, you can do

    csub ./wave1d.x params_time.lua

The `mpisub` command will prefer to run processes on the
same node, but is allowed to run on different nodes.  To force
the processes to run on the same node, please use `ompsub`, e.g.

    ompsub -n 2 mpirun -n 2 ./wave1d_mpi.x params_time.lua

You can visualize the results by opening the output file loading the
file using glwave1d.html with a webgl enabled browser
