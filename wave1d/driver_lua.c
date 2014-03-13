//ldoc on
/**
 * # Lua driver
 * 
 * It is often convenient to be able to change the parameters for
 * a simulation dynamically, without having to recompile.  Moreover,
 * fiddling with the parameters may be the type of thing that's most
 * easily done in a higher level language than C or Fortran.  This is
 * an example of one way to provide such functionality by embedding
 * an interpreter in the code.  In our case, we use the Lua language,
 * which is widely used as an embedded scripting and configuration
 * language in computer games, but originated as a configuration language
 * for finite element simulation codes.  An alternative would be to use
 * Python (and I may do this in a future assignment).
 * 
 * You don't strictly *need* to understand anything in this file,
 * and you aren't expected to modify it for your homework.  It is
 * provided purely in the hopes that you will find it an interesting
 * (and potentially useful) programming pattern.
 *  
 */
//ldoc off

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef USE_MPI
#include <mpi.h>
#else
#include <omp.h>
#define MPI_Wtime omp_get_wtime
#endif

#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"

#include "wave1d.h"

//ldoc on
/**
 * ## Initial conditions from Lua
 * 
 * Given a Lua function that computes $g(x)$ for a scalar number $x$,
 * the `get_lua_fx` function evaluates $g(x)$ and returns the result.
 * The context variable `ctx` is used to pass in the Lua interpreter.
 * 
 */
double get_lua_fx(double x, void* ctx)
{
    lua_State* L = (lua_State*) ctx;
    lua_pushvalue(L,-1);
    lua_pushnumber(L,x);
    lua_call(L,1,1);
    double fx = lua_tonumber(L,-1);
    lua_pop(L,1);
    return fx;
}

/**
 * ## Reading named parameters
 * 
 * Given a Lua interpreter `L` with a table at the top of the stack,
 * the `get_lua_double` routine
 * 
 *  - Tries to fetch a named field (name `s`) from the table
 *  - If the name is not present, returns a default value (`x0`)
 *  - If the named field is a number, returns the number
 *  - If the named field is not a number, throws an error
 * 
 * The `get_lua_int` routine does the same thing, but does an additional
 * check to make sure the number returned is an integer.
 * 
 */
double get_lua_double(lua_State* L, const char* s, double x0)
{
    lua_pushstring(L,s);
    lua_gettable(L,-2);
    if (lua_isnil(L,-1)) {
        lua_pop(L,1);
        return x0;
    } else if (lua_isnumber(L,-1)) {
        double x = lua_tonumber(L,-1);
        lua_pop(L,1);
        return x;
    } else {
        return luaL_error(L, "Unexpected non-numeric value for %s", s);
    }
}

int get_lua_int(lua_State* L, const char* s, int x0)
{
    double x = get_lua_double(L, s, x0);
    if (x != (int) x)
        return luaL_error(L, "Unexpected non-integer value for %s", s);
    return (int) x;
}


/**
 * ## Running and timing the simulation
 * 
 * The `lmain` routine is basically analogous to the `main` routine in
 * the C driver.  Rather than taking (positional) command line arguments,
 * it takes a table of named parameters.  This is most easily described
 * by example; see the `params.lua` file included with this code.
 * 
 */
int lmain(lua_State* L)
{
    int narg = lua_gettop(L);
    if (narg != 1 || !lua_istable(L,1)) 
        return luaL_error(L, "Incorrect argument");

    // Set up default values
    int n        = get_lua_int(L, "n",      100);
    int nsteps   = get_lua_int(L, "nsteps", 100);
    int fstep    = get_lua_int(L, "fstep",  10);
    int verbose  = get_lua_int(L, "verbose", 1);
    double a     = get_lua_double(L, "a",  0.0);
    double b     = get_lua_double(L, "b",  1.0);
    double c     = get_lua_double(L, "c",  1.0);
    double dt    = get_lua_double(L, "dt", 0.8*(b-a)/(n-1)/c);

    // Get rank and nprocs
    int proc, nproc;
#ifdef USE_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc);
#else
    proc = 0;
    nproc = 1;
#endif

    // Get file name
    const char* fname = NULL;
    lua_pushstring(L, "fname");
    lua_gettable(L, -2);
    if (lua_isstring(L,-1))
        fname = lua_tostring(L,-1);
    lua_pop(L,1);

    // Open output file
    FILE* fp = NULL;
    if (proc == 0 && fname != NULL) {
        fp = fopen(fname, "w+");
        if (!fp)
            luaL_error(L, "Could not open output file %s", fname);
    }

    // set initial conditions using indicator function
    lua_pushstring(L, "u0");
    lua_gettable(L, -2);
    if (!lua_isfunction(L,-1))
        return luaL_error(L, "u0 appears not to be a function");
    sim_t sim = sim_init(a, b, c, dt, n, proc, nproc, get_lua_fx, L);
    lua_pop(L,1);

    // Write header
    if (fp != NULL)
        sim_write_header(sim, nsteps/fstep, fp);

    // Run the PDE solver loop and time it 
    double start = MPI_Wtime();
    for (int step = 0; step < nsteps; ++step) {
        if (step % fstep == 0 && fname != NULL)
            sim_write_step(sim, fp);
        sim_advance(sim);
    }
    double t_elapsed = MPI_Wtime()-start;

    // Output the configuration and statistics.
    // The CFL condition (tau < 1) is needed for stability
    double dx  = (b-a)/(n-1);
    double tau = c * dt / dx;
    if (proc == 0 && verbose) {
        printf("n: %d\n"
               "nsteps: %d\n"
               "fsteps: %d\n"
               "c: %f\n"
               "dt: %f\n"
               "dx: %f\n"
               "CFL condition satisfied: %d\n"
               "nproc: %d\n"
               "Elapsed time: %f s\n", 
               n, nsteps, fstep, c, dt, dx, (tau < 1.0),
               nproc, t_elapsed);
    }

    if (fp != NULL)
        fclose(fp);
    sim_free(sim);
    
    lua_pushnumber(L, t_elapsed);
    return 1;
}

/**
 * With all the actual work done by `lmain`, the role of the `main` function
 * is now to launch a Lua interpreter, bind the `lmain` function to a name in
 * the Lua environment, and run a Lua script.
 * 
 */
int main(int argc, char** argv)
{
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
#endif

    if (argc != 2) {
        fprintf(stderr, "Usage: wave1d params.lua\n");
        return -1;
    }
    int status = 0;
    lua_State* L = luaL_newstate();
    luaL_openlibs(L);
    lua_register(L, "run", lmain);

    // Set rank and nproc for consistency with MPI version
    int proc, nproc;
#ifdef USE_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc);
#else
    proc = 0;
    nproc = 1;
#endif
    lua_pushnumber(L, proc);
    lua_setglobal(L, "rank");
    lua_pushnumber(L, nproc);
    lua_setglobal(L, "nproc");

    if (luaL_dofile(L, argv[1])) {
        fprintf(stderr, "%s\n", lua_tostring(L,-1));
        status = -1;
    }
    lua_close(L);

#ifdef USE_MPI
    MPI_Finalize();
#endif
    return status;
}
