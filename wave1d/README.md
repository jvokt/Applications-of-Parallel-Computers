# Homework 2: 1D Wave Equation

-> **Due:Tuesday, Mar 11th by 11:59pm** <-

For this assignment, you will complete the missing pieces of a
parallel implementation for solving the 1D wave equation on a string.
The purpose of this homework is not to understand the numerical method
to solve the 1D wave equation, but to ensure that you:

* Understand how to write simple MPI code
* Can run MPI jobs on the C4 cluster
* Grasp basic performance modeling ideas
* Can perform timing experiments and profile parallel code

This is an individual assignment, and you should produce your own
writeup (not as a team) and code for submission on CMS.  You may ask
the instructional staff or others in the class for help, but you
should acknowledge any assistance you receive in your write-up.
Everybody should get familiar enough with all the tasks to be able to
do them independently.  When you complete the homework you should
submit two files on [CMS][cms]: `wave1d.c`, and a brief write-up
(`writeup.pdf`) in which you describe your results and provide some
figures from your performance experiments.

## Framework

The reference code for the assignment can be found in the `wave1d`
folder under the class repository.  You can read the code, along with
the prompt describing your tasks, either from the repository,
from the [code web page][webhw], or [as a PDF][pdfhw].

The following files are provided:

* `wave1d.c`: a reference (serial) implementation of the numerical code
* `driver_lua.c`: a driver for calling the routines in `wave1d.c`
* `glwave1d.html`: a webgl viewer for the simulation output
* `Makefile`: a Makefile with rules for building the code

Please consult `USAGE.md` for more information on how to run
and pass parameters to the simulation.

You can load the output file using `glwave1d.html` to animate the
output of the 1D string simulation.

## Further Resources

* This assignment borrows from the F11 assignment.
* "Solving Problems on Concurrent Processors, Vol 1" by G.Fox

[cms]: http://cms.csuglab.cornell.edu/web/guest
[webhw]: http://www.cs.cornell.edu/~bindel/class/cs5220-s14/html/hw2.html
[pdfhw]: http://www.cs.cornell.edu/~bindel/class/cs5220-s14/hw2.pdf
