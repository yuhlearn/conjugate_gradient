
## General

Parallel Conjugate Gradient method using MPI


## Compilation

`make clean` removes the compiled executable `cg`.

`make cg` compiles the code with no funny business.

`make CFLAGS=-DVERIFY cg` and max error is printed when `cg` runs.

`make CFLAGS=-DDEBUG cg` and lots of debug info is printed when `cg` runs.
Including all matrices (if the matrix dimension is no larger than 20) and
comparison to the sequential max error. Takes much longer than other versions.

NOTE: If you compile with one of the flags, you will need to run 
`make clean` to recompile, unless the code has been changed.


## Running

`mpirun -np <p> ./cg <N>`, where `<p>` is the desired number of processes
and `<N>` is the problem size - i.e. the N in the NxN matrix generated.
