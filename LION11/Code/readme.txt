
- OpenMP support of the UBCSAT for the improving parallel local search fro sat
- implementation of parallel and cooperative algorithms for SAT
===================

This is a special build of ubcsat that includes a MIP support (for parallel and distributed executions with and without cooperation of the SAT algorithms)
It is based on a preliminary beta version (1.2b10) of ubcsat version 1.2



There are two options to compile the solver. The first one compiles CSLS with the static linking option:

make static_prob
or
make static_pnorm

The second one (in case of a failure of the previous one) compiles the solver without the "-static" flag

make prob
or
make pnorm
