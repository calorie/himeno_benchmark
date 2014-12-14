#include <mpi.h>
#include <stdio.h>
#include "param.h"
#include "himenoBMTxps.h"

int main(int argc, char *argv[]) {
    int target = 60.0;
    int mx = MX0 - 1;
    int my = MY0 - 1;
    int mz = MZ0 - 1;

    MPI_Init(&argc, &argv);

    himeno_init(NDX0, NDY0, NDZ0, MIMAX, MJMAX, MKMAX, mx, my, mz);

    benchmark(target);

    himeno_free();
    MPI_Finalize();

    return 0;
}
