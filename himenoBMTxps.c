/********************************************************************

 This benchmark test program is measuring a cpu performance
 of floating point operation by a Poisson equation solver.

 If you have any question, please ask me via email.
 written by Ryutaro HIMENO, November 26, 2001.
 Version 3.0
 ----------------------------------------------
 Ryutaro Himeno, Dr. of Eng.
 Head of Computer Information Division,
 RIKEN (The Institute of Pysical and Chemical Research)
 Email : himeno@postman.riken.go.jp
 ---------------------------------------------------------------
 You can adjust the size of this benchmark code to fit your target
 computer. In that case, please chose following sets of
 (mimax,mjmax,mkmax):
 small : 33,33,65
 small : 65,65,129
 midium: 129,129,257
 large : 257,257,513
 ext.large: 513,513,1025
 This program is to measure a computer performance in MFLOPS
 by using a kernel which appears in a linear solver of pressure
 Poisson eq. which appears in an incompressible Navier-Stokes solver.
 A point-Jacobi method is employed in this solver as this method can
 be easyly vectrized and be parallelized.
 ------------------
 Finite-difference method, curvilinear coodinate system
 Vectorizable and parallelizable on each grid point
 No. of grid points : imax x jmax x kmax including boundaries
 ------------------
 A,B,C:coefficient matrix, wrk1: source term of Poisson equation
 wrk2 : working area, OMEGA : relaxation parameter
 BND:control variable for boundaries and objects ( = 0 or 1)
 P: pressure
********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "himenoBMTxps.h"

static float *p;
static int p_d1, p_d2, p_d3;
#define P(i, j, k) (p[p_d2 * p_d3 * i + p_d3 * j + k])
static float ****a, ****b, ****c;
static float ***bnd;
static float ***wrk1, ***wrk2;

static int id;
static int ndx, ndy, ndz;
static int imax, jmax, kmax;
static int mimax, mjmax, mkmax;
static int mx, my, mz;
static int ndims = 3, iop[3];
static int npx[2], npy[2], npz[2];
MPI_Comm mpi_comm_cart;
MPI_Datatype ijvec, ikvec, jkvec;

static float jacobi(int);
static int initmax(int, int, int);
static void initmt(int, int);
static void initcomm(int, int, int);
static void sendp(int, int, int);
static void sendp1();
static void sendp2();
static void sendp3();

static double fflop(int, int, int);
static double mflops(int, double, double);

static void malloc_p(float **arr, int p_d1, int p_d2, int p_d3);
static void malloc_3d(float ****arr, int d1, int d2, int d3);
static void malloc_4d(float *****arr, int d1, int d2, int d3, int d4);
static void free_3d(float ***arr, int d1, int d2);
static void free_4d(float ****arr, int d1, int d2, int d3);
static void malloc_failed();
static double rehearsal(double flop);
static void measure(int target, double cpu, double flop);

double fflop(int _mx, int _my, int _mz) {
    return ((double)(_mz - 2) * (double)(_my - 2) * (double)(_mx - 2) * 34.0);
}

double mflops(int nn, double cpu, double flop) {
    return (flop / cpu * 1.e-6 * (double)nn);
}

void initmt(int _mx, int it) {
    int i, j, k;
    malloc_4d(&a, 4, mimax, mjmax, mkmax);
    malloc_4d(&b, 3, mimax, mjmax, mkmax);
    malloc_4d(&c, 3, mimax, mjmax, mkmax);
    malloc_p(&p, mimax, mjmax, mkmax);
    malloc_3d(&bnd, mimax, mjmax, mkmax);
    malloc_3d(&wrk1, mimax, mjmax, mkmax);
    malloc_3d(&wrk2, mimax, mjmax, mkmax);

    for (i = 0; i < imax; ++i)
        for (j = 0; j < jmax; ++j)
            for (k = 0; k < kmax; ++k) {
                a[0][i][j][k] = 1.0;
                a[1][i][j][k] = 1.0;
                a[2][i][j][k] = 1.0;
                a[3][i][j][k] = 1.0 / 6.0;
                b[0][i][j][k] = 0.0;
                b[1][i][j][k] = 0.0;
                b[2][i][j][k] = 0.0;
                c[0][i][j][k] = 1.0;
                c[1][i][j][k] = 1.0;
                c[2][i][j][k] = 1.0;
                P(i, j, k) = (float)((i + it) * (i + it)) /
                             (float)((_mx - 1) * (_mx - 1));
                wrk1[i][j][k] = 0.0;
                wrk2[i][j][k] = 0.0;
                bnd[i][j][k] = 1.0;
            }
}

float jacobi(int nn) {
    int i, j, k, n;
    float gosa, wgosa, s0, ss;

    for (n = 0; n < nn; ++n) {
        gosa = 0.0;
        wgosa = 0.0;

        for (i = 1; i < imax - 1; ++i)
            for (j = 1; j < jmax - 1; ++j)
                for (k = 1; k < kmax - 1; ++k) {
                    s0 = a[0][i][j][k] * P(i + 1, j, k) +
                         a[1][i][j][k] * P(i, j + 1, k) +
                         a[2][i][j][k] * P(i, j, k + 1) +
                         b[0][i][j][k] *
                             (P(i + 1, j + 1, k) - P(i + 1, j - 1, k) -
                              P(i - 1, j + 1, k) + P(i - 1, j - 1, k)) +
                         b[1][i][j][k] *
                             (P(i, j + 1, k + 1) - P(i, j - 1, k + 1) -
                              P(i, j + 1, k - 1) + P(i, j - 1, k - 1)) +
                         b[2][i][j][k] *
                             (P(i + 1, j, k + 1) - P(i - 1, j, k + 1) -
                              P(i + 1, j, k - 1) + P(i - 1, j, k - 1)) +
                         c[0][i][j][k] * P(i - 1, j, k) +
                         c[1][i][j][k] * P(i, j - 1, k) +
                         c[2][i][j][k] * P(i, j, k - 1) + wrk1[i][j][k];

                    ss = (s0 * a[3][i][j][k] - P(i, j, k)) * bnd[i][j][k];
                    wgosa += ss * ss;

                    wrk2[i][j][k] = P(i, j, k) + 0.8 * ss;
                }

        for (i = 1; i < imax - 1; ++i)
            for (j = 1; j < jmax - 1; ++j)
                for (k = 1; k < kmax - 1; ++k) P(i, j, k) = wrk2[i][j][k];

        sendp(ndx, ndy, ndz);

        MPI_Allreduce(&wgosa, &gosa, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    } /* end n loop */

    return (gosa);
}

void initcomm(int ndx0, int ndy0, int ndz0) {
    int i, j, k, tmp, npe;
    int ipd[3], idm[3], ir;
    MPI_Comm icomm;
    MPI_Comm_size(MPI_COMM_WORLD, &npe);
    ndx = ndx0;
    ndy = ndy0;
    ndz = ndz0;

    if (ndx * ndy * ndz != npe) {
        if (id == 0) {
            printf("Invalid number of PE\n");
            printf("Please check partitioning pattern or number of PE\n");
        }
        MPI_Finalize();
        exit(0);
    }

    icomm = MPI_COMM_WORLD;

    idm[0] = ndx;
    idm[1] = ndy;
    idm[2] = ndz;

    ipd[0] = 0;
    ipd[1] = 0;
    ipd[2] = 0;
    ir = 0;

    MPI_Cart_create(icomm, ndims, idm, ipd, ir, &mpi_comm_cart);
    MPI_Cart_get(mpi_comm_cart, ndims, idm, ipd, iop);

    if (ndz > 1) {
        MPI_Cart_shift(mpi_comm_cart, 2, 1, &npz[0], &npz[1]);
    }
    if (ndy > 1) {
        MPI_Cart_shift(mpi_comm_cart, 1, 1, &npy[0], &npy[1]);
    }
    if (ndx > 1) {
        MPI_Cart_shift(mpi_comm_cart, 0, 1, &npx[0], &npx[1]);
    }
}

int initmax(int _mx, int _my, int _mz) {
    int i, tmp, it;
    int mx1[ndx + 1], my1[ndy + 1], mz1[ndz + 1];
    int mx2[ndx + 1], my2[ndy + 1], mz2[ndz + 1];

    tmp = _mx / ndx;
    mx1[0] = 0;
    for (i = 1; i <= ndx; i++) {
        if (i <= _mx % ndx)
            mx1[i] = mx1[i - 1] + tmp + 1;
        else
            mx1[i] = mx1[i - 1] + tmp;
    }
    tmp = _my / ndy;
    my1[0] = 0;
    for (i = 1; i <= ndy; i++) {
        if (i <= _my % ndy)
            my1[i] = my1[i - 1] + tmp + 1;
        else
            my1[i] = my1[i - 1] + tmp;
    }
    tmp = _mz / ndz;
    mz1[0] = 0;
    for (i = 1; i <= ndz; i++) {
        if (i <= _mz % ndz)
            mz1[i] = mz1[i - 1] + tmp + 1;
        else
            mz1[i] = mz1[i - 1] + tmp;
    }

    for (i = 0; i < ndx; i++) {
        mx2[i] = mx1[i + 1] - mx1[i];
        if (i != 0) mx2[i] = mx2[i] + 1;
        if (i != ndx - 1) mx2[i] = mx2[i] + 1;
    }
    for (i = 0; i < ndy; i++) {
        my2[i] = my1[i + 1] - my1[i];
        if (i != 0) my2[i] = my2[i] + 1;
        if (i != ndy - 1) my2[i] = my2[i] + 1;
    }
    for (i = 0; i < ndz; i++) {
        mz2[i] = mz1[i + 1] - mz1[i];
        if (i != 0) mz2[i] = mz2[i] + 1;
        if (i != ndz - 1) mz2[i] = mz2[i] + 1;
    }

    imax = mx2[iop[0]];
    jmax = my2[iop[1]];
    kmax = mz2[iop[2]];

    if (iop[0] == 0)
        it = mx1[iop[0]];
    else
        it = mx1[iop[0]] - 1;

    if (ndx > 1) {
        MPI_Type_vector(jmax, kmax, mkmax, MPI_FLOAT, &jkvec);
        MPI_Type_commit(&jkvec);
    }
    if (ndy > 1) {
        MPI_Type_vector(imax, kmax, mjmax * mkmax, MPI_FLOAT, &ikvec);
        MPI_Type_commit(&ikvec);
    }
    if (ndz > 1) {
        MPI_Type_vector(imax * jmax, 1, mkmax, MPI_FLOAT, &ijvec);
        MPI_Type_commit(&ijvec);
    }

    return (it);
}

void sendp(int ndx, int ndy, int ndz) {
    if (ndz > 1) sendp3();
    if (ndy > 1) sendp2();
    if (ndx > 1) sendp1();
}

void sendp3() {
    MPI_Status st[4];
    MPI_Request req[4];

    MPI_Irecv(&P(0, 0, kmax - 1), 1, ijvec, npz[1], 1, mpi_comm_cart, &req[0]);
    MPI_Irecv(&P(0, 0, 0), 1, ijvec, npz[0], 2, mpi_comm_cart, &req[1]);
    MPI_Isend(&P(0, 0, 1), 1, ijvec, npz[0], 1, mpi_comm_cart, &req[2]);
    MPI_Isend(&P(0, 0, kmax - 2), 1, ijvec, npz[1], 2, mpi_comm_cart, &req[3]);

    MPI_Waitall(4, req, st);
}

void sendp2() {
    MPI_Status st[4];
    MPI_Request req[4];

    MPI_Irecv(&P(0, jmax - 1, 0), 1, ikvec, npy[1], 1, mpi_comm_cart, &req[0]);
    MPI_Irecv(&P(0, 0, 0), 1, ikvec, npy[0], 2, mpi_comm_cart, &req[1]);
    MPI_Isend(&P(0, 1, 0), 1, ikvec, npy[0], 1, mpi_comm_cart, &req[2]);
    MPI_Isend(&P(0, jmax - 2, 0), 1, ikvec, npy[1], 2, mpi_comm_cart, &req[3]);

    MPI_Waitall(4, req, st);
}

void sendp1() {
    MPI_Status st[4];
    MPI_Request req[4];

    MPI_Irecv(&P(imax - 1, 0, 0), 1, jkvec, npx[1], 1, mpi_comm_cart, &req[0]);
    MPI_Irecv(&P(0, 0, 0), 1, jkvec, npx[0], 2, mpi_comm_cart, &req[1]);
    MPI_Isend(&P(1, 0, 0), 1, jkvec, npx[0], 1, mpi_comm_cart, &req[2]);
    MPI_Isend(&P(imax - 2, 0, 0), 1, jkvec, npx[1], 2, mpi_comm_cart, &req[3]);

    MPI_Waitall(4, req, st);
}

void benchmark(float target) {
    double flop = fflop(mz, my, mx);
    double cpu = rehearsal(flop);
    measure(target, cpu, flop);
}

double rehearsal(double flop) {
    int nn = 3;
    float gosa;
    double cpu, cpu0, cpu1;

    if (id == 0) {
        printf("Sequential version array size\n");
        printf(" mimax = %d mjmax = %d mkmax = %d\n", mx, my, mz);
        printf("Parallel version array size\n");
        printf(" mimax = %d mjmax = %d mkmax = %d\n", mimax, mjmax, mkmax);
        printf("imax = %d jmax = %d kmax =%d\n", imax, jmax, kmax);
        printf("I-decomp = %d J-decomp = %d K-decomp =%d\n", ndx, ndy, ndz);
        printf(" Start rehearsal measurement process.\n");
        printf(" Measure the performance in %d times.\n\n", nn);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    cpu0 = MPI_Wtime();
    gosa = jacobi(nn);
    cpu1 = MPI_Wtime() - cpu0;

    MPI_Allreduce(&cpu1, &cpu, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (id == 0) {
        printf(" MFLOPS: %f time(s): %f %e\n\n", mflops(nn, cpu, flop), cpu,
               gosa);
    }
    return cpu;
}

void measure(int target, double cpu, double flop) {
    int nn = (int)(target / (cpu / 3.0));
    float gosa;
    double cpu0, cpu1;

    if (id == 0) {
        printf(" Now, start the actual measurement process.\n");
        printf(" The loop will be excuted in %d times\n", nn);
        printf(" This will take about one minute.\n");
        printf(" Wait for a while\n\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    cpu0 = MPI_Wtime();
    gosa = jacobi(nn);
    cpu1 = MPI_Wtime() - cpu0;

    MPI_Allreduce(&cpu1, &cpu, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (id == 0) {
        printf("cpu : %f sec.\n", cpu);
        printf("Loop executed for %d times\n", nn);
        printf("Gosa : %e \n", gosa);
        printf("MFLOPS measured : %f\n", mflops(nn, cpu, flop));
        printf("Score based on Pentium III 600MHz : %f\n",
               mflops(nn, cpu, flop) / 82.84);
    }
}

void himeno_init(int _ndx, int _ndy, int _ndz, int _mimax, int _mjmax,
                 int _mkmax, int _mx, int _my, int _mz) {
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    ndx = _ndx;
    ndy = _ndy;
    ndz = _ndz;
    mimax = _mimax;
    mjmax = _mjmax;
    mkmax = _mkmax;
    mx = _mx;
    my = _my;
    mz = _mz;
    initcomm(_ndx, _ndy, _ndz);
    initmt(_mx, initmax(_mx, _my, _mz));
}

void himeno_free() {
    free_4d(a, 4, mimax, mjmax);
    free_4d(b, 3, mimax, mjmax);
    free_4d(c, 3, mimax, mjmax);
    free(p);
    free_3d(bnd, mimax, mjmax);
    free_3d(wrk1, mimax, mjmax);
    free_3d(wrk2, mimax, mjmax);
}

void malloc_p(float **arr, int _p_d1, int _p_d2, int _p_d3) {
    p_d1 = _p_d1;
    p_d2 = _p_d2;
    p_d3 = _p_d3;
    float *tmp;
    tmp = (float *)calloc(_p_d1 * _p_d2 * _p_d3, sizeof(float));
    if (!tmp) malloc_failed();
    *arr = tmp;
}

void malloc_3d(float ****arr, int d1, int d2, int d3) {
    int i, j;
    float ***tmp;
    tmp = (float ***)malloc(d1 * sizeof(float **));
    if (!tmp) malloc_failed();
    for (i = 0; i < d1; i++) {
        tmp[i] = (float **)malloc(d2 * sizeof(float *));
        if (!tmp[i]) malloc_failed();
        for (j = 0; j < d2; j++) {
            tmp[i][j] = (float *)calloc(d3, sizeof(float));
            if (!tmp[i][j]) malloc_failed();
        }
    }
    *arr = tmp;
}

void malloc_4d(float *****arr, int d1, int d2, int d3, int d4) {
    int i;
    float ****tmp;
    tmp = (float ****)malloc(d1 * sizeof(float ***));
    if (!tmp) malloc_failed();
    for (i = 0; i < d1; i++) malloc_3d(&tmp[i], d2, d3, d4);
    *arr = tmp;
}

void free_3d(float ***arr, int d1, int d2) {
    int i, j;
    for (i = 0; i < d1; i++) {
        for (j = 0; j < d2; j++) {
            free(arr[i][j]);
        }
        free(arr[i]);
    }
    free(arr);
}

void free_4d(float ****arr, int d1, int d2, int d3) {
    int i, j, k;
    for (i = 0; i < d1; i++) {
        for (j = 0; j < d2; j++) {
            for (k = 0; k < d3; k++) {
                free(arr[i][j][k]);
            }
            free(arr[i][j]);
        }
        free(arr[i]);
    }
    free(arr);
}

void malloc_failed() {
    printf("MALLOC FAILED.\n");
    MPI_Finalize();
    exit(1);
}
