[Himeno benchmark](http://accc.riken.jp/2444.htm)
=======
C + MPI, dynamic allocate version

# Usage

```
$ cp Makefile.sample Makefile
$ ./paramset.sh M 1 1 2
$ make
$ mpirun -np 2 ./bmt
```

# Lisence

LGPL2.0 or later
