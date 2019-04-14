# perf-eng-lab1

---

## Prerequisites

* 2 GB of RAM (1 GB is used while running)
* AVX2 commands are used
* `blas` should be linked properly in `Makefile`
* `chmod +x test.sh`

## Run

```
./test.sh
```

## Results

```
macbook893:perf-eng-lab1 trom$ ./test.sh 
gcc -march=native -O0 -g0 -std=c++14 -lstdc++ -o main main.cpp -I/usr/local/opt/openblas/include -L/usr/local/opt/openblas/lib -lopenblas

Vectors mult and add
raw      0.19s
raw      0.19s
raw      0.18s
vec      0.07s
vec      0.07s
vec      0.06s

Matrices mult
raw      11.89s
raw      9.24s
raw      7.54s
blas     0.09s
blas     0.07s
blas     0.07s
vec      2.76s
vec      2.95s
vec      2.97s

Short strings
raw      0.11s
vec      0.60s
Other strings
raw      2.61s
vec      1.66s
Long strings
raw      5.96s
vec      3.76s
```
