# perf-eng-lab1

---

## Prerequisites

* AVX2 commands are used
* `blas` should be linked properly in `Makefile`
* `chmod +x test.sh`

## Run

```
./test.sh
```

## Results

```
laptop:perf-eng-lab1 trom$ ./test.sh 
gcc -march=native -O0 -g0 -std=c++14 -lstdc++ -o main main.cpp -I/usr/local/opt/openblas/include -L/usr/local/opt/openblas/lib -lopenblas
Vectors mult and add
raw      0.15s
raw      0.15s
raw      0.14s
vec      0.06s
vec      0.27s
vec      0.07s
Matrices mult
raw      13.58s
raw      13.42s
raw      16.13s
blas     0.11s
blas     0.11s
blas     0.11s
vec      3.29s
vec      2.98s
vec      3.03s
```