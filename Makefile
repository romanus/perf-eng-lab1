all: main.cpp
		gcc -march=native -O0 -g0 -std=c++14 -lstdc++ -o main main.cpp -I/usr/local/opt/openblas/include -L/usr/local/opt/openblas/lib -lopenblas