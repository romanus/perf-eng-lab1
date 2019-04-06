all: mult.cpp
		g++ -O0 -g3 -fno-omit-frame-pointer -std=c++14 -o mult_gen ./mult.cpp
		g++ -O1 -g3 -march=skylake -ftree-vectorize -fno-omit-frame-pointer -std=c++14 -o mult_nat ./mult.cpp 