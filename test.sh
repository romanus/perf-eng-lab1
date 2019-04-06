#!/bin/bash

# build
make

# run
echo "Common architecture"
./mult_gen
echo "Specific architecture"
./mult_nat

# clean up
rm -f ./mult_gen
rm -f ./mult_nat
rm -rf ./mult_gen.dSYM
rm -rf ./mult_nat.dSYM