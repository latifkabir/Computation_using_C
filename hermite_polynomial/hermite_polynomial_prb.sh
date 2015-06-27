#!/bin/bash
#
gcc -c -I/$HOME/include hermite_polynomial_prb.c
if [ $? -ne 0 ]; then
  echo "Errors compiling hermite_polynomial_prb.c"
  exit
fi
#
gcc hermite_polynomial_prb.o /$HOME/libc/$ARCH/hermite_polynomial.o -lm
if [ $? -ne 0 ]; then
  echo "Errors linking and loading hermite_polynomial_prb.o"
  exit
fi
#
rm hermite_polynomial_prb.o
#
mv a.out hermite_polynomial_prb
./hermite_polynomial_prb > hermite_polynomial_prb_output.txt
if [ $? -ne 0 ]; then
  echo "Errors running hermite_polynomial_prb."
  exit
fi
rm hermite_polynomial_prb
#
echo "Program output written to hermite_polynomial_prb_output.txt"
