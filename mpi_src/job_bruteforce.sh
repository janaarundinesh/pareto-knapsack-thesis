#!/bin/bash

mpic++ -std=c++17 \
    mpi_brute_force.cpp \
    ../src/dominance.cpp \
    ../src/brute_force_maxima.cpp \
    -I../include \
    -o MPI_BruteForce

if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi

echo "Compilation successful."
time mpirun -np 1 ./MPI_BruteForce