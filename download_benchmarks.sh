#!/bin/bash

# Download benchmarks from the VNN-LIB repository

if [ -z "$1" ]
  then
    echo "Please select the directory to download the benchmarks"
    exit 1
fi

DIRECTORY=$1

echo "Downloading $CATEGORY VNN-LIB benchmarks in ./$DIRECTORY/..."

git clone https://github.com/AndyVale/benchmarks_vnncomp "$DIRECTORY"