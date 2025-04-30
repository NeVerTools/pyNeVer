#!/bin/bash

# Download benchmarks from the VNN-LIB repository
if [ -z "$1" ]
  then
    echo "No cloning directory specified, reverting to default..."
    DIRECTORY="../RegressionBenchmarks"
  else
    DIRECTORY=$1
fi

echo "Downloading benchmarks in $DIRECTORY..."

git clone https://github.com/nevertools/RegressionBenchmarks "$DIRECTORY"