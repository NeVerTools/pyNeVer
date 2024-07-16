#!/bin/bash

# NeVer2
# run_instance.sh script for VNN-COMP 2024

TOOL_NAME=NeVer2
VERSION_STRING=v1

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

CATEGORY=$2
ONNX_FILE=$3
VNNLIB_FILE=$4
RESULTS_FILE=$5
TIMEOUT=$6

echo "Running $TOOL_NAME on benchmark instance in category '$CATEGORY' with onnx file '$ONNX_FILE', vnnlib file '$VNNLIB_FILE', results file $RESULTS_FILE, and timeout $TIMEOUT"

# setup environment variable for tool (doing it earlier won't be persistent with docker)"
DIR=$(dirname $(dirname $(realpath $0)))
export PYTHONPATH="$PYTHONPATH:$DIR"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# run the tool to produce the results file
python3 -m never2_launcher -o "$RESULTS_FILE" -t "$TIMEOUT" "$ONNX_FILE" "$VNNLIB_FILE" ssbp