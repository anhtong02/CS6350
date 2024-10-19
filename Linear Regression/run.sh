#!/bin/bash

cd "$(dirname "$0")/CS6350-HW2/LinearRegression"
echo "curr directory: $(pwd)"
export PYTHONPATH=$(pwd)

python3 GradientDescent.py
