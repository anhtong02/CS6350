#!/bin/bash

cd "$(dirname "$0")/CS6350-HW2/EnsembleLearning"
echo "curr directory: $(pwd)"
export PYTHONPATH=$(pwd)

python3 AdaBoost.py
python3 Bagging.py