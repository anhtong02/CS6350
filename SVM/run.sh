#!/bin/bash

cd "$(dirname "$0")/CS5350-HW4/SVM"
echo "curr directory: $(pwd)"
export PYTHONPATH=$(pwd)

python3 SVM.py