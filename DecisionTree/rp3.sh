#!/bin/bash

cd "$(dirname "$0")/CS6350-HW1/DecisionTree"
echo "curr directory: $(pwd)"
export PYTHONPATH=$(pwd)

python3 car_predict.py
python3 bank_predict.py
