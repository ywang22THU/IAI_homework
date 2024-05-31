#!/bin/bash

for lambda in 0.1 0.001 0.0001 0.00001 0.000001 0.0000001 0.00000001; do
    for eta in 0.1 0.001 0.0001 0.00001 0.000001 0.0000001 0.00000001; do
        echo "running lambda = $lambda, eta = $eta"
        python3 pinyin.py 0 ../data/std_input.txt ../data/output.txt ../data/std_output.txt $lambda $eta >> accuracy.txt
    done
done