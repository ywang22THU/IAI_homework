#!/bin/bash

for lambda in 0.9 0.5 0.1 0.01 0.001 0.0001 0.00001 0.000001 0.0000001; do
    echo "running lambda = $lambda"
    python3 pinyin.py 0 ./std_input.txt ./output.txt ./std_output.txt $lambda >> accuracy.txt
done