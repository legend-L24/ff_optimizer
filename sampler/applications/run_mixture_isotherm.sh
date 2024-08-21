#!/bin/bash

for i in {0..9}
do
    echo $i
    nohup python binary_isotherm.py $i >> output_binary.log &
    nohup python tenary_isotherm.py $i >> output_tenary.log &
done
