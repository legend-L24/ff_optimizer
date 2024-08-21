#!/bin/bash

for i in {0..9}
do
    echo $i
    nohup python tenary_constpress.py $i >> output_humid.log &
done
