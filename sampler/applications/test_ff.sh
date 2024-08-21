#!/usr/bin/bash

source /home/yutao/.aiida_venvs/aiida/bin/activate
cd $1
nohup ./test_ff.py > $2 &
echo $!
