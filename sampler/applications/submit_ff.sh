#!/usr/bin/bash

source /home/yutao/.aiida_venvs/aiida/bin/activate
cd $1 
./submit_ff.py | tee -a $2
#./submit_ff.py >> $2
