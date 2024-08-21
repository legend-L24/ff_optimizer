#!/usr/bin/bash

source /home/yutao/.aiida_venvs/aiida/bin/activate
cd ~/project/aiida/applications
./isotherm_workchain.py #>> sample_long.txt
