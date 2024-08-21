#!/usr/bin/env python  # pylint: disable=invalid-name
# -*- coding: utf-8 -*-

import os
import json
import time
import click

from aiida.engine import run
from aiida.engine import submit
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.orm import Code, Dict
from aiida import load_profile
from aiida.engine import run_get_node
import sys
load_profile()
# Workchain objects
MulticompGcmcWorkChain = WorkflowFactory('lsmo.multicomp_gcmc')  # pylint: disable=invalid-name

# Data objects
CifData = DataFactory('cif')  # pylint: disable=invalid-name
NetworkParameters = DataFactory('zeopp.parameters')  # pylint: disable=invalid-name
SinglefileData = DataFactory('singlefile')
RASPA_CODE_LABEL = 'raspa@lsmosrv6'
ZEOPP_CODE_LABEL = 'zeopp@lsmosrv5'


def main(idx, ratio_h2o):
    """Prepare inputs and submit the workchain.
    verdi run run_Binary.py raspa@lsmosrv5 zeopp@lsmosrv5"""
    idx = int(idx)
    builder = MulticompGcmcWorkChain.get_builder()

    builder.metadata.label = 'Nency/Binary'

    builder.raspa_base.raspa.code = Code.get_from_string(RASPA_CODE_LABEL)
    builder.zeopp.code = Code.get_from_string(ZEOPP_CODE_LABEL)

    options = {
        'resources': {
            'num_machines': 1,
            'tot_num_mpiprocs': 1,
        },
        'max_wallclock_seconds': 96 * 60 * 60,
        'withmpi': False,
    }
    builder.raspa_base.raspa.metadata.options = options
    builder.zeopp.metadata.options = options

    pathstructures = os.getcwd()
    structures = []
    for f in os.listdir(pathstructures + "/CIFs"):
        if f.endswith(".cif"):
            structures.append(f.split('.cif')[0])
    
    builder.structure = CifData(file=os.path.abspath('CIFs/' + structures[idx] + '.cif'), label= structures[idx])
    builder.conditions = Dict(dict={
        'molfraction': {
            'h2o': ratio_h2o,
            'co2': 0.15,
            'n2': 0.79,
        },
        'temp_press': [
        [313.15, 1.01325],
        ]   
    })

    builder.parameters = Dict(
            dict={
                'ff_framework': 'UFF',  # Default: UFF
                'zeopp_probe_scaling': 0.8,
                'zeopp_block_samples': 100,  # Default: 100
                'raspa_gcmc_init_cycles': 15000,  # Default: 1e3
                'raspa_gcmc_prod_cycles': 150000,  # Default: 1e4
                #'ff_optim': json.load(open('Al_graphite.json')),
                })
    results, node = run_get_node(builder)
    print(f"Submitted workchain: PK: {node.pk}, {structures[idx]}")
    with open("PKs_Nency_humid_highpress_tenth.csv", "a+") as outfile:
        outfile.write(f"{node.pk}, {structures[idx]}\n")
    outfile.close()
if __name__ == '__main__':
    for scale in range(10, 16):
        scale = scale/10
        ratio = 0.046*scale#0.07315
        main(sys.argv[1], ratio)  # pylint: disable=no-value-for-parameter

# EOF
