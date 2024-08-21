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
ZEOPP_CODE_LABEL = 'zeopp@lsmosrv6'


def main(idx):
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
        'max_wallclock_seconds': 72 * 60 * 60,
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
            'co2': 0.15,
            'n2': 0.85,
        },
        'temp_press':[
        [313.15, 0.0506625],
        [313.15, 0.101325],
        [313.15, 0.1519875],
        [313.15, 0.20265],
        [313.15, 0.2533125],
        [313.15, 0.303975],
        [313.15, 0.3546375],
        [313.15, 0.4053],
        [313.15, 0.4559625],
        [313.15, 0.506625],
        [313.15, 0.5572875],
        [313.15, 0.60795],
        [313.15, 0.6586125],
        [313.15, 0.709275],
        [313.15, 0.7599375],
        [313.15, 0.8106],
        [313.15, 0.8612625],
        [313.15, 0.911925],
        [313.15, 0.9625875],
        [313.15, 1.01325]
        ]  
    })

    builder.parameters = Dict(
            dict={
                'ff_framework': 'UFF',  # Default: UFF
                'zeopp_probe_scaling': 0.8,
                'zeopp_block_samples': 100,  # Default: 100
                'raspa_gcmc_init_cycles': 15000,  # Default: 1e3
                'raspa_gcmc_prod_cycles': 15000,  # Default: 1e4
                #'ff_optim': json.load(open('Al_graphite.json')),
                })
    results, node = run_get_node(builder)
    print(f"Submitted workchain: PK: {node.pk}, {structures[idx]}")
    with open("PKs_Binary_Nency_whole.csv", "a+") as outfile:
        outfile.write(f"{node.pk}, {structures[idx]}\n")
    outfile.close()
if __name__ == '__main__':
    main(sys.argv[1])  # pylint: disable=no-value-for-parameter

# EOF
