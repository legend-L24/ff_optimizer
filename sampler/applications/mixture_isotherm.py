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
load_profile()
# Workchain objects
MulticompGcmcWorkChain = WorkflowFactory('lsmo.multicomp_gcmc')  # pylint: disable=invalid-name

# Data objects
CifData = DataFactory('cif')  # pylint: disable=invalid-name
NetworkParameters = DataFactory('zeopp.parameters')  # pylint: disable=invalid-name
SinglefileData = DataFactory('singlefile')
RASPA_CODE_LABEL = 'raspa@lsmosrv5'
ZEOPP_CODE_LABEL = 'zeopp@lsmosrv5'


def main():
    """Prepare inputs and submit the workchain.
    verdi run run_Binary.py raspa@lsmosrv5 zeopp@lsmosrv5"""

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
    
    
    builder.structure = CifData(file=os.path.abspath('CIFs/' + structures[0] + '.cif'), label= structures[0])
    builder.conditions = Dict(dict={
        'molfraction': {
            'co2': 0.15,
            'n2': 0.85,
        },
        'temp_press': [
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

    wc = submit(builder)
    print("Submitted workchain; PK: ", wc)

    builder.structure = CifData(file=os.path.abspath('CIFs/' + structures[1] + '.cif'), label= structures[0])
    builder.conditions = Dict(dict={
        'molfraction': {
            'co2': 0.15,
            'n2': 0.85,
        },
        'temp_press': [
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

    wc = submit(builder)
    print("Submitted workchain; PK: ", wc)
if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter

# EOF
