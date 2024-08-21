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
load_profile()

# Workchain objects
MulticompGcmcWorkChain = WorkflowFactory('lsmo.multicomp_gcmc')  # pylint: disable=invalid-name

# Data objects
CifData = DataFactory('cif')  # pylint: disable=invalid-name
NetworkParameters = DataFactory('zeopp.parameters')  # pylint: disable=invalid-name
SinglefileData = DataFactory('singlefile')

ff_data = 'Al_graphite.json'#'/home/yutao/project/aiida/applications/Al_graphite.json'

@click.command('cli')
@click.argument('raspa_code_label')
@click.argument('zeopp_code_label')
def main(raspa_code_label, zeopp_code_label):
    """Prepare inputs and submit the workchain.
    verdi run run_Ternary.py raspa@lsmosrv7 zeopp@lsmosrv7"""
    
    builder = MulticompGcmcWorkChain.get_builder()

    builder.metadata.label = 'Nency/Ternary'

    builder.raspa_base.raspa.code = Code.get_from_string(raspa_code_label)
    builder.zeopp.code = Code.get_from_string(zeopp_code_label)

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
    
    pkslist = []
    strucorder = []
    
    for i in range(len(structures)):
        time.sleep(3)
        builder.structure = CifData(file=os.path.abspath('CIFs/' + structures[i] + '.cif'), label= structures[i])
        builder.conditions = Dict(dict={
            'molfraction': {
                'co2': 0.15,
                'n2': 0.79,
                'h2o': 0.06,
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
                    'ff_optim': json.load(open(ff_data)),
                    })

        wc = submit(builder)
        strucorder.append(structures[i])
        pkslist.append(wc.pk)

    Reference = {
            "Structure" : strucorder,
            "PK" : pkslist
         }

    with open("PKs_Ternary_Nency_new.json", "w") as outfile:
        json.dump(Reference, outfile)
    outfile.close()

if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter

# EOF
