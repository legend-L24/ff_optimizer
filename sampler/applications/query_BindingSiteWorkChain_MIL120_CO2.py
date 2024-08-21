#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run example binding site work chain for CO2 in Zn-MOF-74 framework."""

import os
import click

from aiida.engine import run, run_get_node
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.orm import Dict, Str, SinglefileData, load_group
from aiida import cmdline

# Workchain objects
BindingSiteWorkChain = WorkflowFactory('lsmo.binding_site')  # pylint: disable=invalid-name

# Data objects
CifData = DataFactory('core.cif')  # pylint: disable=invalid-name


@click.command('cli')
@cmdline.utils.decorators.with_dbenv()
@click.option('--raspa_code', type=cmdline.params.types.CodeParamType())
@click.option('--cp2k_code', type=cmdline.params.types.CodeParamType())
def main(raspa_code, cp2k_code):
    """Prepare inputs and submit the work chain."""

    print('Testing BindingSite work chain (FF + DFT) for CO2 in Zn-MOF-74 ...')
    print('[NOTE: this test will run on 4 cpus and take ca. 10 minutes]')

    builder = BindingSiteWorkChain.get_builder()
    builder.metadata.label = 'test'
    builder.raspa_base.raspa.code = raspa_code
    builder.cp2k_base.cp2k.code = cp2k_code
    builder.raspa_base.raspa.metadata.options = {
        'resources': {
            'num_machines': 1,
            'tot_num_mpiprocs': 1,
        },
        'max_wallclock_seconds': 1 * 10 * 60,
    }
    builder.raspa_base.raspa.parameter = Dict(
            dict={
                "Component": {
                "co2": {
                    "BlockPocketsFileName": "block_file",
                },
                },
                })
    block_pocket_node = SinglefileData(file="/home/yutao/project/aiida/applications/data/out.block").store()
    #builder.raspa_base.raspa.code.block_pocket = {"block_co2":block_pocket_node,}
    builder.raspa_base.raspa.block_pocket = {"block_file":block_pocket_node,}

    builder.cp2k_base.cp2k.metadata.options = {
        'resources': {
            'num_machines': 1,
            'tot_num_mpiprocs': 4,
        },
        'max_wallclock_seconds': 1 * 10 * 60,
    }
    builder.structure = CifData(file=os.path.abspath('data/MIL-120.cif'), label='MIL-120')
    builder.molecule = Str('co2')
    builder.parameters = Dict(
        {
            'ff_framework': 'UFF',  # (str) Forcefield of the structure.
            'mc_steps': int(1),  # (int) Number of MC cycles.
            'temperature_list': [300, 250, 200, 250, 100, 50],
        })
    builder.protocol_tag = Str('test')
    builder.cp2k_base.cp2k.parameters = Dict({ # Lowering CP2K default setting for a faster test calculation
        'FORCE_EVAL': {
            'DFT': {
                'SCF': {
                    'EPS_SCF': 1.0E-4,
                    'OUTER_SCF': {
                        'EPS_SCF': 1.0E-4,
                    },
                },
            },
        },
        'MOTION': {
            'GEO_OPT': {
                'MAX_ITER': 1
            }
        },
    })

    results, node = run_get_node(builder)
    group = load_group(label='ff_optim')
    group.add_nodes(node)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter

# EOF
