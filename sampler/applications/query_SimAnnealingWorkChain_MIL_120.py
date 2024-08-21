#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run example sim annealing of CO2 in Zn-MOF-74 framework."""

from pathlib import Path
import os
import click
import pytest
from aiida import engine
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.orm import Dict, Str, SinglefileData, load_group
from aiida import cmdline

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / 'data'

# Workchain objects
SimAnnealingWorkChain = WorkflowFactory('lsmo.sim_annealing')  # pylint: disable=invalid-name

# Data objects
CifData = DataFactory('core.cif')  # pylint: disable=invalid-name


@pytest.fixture(scope='function')
def zn_mof74_cifdata():
    """CifData for Zn MOF74 CIF."""
    with open(os.path.join(DATA_DIR, 'MIL-120.cif'), 'rb') as handle:
        cif = CifData(file=handle)

    return cif


def run_sim_annealing_zn_mof74(raspa_code, zn_mof74_cifdata):  # pylint: disable=redefined-outer-name
    """Prepare inputs and submit the work chain."""

    builder = SimAnnealingWorkChain.get_builder()
    builder.metadata.label = 'test'
    builder.raspa_base.raspa.code = raspa_code
    builder.raspa_base.raspa.metadata.options = {
        'resources': {
            'num_machines': 1,
            'tot_num_mpiprocs': 1,
        },
        'max_wallclock_seconds': 5 * 60 * 60,
    }
    pwd = os.path.dirname(os.path.realpath(__file__))
    builder.raspa_base.raspa.file ={
        "file_1": SinglefileData(file=os.path.join(pwd, "data", "force_field_mixing_rules.def")).store(),
        "file_2": SinglefileData(file=os.path.join(pwd, "data", "pseudo_atoms.def")).store(),
            }
    builder.raspa_base.raspa.parameter = Dict(
            dict={
                "Component": {
                "co2": {
                    "BlockPocketsFileName": "block_file",
                },
                },
                "GeneralSettings": {
                "Forcefield": "Local",
                },
                })

    block_pocket_node = SinglefileData(file="/home/yutao/project/aiida/applications/data/out_1.block").store()
    #builder.raspa_base.raspa.code.block_pocket = {"block_co2":block_pocket_node,}
    builder.raspa_base.raspa.block_pocket = {"block_file":block_pocket_node,}
    builder.structure = zn_mof74_cifdata
    builder.molecule = Str('co2')
    builder.parameters = Dict({
    'ff_framework': 'Local',  # (str) Forcefield of the structure.
        'mc_steps': int(1),  # (int) Number of MC cycles.
    })
    results, node = engine.run_get_node(builder)
    group = load_group(lable='test_group')
    group.add_nodes(node)
    assert node.is_finished_ok, results
    params = results['output_parameters'].get_dict()
    #assert params['energy_host/ads_tot_final'][-1] == pytest.approx(-36, abs=5)


@click.command()
@cmdline.utils.decorators.with_dbenv()
@click.option('--raspa-code', type=cmdline.params.types.CodeParamType())
def cli(raspa_code):
    """Run example.

    Example usage: $ ./test_SimAnnealingWorkChain_MOF74_CO2.py --raspa-code raspa-4467e14@fidis

    Help: $ ./test_SimAnnealingWorkChain_MOF74_CO2.py --help
    """
    with open(os.path.join(DATA_DIR, 'MIL-120.cif'), 'rb') as handle:
        cif = CifData(file=handle)

    run_sim_annealing_zn_mof74(raspa_code, cif)


if __name__ == '__main__':
    cli()  # pylint: disable=no-value-for-parameter
