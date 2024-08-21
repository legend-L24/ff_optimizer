#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run example isotherm calculation on MIL-120 to do GCMC sample."""

from pathlib import Path
import os

from aiida import engine
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.orm import Dict, Str, load_code, SinglefileData
from aiida import load_profile

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / 'data'
load_profile()
# Workchain objects
IsothermWorkChain = WorkflowFactory('lsmo.isotherm')

# Data objects
CifData = DataFactory('core.cif')
NetworkParameters = DataFactory('zeopp.parameters')

RASPA_CODE_LABEL = 'raspa@lsmosrv5'
ZEOPP_CODE_LABEL = 'zeopp@lsmosrv4'

def run_isotherm(raspa_code, zeopp_code, cifdata):  # pylint: disable=redefined-outer-name
    builder = IsothermWorkChain.get_builder()

    builder.zeopp.code = zeopp_code
    builder.zeopp.metadata.options = {'max_wallclock_seconds': 60}

    builder.raspa_base.raspa.code = raspa_code
    builder.raspa_base.raspa.metadata.options = {
        'resources': {
            'num_machines': 1,
            'tot_num_mpiprocs': 1,
        },
        'max_wallclock_seconds': 1 * 60 * 60,
        'withmpi': False,
    }

    with open(os.path.join(DATA_DIR, cifdata), 'rb') as handle:
        cif = CifData(file=handle)
    
    builder.structure = cif
    builder.molecule = Str('co2')
    builder.parameters = Dict(
        {
            'ff_framework': 'UFF',  # Default: UFF
            "ff_separate_interactions": False,  # (bool) Use "separate_interactions" in the FF builder.
            "ff_mixing_rule": "Lorentz-Berthelot",  # (string) Choose 'Lorentz-Berthelot' or 'Jorgensen'.
            "ff_tail_corrections": True,  # (bool) Apply tail corrections.
            "ff_shifted": True,  # (bool) Shift or truncate the potential at cutoff.
            "ff_cutoff": 12.5,  # (float) CutOff truncation for the VdW interactions (Angstrom).
            'temperature': 303,  # (K) Note: higher temperature will have less adsorbate and it is faster
            'zeopp_volpo_samples': 10,  # Default: 1e5 *NOTE: default is good for standard real-case!
            'zeopp_block_samples': 1,  # Default: 100
            "zeopp_probe_scaling": 0,
            'raspa_widom_cycles': 100,  # Default: 1e5
            'raspa_gcmc_init_cycles': 1000,  # Default: 1e3
            'raspa_gcmc_prod_cycles': 6000,  # Default: 1e4
            "raspa_verbosity": 10,  # (int) Print stats every: number of cycles / raspa_verbosity.
            'pressure_list': [0.02164887307236063, 0.03825622775800713, 0.05901542111506525, 0.07769869513641753, 0.09638196915776986, 0.1482799525504152, 0.20017793594306055, 0.24584816132858844, 0.29774614472123384, 0.3475682087781732, 0.39739027283511275, 0.44928825622775814, 0.49495848161328604, 0.6008303677342824, 0.6983985765124557, 0.8021945432977464, 0.9018386714116253, 0.9973309608540928, 1.1986951364175567, 1.3959074733096088, 1.5993475682087785, 1.7986358244365368, 2.0020759193357063],
            #'pressure_list': [0.00705972, 0.02005115, 0.07155012, 0.13724632, 0.23730758],
            #'pressure_min': 0.001,  # Default: 0.001 (bar)
            #'pressure_max': 3,  # Default: 10 (bar)
        })
    
    extra_dic=Dict(
            dict={
                "Component": {
                "co2": {
                    "BlockPocketsFileName": "block_file",
                },
                },
                })
    block_pocket_node = SinglefileData(file="/home/yutao/project/aiida/ff_optim/data/out.block").store()
    builder.raspa_base.raspa.block_pocket = {"block_file":block_pocket_node,}
    builder.raspa_base.raspa.parameter = extra_dic

    results, node = engine.run_get_node(builder)

    assert node.is_finished_ok, results
    print("This is the final pk values: ", node.pk)
    return results, node
    

if __name__ == '__main__':
    run_isotherm(load_code(RASPA_CODE_LABEL), load_code(ZEOPP_CODE_LABEL),"MIL-120.cif")  # pylint: disable=no-value-for-parameter
