#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run example isotherm calculation on MIL-120 to do GCMC sample."""

from pathlib import Path
import os
import sys
from aiida import orm
from aiida import engine
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.orm import Dict, Str, load_code, SinglefileData, CifData, load_code, load_group, Str
from aiida import load_profile
import json
import exp_config

Temperature = exp_config.Temperature
pressure_list = exp_config.pressure_list
cif_path = exp_config.cif_path
ff_data = exp_config.ff_data
molecule = exp_config.molecule

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / 'data'
load_profile()
# Workchain objects
IsothermWorkChain = WorkflowFactory('lsmo.isotherm')
# Workchain objects
BindingSiteWorkChain = WorkflowFactory('lsmo.binding_site')  # pylint: disable=invalid-name

# Data objects
CifData = DataFactory('core.cif')
NetworkParameters = DataFactory('zeopp.parameters')
WORKCHAIN_LABEL = 'bindingsite'
RASPA_CODE_LABEL = 'raspa@lsmosrv6'
ZEOPP_CODE_LABEL = 'zeopp@lsmosrv6'
PROTOCOL_FILE = os.path.abspath('/home/yutao/project/aiida/ff_optim/multistage_cualpyc_supercell_test.yaml')
CP2K_CODE_LABEL = 'cp2k-new@jed'
def submit_isotherm(raspa_code, zeopp_code, cifdata):  # pylint: disable=redefined-outer-name
    global pressure_list
    ZEOPP_OPTIONS = {
            'resources': {
                'num_machines': 1,
                'tot_num_mpiprocs': 1,
            },
            'max_wallclock_seconds': 10 * 60,
            'withmpi': False,
    }
    
    builder = IsothermWorkChain.get_builder()

    builder.zeopp.code = zeopp_code
    builder.zeopp.metadata.options = ZEOPP_OPTIONS

    builder.raspa_base.raspa.code = raspa_code
    builder.raspa_base.raspa.metadata.options = {
        'resources': {
            'num_machines': 1,
            'tot_num_mpiprocs': 1,
        },
        'max_wallclock_seconds': 12 * 60 * 60,
        'withmpi': False,
    }

    with open(os.path.join(DATA_DIR, cifdata), 'rb') as handle:
        cif = CifData(file=handle)
    
    builder.structure = cif
    builder.molecule = Str(molecule)
    builder.parameters = Dict(
        {
            'ff_framework': 'UFF',#'DREIDING',  # Default: UFF
            "ff_separate_interactions": False,  # (bool) Use "separate_interactions" in the FF builder.
            "ff_mixing_rule": "Lorentz-Berthelot",  # (string) Choose 'Lorentz-Berthelot' or 'Jorgensen'.
            "ff_tail_corrections": True,  # (bool) Apply tail corrections.
            "ff_optim": json.load(open(ff_data)),
            "ff_shifted": True,  # (bool) Shift or truncate the potential at cutoff.
            "ff_cutoff": 12.8,  # (float) CutOff truncation for the VdW interactions (Angstrom).
            'temperature': Temperature,  # (K) Note: higher temperature will have less adsorbate and it is faster
            'zeopp_volpo_samples': 100000,  # Default: 1e5 *NOTE: default is good for standard real-case!
            'zeopp_block_samples': 100,  # Default: 100
            "zeopp_probe_scaling": 0.8, #0.0 is necessary for ZIDDIB
            'raspa_widom_cycles': 1000,  # Default: 1e5
            'raspa_gcmc_init_cycles': 15000,  # Default: 1e3
            'raspa_gcmc_prod_cycles': 15000,  # Default: 1e4
            "raspa_verbosity": 10,  # (int) Print stats every: number of cycles / raspa_verbosity.
            'pressure_list': pressure_list,
            #'pressure_list': [0.00705972, 0.02005115, 0.07155012, 0.13724632, 0.23730758],
            #'pressure_min': 0.001,  # Default: 0.001 (bar)
            #'pressure_max': 3,  # Default: 10 (bar)
        })
    
    extra_dic=Dict(
            dict={
                "Component": {
                molecule: {
                    "BlockPocketsFileName": "block_file",
                },
                },
                })
    block_pocket_node = SinglefileData(file="/home/yutao/project/aiida/ff_optim/data/out.block").store()
    builder.raspa_base.raspa.block_pocket = {"block_file":block_pocket_node,}
    builder.raspa_base.raspa.parameter = extra_dic
    wc = engine.submit(builder)
    print("This is the final pk values for isotherm workflow: ", wc.pk)

def submit_bindingsite(cif_filename):

    cifdata = CifData(file=os.path.abspath(cif_filename))

    filepath = Path(cif_filename)
    structure_label = filepath.name.rstrip('.cif')

    RASPA_OPTIONS = {
            'resources': {
                'num_machines': 1,
                'tot_num_mpiprocs': 1,
            },
            'max_wallclock_seconds': 48 * 60 * 60,
            'withmpi': True,
    }

    CP2K_OPTIONS = {
        'resources': {
            'num_machines': 1,
            'num_mpiprocs_per_machine': 72
        },
        'max_wallclock_seconds': 48 * 60 * 60,
        'withmpi': True,
    }

    builder = BindingSiteWorkChain.get_builder()    
    
    cifdata.label = structure_label
    builder.structure = cifdata
    
    builder.metadata.label = WORKCHAIN_LABEL

    builder.raspa_base.raspa.code = load_code(RASPA_CODE_LABEL)
    builder.raspa_base.raspa.metadata.options = RASPA_OPTIONS

    builder.protocol_yaml = SinglefileData(file=PROTOCOL_FILE)
    
    builder.cp2k_base.cp2k.code = load_code(CP2K_CODE_LABEL)
    builder.cp2k_base.cp2k.metadata.options = CP2K_OPTIONS
    builder.cp2k_base.cp2k.parameters = Dict(dict={
        'GLOBAL': {
            'PREFERRED_DIAG_LIBRARY': 'SL'
            },  
        'MOTION': {
            'GEO_OPT': {
                'MAX_FORCE': 5.0e-6,
            },
        },
    },)

    if molecule == 'ch4-mod':
        print("I use specific parameters for CH4")
        builder.molecule = Dict(
        dict={
            'name': 'CH4',
            'forcefield': 'TraPPE-mod', 
            'molsatdens':27.2,
            'proberad':1.865,
            'singlebead': False,
            'charged': False,
        })
        
        builder.parameters = Dict(
            dict={
                'ff_framework': 'UFF',  # (str) Forcefield of the structure.'UFF'
                'ff_tail_corrections': True, # appyl tail corrections
                'ff_shifted': False,
                'ff_separate_interactions': True,
                'ff_cutoff': 12.8,
                'mc_steps': int(10000),  # 100000(int) Number of MC cycles.
                'temperature_list': [300, 250, 200, 150, 100, 50],
                'number_of_molecules': int(1),
                'ff_optim': json.load(open(ff_data)),
            })
    else:

        builder.molecule = Str(molecule)
        builder.parameters = Dict(
            dict={
                'ff_framework': 'UFF',  # (str) Forcefield of the structure.'UFF'
                'ff_tail_corrections': True, # appyl tail corrections
                'ff_shifted': False,
                #'ff_separate_interactions': True,
                'ff_cutoff': 12.8,
                'mc_steps': int(10000),  # 100000(int) Number of MC cycles.
                'temperature_list': [300, 250, 200, 150, 100, 50],
                'number_of_molecules': int(1),
                'ff_optim': json.load(open(ff_data)),
            })
    
    wc = engine.submit(builder)
    print("This is the final pk values for binding sites workflow: ", wc.pk)

if __name__ == '__main__':
    print("The tested force field is: ", ff_data)
    print("The tested cif is: ", cif_path)
    print("The simulation temperature is: ", Temperature)
    submit_isotherm(load_code(RASPA_CODE_LABEL), load_code(ZEOPP_CODE_LABEL),cif_path)  # pylint: disable=no-value-for-parameter
    #submit_bindingsite(cif_path)
