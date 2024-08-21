#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Submit ZeoppMultistageDdecWorkChain for H2O"""

import os
import click

from aiida.engine import run
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.orm import Dict, Str

from aiida.orm import load_code
from aiida import load_profile
load_profile()
cif_path = "/home/yutao/project/toluene_adsobent/MIP-211/Al-muconate.cif" #"/home/yutao/project/toluene_adsobent/In-MOF/CIXDUJ.cif"

ZEOPP_CODE_LABEL = 'zeopp@lsmosrv6'
CP2K_CODE_LABEL = 'cp2k-new@jed'
DDEC_CODE_LABEL = 'ddec2017@jed'
PROTOCOL_FILE = os.path.abspath('/home/yutao/project/aiida/ff_optim/multistage_cualpyc_supercell_test.yaml')
# Workchain objects
ZeoppMultistageDdecWorkChain = WorkflowFactory('lsmo.zeopp_multistage_ddec')  # pylint: disable=invalid-name

#Data objects
CifData = DataFactory('core.cif')  # pylint: disable=invalid-name
NetworkParameters = DataFactory('zeopp.parameters')  # pylint: disable=invalid-name



#@click.argument('ddec_atdens_path')
def main(zeopp_code, cp2k_code, ddec_code, cif_path):
    """Example usage:
    ATDENS_PATH='/home/daniele/Programs/aiida-database/data/chargemol_09_26_2017/atomic_densities/'
    verdi run run_ZeoppMultistageDdecWorkChain_H2O.py zeopp@localhost cp2k@localhost ddec@localhost $ATDENS_PATH
    """

    cp2k_options = {
        'resources': {
            'num_machines': 1,
            'num_mpiprocs_per_machine': 72
        },
        'max_wallclock_seconds': 72 * 60 * 60,
        'withmpi': True,
    }

    ddec_options = {
        'resources': {
            'num_machines': 1
        },
        'max_wallclock_seconds': 24 *60 * 60,
        'withmpi': False,
    }

    zeopp_options = {
        'resources': {
            'num_machines': 1
        },
        'max_wallclock_seconds': 10 * 60,
        'withmpi': False,
    }

    ddec_params = Dict(
        {
            'net charge': 0.0,
            'charge type': 'DDEC6',
            'periodicity along A, B, and C vectors': [True, True, True],
            'compute BOs': False,
            'atomic densities directory complete path': '/home/yutli/software/chargemol_09_26_2017/atomic_densities/',
            'input filename': 'valence_density',
        })

    zeopp_params = NetworkParameters(
        dict={
            'ha': 'DEF',  # Using high accuracy (mandatory!)
            'res': True,  # Max included, free and incl in free sphere
            'sa': [1.86, 1.86, 1000],  # Nitrogen probe to compute surface
            'vol': [0.0, 0.0, 1000],  # Geometric pore volume
        })

    structure = CifData(file=cif_path).store()
    structure.label = 'MOF_structure'

    inputs = {
        'structure': structure,
        'protocol_tag': Str('standard'),
        'metadata': {
            'label': 'test',
        },
        'cp2k_base': {
            'cp2k': {
                'code': cp2k_code,
                'metadata': {
                    'options': cp2k_options,
                }
            }
        },
        'ddec': {
            'parameters': ddec_params,
            'code': ddec_code,
            'metadata': {
                'options': ddec_options,
            }
        },
        'zeopp': {
            'parameters': zeopp_params,
            'code': zeopp_code,
            'metadata': {
                'options': zeopp_options,
            }
        }
    }

    run(ZeoppMultistageDdecWorkChain, **inputs)


if __name__ == '__main__':
    print(f"Start optimize {cif_path}")
    main(load_code(ZEOPP_CODE_LABEL),load_code(CP2K_CODE_LABEL), load_code(DDEC_CODE_LABEL), cif_path)  # pylint: disable=no-value-for-parameter
