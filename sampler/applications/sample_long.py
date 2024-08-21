#!/home/yutao/.aiida_venvs/aiida/bin/python
# -*- coding: utf-8 -*-
"""Run example isotherm calculation on MIL-120 to do GCMC sample."""

from pathlib import Path
import os
import json
from aiida import engine
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.orm import Dict, Str, load_code, SinglefileData
from aiida import load_profile
from aiida.orm import QueryBuilder, WorkChainNode, CalcJobNode, RemoteData, load_node
from aiida_raspa.workchains.base import RaspaBaseWorkChain
import os

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / 'data'
load_profile()
# Workchain objects
IsothermWorkChain = WorkflowFactory('lsmo.isotherm')

# Data objects
CifData = DataFactory('core.cif')
NetworkParameters = DataFactory('zeopp.parameters')

RASPA_CODE_LABEL = 'raspa@lsmosrv4'
ZEOPP_CODE_LABEL = 'zeopp@lsmosrv4'
ff_data = '/home/yutao/project/aiida/applications/ff_2.json'
def run_isotherm(raspa_code, zeopp_code, cifdata):  # pylint: disable=redefined-outer-name
    global ff_data
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
            "ff_optim": json.load(open(ff_data)),
            "ff_shifted": True,  # (bool) Shift or truncate the potential at cutoff.
            "ff_cutoff": 12.0,  # (float) CutOff truncation for the VdW interactions (Angstrom).
            'temperature': 303,  # (K) Note: higher temperature will have less adsorbate and it is faster
            'zeopp_volpo_samples': 10,  # Default: 1e5 *NOTE: default is good for standard real-case!
            'zeopp_block_samples': 1,  # Default: 100
            "zeopp_probe_scaling": 0,
            'raspa_widom_cycles': 100,  # Default: 1e5
            'raspa_gcmc_init_cycles': 1000,  # Default: 1e3
            'raspa_gcmc_prod_cycles': 6000,  # Default: 1e4
            "raspa_verbosity": 10,  # (int) Print stats every: number of cycles / raspa_verbosity.
            'pressure_list': [0.021648873072361,0.038256227758007,0.059015421115065]
            #'pressure_list': [0.00705972, 0.02005115, 0.07155012, 0.13724632, 0.23730758],
            #'pressure_min': 0.001,  # Default: 0.001 (bar)
            #'pressure_max': 3,  # Default: 10 (bar)
        })

    results, node = engine.run_get_node(builder)

    assert node.is_finished_ok, results
    print("This is the final pk values: ", node.pk)
    return results, node
    

copy_from_remote = "Movies/System_0/"
dest_path = "/home/yutao/project/MIL-120/traj1/"

if __name__ == '__main__':
    print("Aiida based GCMC sampling started.")
    results, node = run_isotherm(load_code(RASPA_CODE_LABEL), load_code(ZEOPP_CODE_LABEL),"MIL-120.cif")  # pylint: disable=no-value-for-parameter
    pk = node.pk
    print("This is the pk value: ", pk)
    qb = QueryBuilder()
    qb.append(WorkChainNode, filters={'id': pk}, tag="workchain")
    qb.append(RaspaBaseWorkChain, with_incoming="workchain", tag="raspa")
    qb.append(CalcJobNode, with_incoming="raspa", tag="calcjob")
    qb.append(RemoteData, with_incoming="calcjob", tag="remote")

    for remote_data in qb.all()[1:]:
        # Load the RemoteData node
        remote_data = remote_data[0]
        file_ls = remote_data.listdir(copy_from_remote)
        movie_pdb_files = [filename for filename in file_ls if filename.startswith("Movie_") and filename.endswith(".pdb")]
        #print(movie_pdb_files)
        if len(movie_pdb_files) != 3:
            print(movie_pdb_files)
            raise ValueError("Wrong number of files")
        for pdbfile in movie_pdb_files:
            if "component_CO2_0" in pdbfile:
                file = pdbfile
                break
        remote_data.getfile(os.path.join(copy_from_remote, file), os.path.join(dest_path, file))
    
    Transfer_unit = 2.7719416667/5.6100437023
    import numpy as np
    arr_3 = np.loadtxt("/home/yutao/dataset/exp_303.txt", delimiter=',')
    picked_ls =[0,1,2]#[0, 3, 6, 9, 12, 15, 18]

    #print(picked_pressure)
    #picked_isotherm = [arr_3[i,1]*2/1.5 for i in picked_ls]
    picked_isotherm = [arr_3[i,1]*Transfer_unit for i in picked_ls]
    #print(picked_isotherm)
    pk = pk
    qb = QueryBuilder()
    qb.append(WorkChainNode, filters={'id':pk}, tag="workchain")
    qb.append(Dict, with_incoming="workchain")
    outdict_ls = qb.all()[:]
    p_ls = outdict_ls[0][0].get_dict()['isotherm']['pressure']
    loading_ls = outdict_ls[0][0].get_dict()['isotherm']['loading_absolute_average']
    loading_ls = [loading_ls[idx]*Transfer_unit for idx in range(0,len(picked_ls))]
    print(loading_ls)
    print(picked_isotherm)
    loss = np.sum(np.abs(np.array(picked_isotherm)-np.array(loading_ls)))

    print("This is loss function: ", loss)

