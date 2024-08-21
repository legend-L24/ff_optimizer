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
import config

cif_path = config.cif_path
ff_data = config.ff_data
copy_from_remote = config.copy_from_remote
dest_path = config.dest_path
exp_path = config.exp_path
Number_of_points = config.Number_of_points
pressure_list = config.pressure_list
Transfer_unit = config.Transfer_unit
Temperature = config.Temperature

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / 'data'
load_profile()
# Workchain objects
IsothermWorkChain = WorkflowFactory('lsmo.isotherm')
# Workchain objects
SimAnnealingWorkChain = WorkflowFactory('lsmo.sim_annealing')  # pylint: disable=invalid-name


# Data objects
CifData = DataFactory('core.cif')
NetworkParameters = DataFactory('zeopp.parameters')
RASPA_CODE_LABEL = 'raspa@lsmosrv5'
ZEOPP_CODE_LABEL = 'zeopp@lsmosrv5'

def run_isotherm(raspa_code, zeopp_code, cifdata, pressure_list):  # pylint: disable=redefined-outer-name
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
            "ff_cutoff": 12.8,  # (float) CutOff truncation for the VdW interactions (Angstrom).
            'temperature': Temperature,  # (K) Note: higher temperature will have less adsorbate and it is faster
            'zeopp_volpo_samples': 10,  # Default: 1e5 *NOTE: default is good for standard real-case!
            'zeopp_block_samples': 1,  # Default: 100
            "zeopp_probe_scaling": 0,
            'raspa_widom_cycles': 100,  # Default: 1e5
            'raspa_gcmc_init_cycles': 1000,  # Default: 1e3
            'raspa_gcmc_prod_cycles': 6000,  # Default: 1e4
            "raspa_verbosity": 10,  # (int) Print stats every: number of cycles / raspa_verbosity.
            'pressure_list': pressure_list#,0.059015421115065,0.077698695136418,0.09638196915777]#[0.021648873072361, 0.077698695136418, 0.20017793594306]#[0.021648873072361, 0.077698695136418, 0.20017793594306, 0.34756820877817, 0.49495848161329, 0.80219454329775, 1.1986951364176]# [0.02164887307236063, 0.03825622775800713, 0.05901542111506525, 0.07769869513641753, 0.09638196915776986, 0.1482799525504152, 0.20017793594306055, 0.24584816132858844, 0.29774614472123384, 0.3475682087781732, 0.39739027283511275, 0.44928825622775814, 0.49495848161328604, 0.6008303677342824, 0.6983985765124557, 0.8021945432977464, 0.9018386714116253, 0.9973309608540928, 1.1986951364175567, 1.3959074733096088, 1.5993475682087785, 1.7986358244365368, 2.0020759193357063],
            #'pressure_list': [0.00705972, 0.02005115, 0.07155012, 0.13724632, 0.23730758],
            #'pressure_min': 0.001,  # Default: 0.001 (bar)
            #'pressure_max': 3,  # Default: 10 (bar)
        })

    results, node = engine.run_get_node(builder)

    assert node.is_finished_ok, results
    print("This is the final pk values: ", node.pk)
    return results, node
    


if __name__ == '__main__':
    results, node = run_isotherm(load_code(RASPA_CODE_LABEL), load_code(ZEOPP_CODE_LABEL),cif_path,pressure_list)  # pylint: disable=no-value-for-parameter
    pk = node.pk
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
            raise ValueError("Wrong number of files")
        for pdbfile in movie_pdb_files:
            if "component_CO2_0" in pdbfile:
                file = pdbfile
                break
        remote_data.getfile(os.path.join(copy_from_remote, file), os.path.join(dest_path, file))
    
   
    import numpy as np
    arr_3 = np.loadtxt(exp_path, delimiter=',')
    picked_ls = list(range(Number_of_points))#,2,3,4]#[0]#[0, 3, 6, 9, 12, 15, 18]

    picked_isotherm = [arr_3[i,1]*Transfer_unit/22.4 for i in picked_ls] # because the experimental data is STP
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
