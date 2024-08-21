# this is some self-defined functions for testing the model

# These package is inherited from Lenard-Jones optimization part of DMFF

import openmm.app as app
import openmm as mm
import openmm.unit as unit
import numpy as np
import jax
import jax.numpy as jnp
import dmff
from dmff.api.xmlio import XMLIO
from dmff.api.paramset import ParamSet
from dmff.generators.classical import CoulombGenerator, LennardJonesGenerator
from dmff.api.hamiltonian import Hamiltonian
from dmff.operators import ParmedLennardJonesOperator
from dmff import NeighborListFreud
from dmff.mbar import ReweightEstimator
import mdtraj as md
from tqdm import tqdm, trange
import parmed
import sys
import os
from dmff.api.topology import DMFFTopology
# this is a package I write to solve some IO problems utils.py
from utils import create_supercell, gas_generate,add_loading, simple_merge
from utils import cutoff_topology
import matplotlib.pyplot as plt
import optax
from utils import extract_from_raspa
from IPython.display import display
"""

Superparameters for Lenard-Jone Potential optimization

"""
Number_points = 3           ## must be smaller than len(picked_ls)
Trajectory_length = 250#250          #液体pdb文件的个数
target_site1 = -50.60                  #拟合的目标binding energy
target_site2 = -46.69           #拟合的目标binding energy
SET_temperature=  303           #温度设定
time_gap=   2.2                      #分子动力学模拟过程中每一个frame的时间间隔，单位是皮秒picosecond   推荐2-4ps
loop_time =   100                  #迭代循环次数    推荐50-100
scaling_factors = (3,3,2)
cutoff = 0.905     #1.3 # unit is nanometer

Transfer_unit = 2.7719416667/5.6100437023 

pressure_list = [
            0.021648873072361,
            0.038256227758007,
            0.059015421115065,
            0.077698695136418,
            0.09638196915777,
            0.14827995255042,
            0.20017793594306,
            0.24584816132859,
            0.29774614472123,
            0.34756820877817,
            0.39739027283511,
            0.44928825622776,
            0.49495848161329,
            0.60083036773428,
            0.69839857651246,
            0.80219454329775,
            0.90183867141163,
            0.99733096085409,
            1.1986951364176,
            1.3959074733096,
            1.5993475682088,
            1.7986358244365,
            2.0020759193357
        ]

arr_3 = np.loadtxt("/home/yutao/dataset/exp_303.txt", delimiter=',')

picked_ls = list(range(Number_points))#[0,1,2,3,4,5,6,7,8,9] #[0, 2, 4, 6, 8, 10, 14, 18, 22]#[0, 3, 6, 9, 12, 15, 18]
picked_pressure = [pressure_list[i] for i in picked_ls]
#print(picked_pressure)
#picked_isotherm = [arr_3[i,1]*Transfer_unit*2/1.5 for i in picked_ls]
picked_isotherm = [arr_3[i,1]*Transfer_unit for i in picked_ls]
def is_close_to_list(value, value_list):
    for list_value in value_list:
        relative_error = abs((value - list_value) / list_value)
        if relative_error < 0.01:
            return 1
    return 0

import os
import numpy as np
from utils import scaling_gas, extract_from_raspa, write_scaling_gas
bar = 10**5
def move_traj(dest_path = "/home/yutao/project/MIL-120/traj1/",picked_pressure=picked_pressure, copy_to_path = "./traj1/"):
    traj_ls = os.listdir(dest_path)
    isotherm_data = [[],[]] # the first list is for pressure, the second is for loading
    jdx = 0 
    for traj in extract_from_raspa(traj_ls):
        pdb_file = traj[1]
        if not pdb_file.endswith(".pdb") or 'Movie_framework' not in pdb_file:
            continue
        if not is_close_to_list(float(traj[0])/bar, picked_pressure):
            continue
        isotherm_data[0].append(float(traj[0])/bar)
        pdb_path = os.path.join(dest_path, pdb_file)
        with open(pdb_path) as f:
            lines = f.readlines()
        num_atoms_list = []  # List to store the number of atoms in each structure
        index = 0
        write_idx = 1
        num_atoms = 0  # Variable to store the number of atoms in the current structure
        directory = copy_to_path+f"{jdx+1}"
        jdx += 1
        if not os.path.exists(directory):
            os.makedirs(directory)
            print("Directory created:", directory)
        for line in lines:
            if line.startswith("MODEL"):
                if index>=150:
                    write_scaling_gas(block_coords, "data/gas.pdb", write_idx, dest_path=directory)
                    write_idx += 1
                block_coords = []
                block_Csym = []
                index += 1
                num_atoms_list.append(num_atoms)  # Add the number of atoms to the list
                num_atoms = 0  # Reset the number of atoms for the next structure
            if line.startswith("ATOM"):
                parts = line.split()
                coords = np.array([float(parts[4]), float(parts[5]), float(parts[6])])
                block_coords.append(coords)
                block_Csym.append(parts[-1])
                num_atoms += 1  # Increment the number of atoms
        #num_atoms_list.append(num_atoms)  # Add the number of atoms for the last structure
        isotherm_data[1].append(sum(num_atoms_list)/len(num_atoms_list)/3/3/2/3)
        #print("Number of atoms in each structure for", pdb_file, ":", num_atoms_list)
    return isotherm_data

def update_mask(parameters, mask):
    updated_parameters = parameters.copy()
    
    for force_type, force_params in mask.items():
        if force_type in parameters:
            for param, mask_array in force_params.items():
                if param in parameters[force_type]:
                    # Update values based on the mask
                    updated_parameters[force_type][param] = jnp.where(mask_array == 1, 
                                                                      parameters[force_type][param], 
                                                                      0)
    return updated_parameters

def compute_binding_energy(paramset,topo, pos, lj_gen, numframe=720,cutoff=cutoff):
    topodata = dmff.DMFFTopology(topo)
    # Because dmfftopology does not provide a good entry for open.topology object generated by pdb file, I had to suplement something
    for atom in topodata.atoms():
        if atom.residue.name=="MOL":
            atom.meta['type']=atom.meta['element']
            atom.meta['class']=atom.meta['element']
        elif atom.residue.name=="GAS":
            #print(atom.meta)
            atom.meta['type']=atom.meta['element']+"_co2"
            atom.meta['class']=atom.meta['element']+"_co2"
        #print(atom.meta['element'])
    cov_mat = topodata.buildCovMat()
    lj_force = lj_gen.createPotential(
    topodata, nonbondedMethod=app.PME, nonbondedCutoff=cutoff, args={})
    pos_jnp = jnp.array(pos.value_in_unit(unit.nanometer))
    cell_jnp = jnp.array(topo.getPeriodicBoxVectors().value_in_unit(unit.nanometer))
    cov_mat=cov_mat.at[:numframe,:numframe].set(1)
    nblist = NeighborListFreud(topo.getPeriodicBoxVectors().value_in_unit(unit.nanometer), cutoff, cov_mat)
    nblist.allocate(pos_jnp, cell_jnp)
    pairs = jnp.array(nblist.pairs)
    ener = lj_force(pos_jnp,cell_jnp, pairs, paramset)
    return ener

def detect_parameter_change(paramset_new, paramset_old, error_threshold=0.4):
    # Get the initial parameters
    initial_sigma = paramset_old.parameters['LennardJonesForce']['sigma']
    initial_epsilon = paramset_old.parameters['LennardJonesForce']['epsilon']
    
    # Get the updated parameters
    updated_sigma = paramset_new.parameters['LennardJonesForce']['sigma']
    updated_epsilon = paramset_new.parameters['LennardJonesForce']['epsilon']
    
    # Calculate the percentage change for each parameter
    sigma_change = np.abs(updated_sigma - initial_sigma) / initial_sigma
    epsilon_change = np.abs(updated_epsilon - initial_epsilon) / initial_epsilon

    # Find the indices of values that have changed by more than 40%
    sigma_indices = np.where(sigma_change > error_threshold)[0]
    epsilon_indices = np.where(epsilon_change > error_threshold)[0]
    
    return sigma_indices, epsilon_indices

def fix_changed_parameters(paramset, sigma_indices, epsilon_indices):
    for idx in sigma_indices:
        paramset.mask['LennardJonesForce']['sigma'] = paramset.mask['LennardJonesForce']['sigma'].at[idx].set(0)
    for idx in epsilon_indices:
        if idx==0:continue
        paramset.mask['LennardJonesForce']['epsilon'] = paramset.mask['LennardJonesForce']['epsilon'].at[idx].set(0)
    return paramset


import json
Transfer_energy_unit = 254.152/2.11525
Transfer_length_unit = 10
def update_ff(paramset, dest_path='/home/yutao/project/aiida/applications/ff_2.json'):
    global Transfer_energy_unit, Transfer_length_unit
    element_list = ['Al_', 'C_', 'H_', 'O_']
    params = paramset.parameters
    ff_data = {}
    if len(element_list) != params['LennardJonesForce']['sigma'].shape[0]-2:
        raise ValueError("Length of element list and parameter list does not match")
    sigma_list = params['LennardJonesForce']['sigma'].tolist()
    epsilon_list = params['LennardJonesForce']['epsilon'].tolist()
    for idx in range(len(element_list)):
        ff_data[element_list[idx]] = ['lennard-jones', epsilon_list[idx]*Transfer_energy_unit, sigma_list[idx]*Transfer_length_unit]
    with open(dest_path, 'w') as f:
        json.dump(ff_data, f, indent=4)

from jax import clear_backends
def analyse_traj(paramset, lj_gen, dest_path = "./traj1/", interval=3):
    traj_dict = {}
    global Number_points, cutoff
    traj_ls = os.listdir(dest_path)
    create_supercell("data/MIL-120.pdb", scaling_factors, "scaled_frame.pdb")

    # Filter out file names and keep only directory names
    dir_names = [name for name in traj_ls if os.path.isdir(os.path.join(dest_path, name)) and name.isdigit()]
    dir_names = sorted(map(int, dir_names))
    dir_names = [str(i) for i in dir_names]
    for directory in dir_names[:Number_points]:
        idx = int(directory)
        traj_dict[idx] = {'experiment': {'pressure': picked_pressure[idx-1], 'loading': picked_isotherm[idx-1]}, 'structure': [], 'refer_energy':[], 'loading':[]}
        gas_dir = os.path.join(dest_path, directory)
        for gas_path in os.listdir(gas_dir)[::interval]:
            topo, pos, num = simple_merge("scaled_frame.pdb",os.path.join(gas_dir,gas_path))
            ener_lj = compute_binding_energy(paramset,topo, pos, lj_gen, numframe=720,cutoff=cutoff)
            traj_dict[idx]['structure'].append([topo, pos])
            traj_dict[idx]['loading'].append(num/scaling_factors[0]/scaling_factors[1]/scaling_factors[2]/3)
            traj_dict[idx]['refer_energy'].append(ener_lj)
        clear_backends()
    for key in traj_dict.keys():
        traj_dict[key]['refer_energy'] = jnp.array(traj_dict[key]['refer_energy'])
        traj_dict[key]['loading'] = jnp.array(traj_dict[key]['loading'])
        traj_dict[key]['estimator'] = ReweightEstimator(ref_energies=traj_dict[key]['refer_energy'], temperature=SET_temperature)
    return traj_dict

import subprocess
def sample():
    command = ["/home/yutao/project/aiida/applications/sample_long.sh"]
    # Run the script using subprocess
    completed_process = subprocess.run(command, capture_output=True, cwd="/home/yutao/project/aiida/applications",text=True)
    print("As long as it finishes,",completed_process.returncode)
    # Check the return code
    if completed_process.returncode == 0:
        # The script finished successfully
        display("Script finished successfully!")
        # Display the output in the notebook
        display("Script output:")
        display(completed_process.stdout)
        # Continue with your program logic here
    else:
        # The script encountered an error
        display("Script encountered an error:", completed_process.stderr)
        # Handle the error or exit the program

# Initial Optimized parameters
xmlio = XMLIO()
xmlio.loadXML("data/init.xml")
ffinfo = xmlio.parseXML()

paramset_old = ParamSet()
lj_gen = LennardJonesGenerator(ffinfo, paramset_old)

xmlio = XMLIO()
xmlio.loadXML("data/init.xml")
#xmlio.loadXML("0226_long_new.xml")
ffinfo = xmlio.parseXML()
paramset = ParamSet()
lj_gen = LennardJonesGenerator(ffinfo, paramset)
paramset.mask['LennardJonesForce']['sigma'] = paramset.mask['LennardJonesForce']['sigma'].at[0].set(0)
paramset.mask['LennardJonesForce']['sigma'] = paramset.mask['LennardJonesForce']['sigma'].at[1].set(0)
paramset.mask['LennardJonesForce']['sigma'] = paramset.mask['LennardJonesForce']['sigma'].at[2].set(0)
paramset.mask['LennardJonesForce']['sigma'] = paramset.mask['LennardJonesForce']['sigma'].at[3].set(0)
paramset.mask['LennardJonesForce']['epsilon'] = paramset.mask['LennardJonesForce']['epsilon'].at[1].set(0)
optimizer = optax.adam(0.02)
opt_state = optimizer.init(paramset)


scalar_epsilon = paramset_old.parameters['LennardJonesForce']['epsilon']
scalar_epsilon = scalar_epsilon/jnp.max(scalar_epsilon)

from jax import clear_caches, clear_backends
os.system("cp /home/yutao/project/aiida/applications/UFF.json /home/yutao/project/aiida/applications/ff_2.json")
#os.system("cp 0226_long_new.json /home/yutao/project/aiida/applications/ff_2.json")
for nloop in range(100):
    print(f"{nloop} optimization started")
    sample()
    move_traj(dest_path="/home/yutao/project/MIL-120/traj1/",picked_pressure=picked_pressure, copy_to_path = "./traj1/")
    traj_dict = analyse_traj(paramset, lj_gen, dest_path="./traj1/", interval=15)
    #print(len(traj_dict[1]['structure']))
    #print(traj_dict.keys())
    def loss(paramset):
        errors = []
        for idx in range(1, Number_points+1):
            energies = []
            for jdx in range(len(traj_dict[idx]['structure'])):  
                ener = compute_binding_energy(paramset, traj_dict[idx]['structure'][jdx][0], traj_dict[idx]['structure'][jdx][1], lj_gen, numframe=720,cutoff=cutoff)
                energies.append(ener.reshape((1,)))
            energies = jnp.concatenate(energies)
            weight = traj_dict[idx]['estimator'].estimate_weight(energies)
            reweight_loading = traj_dict[idx]['loading'] * weight
            #print(f"This is {jdx}th reweight_loading results from dmff code.",jnp.average(traj_dict[idx]['loading']),jnp.average(reweight_loading))
            error = jnp.abs(jnp.average(reweight_loading)-traj_dict[idx]['experiment']['loading'])
            errors.append(error.reshape((1,)))
            #print(error)
        errors = jnp.concatenate(errors)
        return jnp.sum(errors)

    v_and_g = jax.value_and_grad(loss, 0)
    v, g = v_and_g(paramset)

    print("This is before derivative",g.parameters['LennardJonesForce']['epsilon'])
    g.parameters['LennardJonesForce']['epsilon'] = g.parameters['LennardJonesForce']['epsilon']*scalar_epsilon
    print("This is scaled derivative",g.parameters['LennardJonesForce']['epsilon'])
    updates, opt_state = optimizer.update(g, opt_state)
    updates.parameters = update_mask(updates.parameters,paramset.mask)
    paramset = optax.apply_updates(paramset, updates)
    paramset = jax.tree_map(lambda x: jnp.clip(x, 0.0, 1e8), paramset)
    update_ff(paramset)
    lj_gen.overwrite(paramset)
    sigma_indices, epsilon_indices = detect_parameter_change(paramset, paramset_old,0.9)
    paramset = fix_changed_parameters(paramset, sigma_indices, epsilon_indices)
    print(f"This is {nloop}th time", f" Loss: {v} and Parameters: ",paramset.parameters['LennardJonesForce']['sigma'], paramset.parameters['LennardJonesForce']['epsilon'])
    clear_caches()
    clear_backends()    
   
