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
from utils import scaling_gas, extract_from_raspa, write_scaling_gas
from jax import clear_caches, clear_backends


"""

Superparameters for Lenard-Jone Potential optimization, some parameters need to read aiida workflow and set them

"""

'''

Here is the advanced version to do it. I will add those element I want to refine

def write_config(structure_folder, experiment_path, cif_path, dest_path, copy_to_path, ff_path, Transfer_unit, SET_temperature, path = "/home/yutao/project/aiida/applications/config.py"):
    with open(path, 'w') as f:
        f.write(f"structure_folder = '{structure_folder}'\n")
        f.write(f"experiment_path = '{experiment_path}'\n")
        f.write(f"cif_path = '{cif_path}'\n")
        f.write(f"dest_path = '{dest_path}'\n")
        f.write(f"copy_to_path = '{copy_to_path}'\n")
        f.write(f"ff_path = '{ff_path}'\n")
        f.write(f"Transfer_unit = {Transfer_unit}\n")
        f.write(f"SET_temperature = {SET_temperature}\n")

# Call the function to create the config.py file
write_config(structure_folder, experiment_path, cif_path, dest_path, copy_to_path, ff_path, Transfer_unit, SET_temperature)

# Now import the config module
import config

# Now you can use the variables from config.py
structure_folder = config.structure_folder
experiment_path = config.experiment_path
cif_path = config.cif_path
dest_path = config.dest_path
copy_to_path = config.copy_to_path
ff_path = config.ff_path
Transfer_unit = config.Transfer_unit
SET_temperature = config.SET_temperature

'''

'''
Number_points = 5           ## must be smaller than len(picked_ls)
Trajectory_length = 250#250          #液体pdb文件的个数
loop_time =   100                  #迭代循环次数    推荐50-100
scaling_factors = (2,2,2)          # This is read from aiida workflow 2,2,2 for NOTT-300
cutoff = 0.905     #This value need to check. Because Openmm a little weired to compute the cutoff, for aiida, the cutoff is 12.0
experiment_path = "/home/yutao/project/Al-MOF/nott300/Default_Dataset.csv"
Transfer_unit = 8.0284625000/9.6917060506 #It also depends on different structure
SET_temperature = 273
Framework_path = "/home/yutao/project/Al-MOF/nott300/structure.pdb"
Forcefiled_path = "/home/yutao/project/Al-MOF/nott300/forcefield.xml"
#In the whole workflow, new files will be written to the dest_path, and the original files will be copied to the copy_to_path
aiida_path = "/home/yutao/project/aiida/applications/"
Scaled_frame_path = "/home/yutao/project/Al-MOF/nott300/scaled_frame.pdb"
dest_path = "/home/yutao/project/MIL-120/traj2/"
copy_to_path = "./traj2/"
ff_path = '/home/yutao/project/aiida/applications/ff_3.json'
cif_path = "/home/yutao/project/Al-MOF/nott300/RSM0516.cif"
for direct in [dest_path, copy_to_path]:
    if not os.path.exists(direct):
        os.makedirs(direct)
        print("Create directory: ", direct)
'''

'''
structure_folder = "/home/yutao/project/Al-MOF/MIL-160/"
experiment_path = os.path.join(structure_folder, "298K_short.csv")
cif_path = os.path.join(structure_folder, "MIL-160-Al.cif")
dest_path = "/home/yutao/project/MIL-120/traj2/"
copy_to_path = "./traj2/"
ff_path = '/home/yutao/project/aiida/applications/ff_2.json'
Transfer_unit = 15.9036500000/5.0184135189 #It also depends on different structure, it also contains transfer from STP to mol/Kg
SET_temperature = 298
'''

structure_folder = "/home/yutao/project/local/Al-MOF/MIL-160-new/"
experiment_path = os.path.join(structure_folder, "298K_short.csv")
cif_path = os.path.join(structure_folder, "MIL-160-Al.cif")
dest_path = "/home/yutao/project/MIL-120/traj1/"
copy_to_path = "./traj1/"
ff_path = '/home/yutao/project/aiida/applications/ff_1.json'
Transfer_unit = 15.9036500000/5.0184135189 #It also depends on different structure, it also contains transfer from STP to mol/Kg
SET_temperature = 298

"""

Superparameters for Lenard-Jone Potential optimization, some parameters need to read aiida workflow and set them

"""
# remember to change UFF_local.json and init_local.xml, extra_atoms are after the framework atoms
extra_atoms = ['Os', 'Ho']
suplement_dict = {
"NonbonedForce": [{"epsilon": "0.25079", "sigma": "0.31181", "type": "Oal"},
                  {"epsilon": "0.18436", "sigma": "0.25711", "type": "Ho"}],
"AtomTypes": [{"name": "Os", "class": "Os", "element": "Os", "mass": "0"},
              {"name": "Ho", "class": "Ho", "element": "Ho", "mass": "0"},],
}


Number_points = 3          ## must be smaller than len(picked_ls)
Trajectory_length = 250#250          #液体pdb文件的个数
loop_time =   100                  #迭代循环次数    推荐50-100


cutoff = 0.905     #This value need to check. Because Openmm a little weired to compute the cutoff, for aiida, the cutoff is 12.0


Framework_path = os.path.join(structure_folder,"structure.pdb")
Forcefiled_path = os.path.join(structure_folder,"forcefield.xml")
#In the whole workflow, new files will be written to the dest_path, and the original files will be copied to the copy_to_path
aiida_path = "/home/yutao/project/aiida/applications/"
Scaled_frame_path = os.path.join(structure_folder,"scaled_frame.pdb")

for direct in [dest_path, copy_to_path]:
    if not os.path.exists(direct):
        os.makedirs(direct)
        print("Create directory: ", direct)



'''

The format of experimental data: two columns which can be read by np.loadtxt without skiprows

'''

data = np.loadtxt(experiment_path, delimiter=',')
picked_ls = list(range(Number_points))#[0,1,2,3,4,5,6,7,8,9] #[0, 2, 4, 6, 8, 10, 14, 18, 22]#[0, 3, 6, 9, 12, 15, 18]
picked_pressure = [data[i,0] for i in picked_ls]
picked_isotherm = [data[i,1]*Transfer_unit/22.4 for i in picked_ls]

bar = 10**5

def is_close_to_list(value, value_list):
    for list_value in value_list:
        relative_error = abs((value - list_value) / list_value)
        if relative_error < 0.01:
            return 1
    return 0

def move_traj(dest_path ,picked_pressure, copy_to_path):
    global bar
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


def compute_binding_energy(paramset,topo, pos, lj_gen, numframe,cutoff):
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

def detect_parameter_change(paramset_new, paramset_old, error_threshold):
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
        paramset.mask['LennardJonesForce']['epsilon'] = paramset.mask['LennardJonesForce']['epsilon'].at[idx].set(0)
    return paramset


import json
Transfer_energy_unit = 254.152/2.11525
Transfer_length_unit = 10
def update_ff(paramset, dest_path):
    global Transfer_energy_unit, Transfer_length_unit, extra_atoms
    element_list = ['Al_', 'C_', 'H_', 'O_']
    extra_atoms_new = [atom_name+"_" for atom_name in extra_atoms]
    if len(extra_atoms) !=0:
        element_list += extra_atoms_new
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
def analyse_traj(paramset, lj_gen, dest_path, numframe, cutoff,  interval):

    global Framework_path, Scaled_frame_path, Number_points, picked_pressure, picked_isotherm, scaling_factors, SET_temperature

    traj_dict = {}

    traj_ls = os.listdir(dest_path)
    create_supercell(Framework_path, scaling_factors, Scaled_frame_path)

    # Filter out file names and keep only directory names
    dir_names = [name for name in traj_ls if os.path.isdir(os.path.join(dest_path, name)) and name.isdigit()]
    dir_names = sorted(map(int, dir_names))
    dir_names = [str(i) for i in dir_names]
    for directory in dir_names[:Number_points]:
        idx = int(directory)
        traj_dict[idx] = {'experiment': {'pressure': picked_pressure[idx-1], 'loading': picked_isotherm[idx-1]}, 'structure': [], 'refer_energy':[], 'loading':[]}
        gas_dir = os.path.join(dest_path, directory)
        for gas_path in os.listdir(gas_dir)[::interval]:
            topo, pos, num = simple_merge(Scaled_frame_path,os.path.join(gas_dir,gas_path))
            ener_lj = compute_binding_energy(paramset,topo, pos, lj_gen, numframe,cutoff)
            traj_dict[idx]['structure'].append([topo, pos])
            traj_dict[idx]['loading'].append(num/scaling_factors[0]/scaling_factors[1]/scaling_factors[2]/3)
            traj_dict[idx]['refer_energy'].append(ener_lj)

    for key in traj_dict.keys():
        traj_dict[key]['refer_energy'] = jnp.array(traj_dict[key]['refer_energy'])
        traj_dict[key]['loading'] = jnp.array(traj_dict[key]['loading'])
        traj_dict[key]['estimator'] = ReweightEstimator(ref_energies=traj_dict[key]['refer_energy'], temperature=SET_temperature)
    return traj_dict

import subprocess

def generate_config(cif_path, picked_pressure, Transfer_unit, save_path,ff_path=ff_path, copy_from_remote="Movies/System_0/", dest_path=dest_path, exp_path=experiment_path, Number_points=Number_points, Temperature = SET_temperature,path = "/home/yutao/project/aiida/applications/config.py"):
    with open(os.path.join(save_path, 'config.py'), 'w') as f:
        f.write(f"ff_data = '{ff_path}'\n")
        f.write(f"copy_from_remote = '{copy_from_remote}'\n")
        f.write(f"dest_path = '{dest_path}'\n")
        f.write(f"exp_path = '{exp_path}'\n")
        f.write(f"Number_of_points = {Number_points}\n")
        f.write(f"cif_path = '{cif_path}'\n")
        f.write(f"pressure_list = {picked_pressure}\n")
        f.write(f"Transfer_unit = {Transfer_unit}\n")
        f.write(f"Temperature = {Temperature}\n")

def sample(cif_path, picked_pressure):
    global aiida_path, Transfer_unit
    generate_config(cif_path, picked_pressure, Transfer_unit, aiida_path) 
    command = [os.path.join(aiida_path, "sample_workflow.sh")]
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



"""

Write the necessary files

"""

from utils import write_force_field, write_pdb_file, rename_atoms, read_cif_file, transform_cif_info
from ase.io import read
from openmm import app
# co2 form TraPPE File, O17, C18 are just inherited from the first example: MIL-120 
co2_info = [{"name": "O17", "type": "O_co2", "charge": "-0.35"},
            {"name": "C18", "type": "C_co2", "charge": "0.70"}]



atoms = read(cif_path)
atomic_number = len(atoms)*scaling_factors[0]*scaling_factors[1]*scaling_factors[2]

cell_parameters = atoms.get_cell_lengths_and_angles() # Get the cell parameters
carterisian_pos = atoms.get_positions()
cif_info = read_cif_file(cif_path)
transformed_info = transform_cif_info(cif_info)
pos_info = rename_atoms(cif_info, carterisian_pos)

write_force_field(transformed_info, co2_info, Forcefiled_path,suplement_dict)
write_pdb_file(pos_info,cell_parameters, Framework_path)


# Initial Optimized parameters
xmlio = XMLIO()
xmlio.loadXML("data/init_OH.xml")
ffinfo = xmlio.parseXML()
paramset_old = ParamSet()
lj_gen = LennardJonesGenerator(ffinfo, paramset_old)

xmlio = XMLIO()
#xmlio.loadXML("data/init.xml")
xmlio.loadXML("data/init_OH.xml")
#xmlio.loadXML("0219.xml")
ffinfo = xmlio.parseXML()
paramset = ParamSet()
lj_gen = LennardJonesGenerator(ffinfo, paramset)


paramset.mask['LennardJonesForce']['sigma'] = paramset.mask['LennardJonesForce']['sigma'].at[0].set(0)
paramset.mask['LennardJonesForce']['sigma'] = paramset.mask['LennardJonesForce']['sigma'].at[1].set(0)
paramset.mask['LennardJonesForce']['sigma'] = paramset.mask['LennardJonesForce']['sigma'].at[2].set(0)
paramset.mask['LennardJonesForce']['sigma'] = paramset.mask['LennardJonesForce']['sigma'].at[3].set(0)
paramset.mask['LennardJonesForce']['sigma'] = paramset.mask['LennardJonesForce']['sigma'].at[4].set(0)
paramset.mask['LennardJonesForce']['sigma'] = paramset.mask['LennardJonesForce']['sigma'].at[5].set(0)

paramset.mask['LennardJonesForce']['epsilon'] = paramset.mask['LennardJonesForce']['epsilon'].at[1].set(0)
paramset.mask['LennardJonesForce']['epsilon'] = paramset.mask['LennardJonesForce']['epsilon'].at[2].set(0)
paramset.mask['LennardJonesForce']['epsilon'] = paramset.mask['LennardJonesForce']['epsilon'].at[3].set(0)

optimizer = optax.adam(0.02)
opt_state = optimizer.init(paramset)


os.system(f"cp /home/yutao/project/aiida/applications/UFF_OH.json {ff_path}")
for nloop in range(100):
    print(f"{nloop} optimization started")
    sample(cif_path, picked_pressure)
    move_traj(dest_path,picked_pressure, copy_to_path)
    traj_dict = analyse_traj(paramset=paramset, lj_gen=lj_gen, dest_path=copy_to_path, numframe=atomic_number, cutoff=cutoff, interval=6)
    
    for i in range(1,Number_points+1):
    #print(np.average(traj_dict[i]['experiment']['loading']))
    #print(np.average(traj_dict[i]['loading']))
        print(f"Range of energy: {min(traj_dict[i]['refer_energy'])} -- {max(traj_dict[i]['refer_energy'])}")

    def loss(paramset):
        errors = []
        for idx in range(1, Number_points+1):
            energies = []
            for jdx in range(len(traj_dict[idx]['structure'])):  
                ener = compute_binding_energy(paramset, traj_dict[idx]['structure'][jdx][0], traj_dict[idx]['structure'][jdx][1], lj_gen, numframe=atomic_number,cutoff=cutoff)
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
    #g.parameters['LennardJonesForce']['epsilon'] = g.parameters['LennardJonesForce']['epsilon']*scalar_epsilon
    #print("This is scaled derivative",g.parameters['LennardJonesForce']['epsilon'])
    updates, opt_state = optimizer.update(g, opt_state)
    updates.parameters = update_mask(updates.parameters,paramset.mask)
    paramset = optax.apply_updates(paramset, updates)
    paramset = jax.tree_map(lambda x: jnp.clip(x, 0.0, 1e8), paramset)
    update_ff(paramset, ff_path)
    lj_gen.overwrite(paramset)
    sigma_indices, epsilon_indices = detect_parameter_change(paramset, paramset_old,0.9)
    paramset = fix_changed_parameters(paramset, sigma_indices, epsilon_indices)
    print(f"This is {nloop}th time", f" Loss: {v} and Parameters: ")
    print(paramset.parameters['LennardJonesForce']['sigma'])
    print(paramset.parameters['LennardJonesForce']['epsilon'])
    clear_caches()
    clear_backends()  
