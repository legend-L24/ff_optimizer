from openmm import app, unit
import openmm as mm

"""

The first part is in order to make supercell from a PDB file

"""
import numpy as np
from ase.geometry import wrap_positions
import os

"""
Scale the MIL-120 to 3 2 2 in order to ensure enough long cutoff for nonbonding Forces.
"""

from ase.geometry import wrap_positions

import re
def extract_from_raspa(filenames):    
    values = []
    pattern = r"_([\d.]+)_component_CO2_0\.pdb"

    for filename in filenames:
        match = re.search(pattern, filename)
        if match:
            value = match.group(1)
            values.append(float(value))
    return sorted(zip(values,filenames), key=lambda x: x[0])
def write_scaling_gas(block_coords, pdb_template, index, dest_path="./traj"):
    '''
    input:
        block_coords: coordination of CO2 atoms
        pdb: gas.pdb
        index: index of the structure in the trajectory
    '''
    num = int(len(block_coords)/3)
    pdb = app.PDBFile(pdb_template)
    original_topology = pdb.topology

    new_topology = app.Topology()
    for i in range(num):
        for chain in original_topology.chains():
            
            atom_map = {}  # Keep track of the mapping between original atoms and new atoms

            new_chain = new_topology.addChain()
            for residue in chain.residues():
                new_residue = new_topology.addResidue(residue.name, new_chain)
                for atom in residue.atoms():
                    new_atom = new_topology.addAtom(atom.name, atom.element, new_residue)
                    atom_map[atom] = new_atom
            
        for bond in original_topology.bonds():
            atom1, atom2 = bond
            if atom1 in atom_map and atom2 in atom_map:
                new_topology.addBond(atom_map[atom1], atom_map[atom2])
    app.PDBFile.writeFile(new_topology, block_coords, open(os.path.join(dest_path, f"{index}.pdb"), 'w'))

def scaling_gas(block_coords, pdb_template, index, dest_path="./traj"):
    '''
    input:
        block_coords: coordination of CO2 atoms
        pdb: gas.pdb
        index: index of the structure in the trajectory
    '''
    num = int(len(block_coords)/3)
    pdb = app.PDBFile(pdb_template)
    original_topology = pdb.topology

    new_topology = app.Topology()
    for i in range(num):
        for chain in original_topology.chains():
            
            atom_map = {}  # Keep track of the mapping between original atoms and new atoms

            new_chain = new_topology.addChain()
            for residue in chain.residues():
                new_residue = new_topology.addResidue(residue.name, new_chain)
                for atom in residue.atoms():
                    new_atom = new_topology.addAtom(atom.name, atom.element, new_residue)
                    atom_map[atom] = new_atom
            
        for bond in original_topology.bonds():
            atom1, atom2 = bond
            if atom1 in atom_map and atom2 in atom_map:
                new_topology.addBond(atom_map[atom1], atom_map[atom2])
    return new_topology, block_coords
    #app.PDBFile.writeFile(new_topology, block_coords, open(os.path.join(dest_path, f"{index}.pdb"), 'w'))

def gas_generate(path):
    """
    Input:
        Gas pdb file path
    Output:
        gas pdb topology and the new positions (nm) I check by wrap_positions
    """
    gas = app.PDBFile(path)
    direct_vectors = gas.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
    neigh = np.dot(gas.getPositions().value_in_unit(unit.nanometer)[1],np.linalg.inv(direct_vectors))
    new_positions = wrap_positions(gas.getPositions().value_in_unit(unit.nanometer), 
                                   cell=gas.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer),
                                   pbc=[1, 1, 1],center=neigh)
    new_positions = unit.Quantity(value=new_positions, unit=unit.nanometer)
    return gas.topology, new_positions

def create_supercell(pdb_path, scaling_factors,supercell_path):
    """

    The functions generate many atoms with the same name, when openmm read the function again, it will be delete,so I write generate_supercell

    Input:
        pdb_path: pdb file contains the unit cell
        scaling_factors: a tuple or other iterative object (can use scaling_factors[idx])
        supercell_path: write a pdb file including the supercell
    Output:
        None, just write a new function.
    """
    pdbfile = app.PDBFile(pdb_path)

    # Get the original topology and positions from the PDB file
    original_topology = pdbfile.topology
    original_positions = pdbfile.getPositions()

    # Get the original box vectors
    original_box_vectors = original_topology.getPeriodicBoxVectors()

    # Scale the box vectors
    scaled_box_vectors = [
        original_box_vectors[0] * scaling_factors[0],
        original_box_vectors[1] * scaling_factors[1],
        original_box_vectors[2] * scaling_factors[2]
    ]
    # Convert the scaled box vectors to Vec3 objects
    scaled_box_vectors_plain = [vec.value_in_unit(unit.nanometer) for vec in scaled_box_vectors]
    original_box_vectors_plain = [vec.value_in_unit(unit.nanometer) for vec in original_box_vectors]
    original_positions = [vec.value_in_unit(unit.nanometer) for vec in original_positions]
    # Create a new topology with the scaled box vectors and more atoms
    new_topology = app.Topology()
    new_positions = []

    for chain in original_topology.chains():
        new_chain = new_topology.addChain()
        for residue in chain.residues():
            for i in range(scaling_factors[0]):
                for j in range(scaling_factors[1]):
                    for k in range(scaling_factors[2]):
                        new_residue = new_topology.addResidue(residue.name, new_chain)
                        for atom in residue.atoms():
                            new_atom = new_topology.addAtom(atom.name, atom.element, new_residue)
                            vec = original_positions[atom.index] + i * original_box_vectors_plain[0] + \
                                            j * original_box_vectors_plain[1] + k * original_box_vectors_plain[2]
                            new_positions.append(vec*10) # convert nm to angstrom
    # the new_topology default unit is nanometer, so we can input nanometer value
    # the new_topology default unit is nanometer, so we can input nanometer value
    new_topology.setPeriodicBoxVectors(scaled_box_vectors_plain)
    # Here new_positions should be Quantity or vec in unit of angstrom 
    app.PDBFile.writeFile(new_topology, new_positions, open(supercell_path, 'w'))

def add_loading(frame, loading, outputpath):
    """
    
    This function is writen for CO2. It is common to meet problems when you try to transfer

    """
    gas_topo, gas_pos = gas_generate(loading)
    frame = app.PDBFile(frame)
    new_topology = frame.topology
    new_positions = frame.getPositions().value_in_unit(unit=unit.angstrom)
    for chain in gas_topo.chains():
        atom_map = {}  # Keep track of the mapping between original atoms and new atoms
        for residue in chain.residues():
            for chain0 in new_topology.chains():  
                new_residue = new_topology.addResidue(residue.name, chain0)
                continue
            for atom in residue.atoms():
                #print(atom)
                new_atom = new_topology.addAtom(atom.name, atom.element, new_residue)
                atom_map[atom] = new_atom
            continue
        # Copy bonds for the unique residue type
        for bond in gas_topo.bonds():
            atom1, atom2 = bond
            if atom1 in atom_map and atom2 in atom_map:
                #print(atom1,atom_map[atom1])
                #print(atom2,atom_map[atom2])
                new_topology.addBond(atom_map[atom1], atom_map[atom2])  # Add the bond to the subset topology

    for pos in gas_pos.value_in_unit(unit=unit.angstrom):
        new_positions.append(pos)
    app.PDBFile.writeFile(new_topology, new_positions, open(outputpath, 'w'))

def simple_merge(frame, loading):
    """
    
    This function is writen for CO2. It is common to meet problems when you try to transfer

    """
    try:
        gas = app.PDBFile(loading)
    except:
        frame = app.PDBFile(frame)
        new_topology = frame.topology
        new_positions = frame.getPositions()
        return new_topology, new_positions, 0
    gas_topo, gas_pos = gas.topology, gas.positions
    frame = app.PDBFile(frame)
    new_topology = frame.topology
    new_positions = frame.getPositions()+gas_pos
    for chain in gas_topo.chains():
        atom_map = {}  # Keep track of the mapping between original atoms and new atoms
        for residue in chain.residues():
            for chain0 in new_topology.chains():  
                new_residue = new_topology.addResidue(residue.name, chain0)
                continue
            for atom in residue.atoms():
                #print(atom)
                new_atom = new_topology.addAtom(atom.name, atom.element, new_residue)
                atom_map[atom] = new_atom
            continue
        # Copy bonds for the unique residue type
        for bond in gas_topo.bonds():
            atom1, atom2 = bond
            if atom1 in atom_map and atom2 in atom_map:
                #print(atom1,atom_map[atom1])
                #print(atom2,atom_map[atom2])
                new_topology.addBond(atom_map[atom1], atom_map[atom2])  # Add the bond to the subset topology
    '''
    for pos in gas_pos.value_in_unit(unit=unit.angstrom):
        new_positions.append(pos)
    '''
    num = gas_topo.getNumAtoms()
    return new_topology, new_positions, num
    #app.PDBFile.writeFile(new_topology, new_positions, open(outputpath, 'w'))

def cutoff_topology(topo):
    """
    Input: 
        openmm.app.topology.Topology object, simulation.topology or pdb.topology

    Output:
        MOF framework (MOL residue) without bondings
        GAS molecule (GAS residue) without bondings

    This function will cut the topolgy according to different residue names
    """
    subset_topologies = []

    # Iterate over chains and create a new topology for each chain
    # Iterate over chains and create a new topology for each unique residue type within the chain
    for chain in topo.chains():
        unique_residues = set(residue.name for residue in chain.residues())
        
        for unique_residue_name in unique_residues:
            subset_topology = app.Topology()
            
            # Add the chain to the subset topology
            new_chain = subset_topology.addChain(id=chain.id)
            
            # Copy residues and atoms for the unique residue type
            atom_map = {}  # Keep track of the mapping between original atoms and new atoms
            for residue in chain.residues():
                if residue.name == unique_residue_name:
                    new_residue = subset_topology.addResidue(residue.name, new_chain, id=residue.id)
                    for atom in residue.atoms():
                        new_atom = subset_topology.addAtom(atom.name, atom.element, new_residue)
                        atom_map[atom] = new_atom
            
            # Copy bonds for the unique residue type
            for bond in topo.bonds():
                atom1, atom2 = bond
                if atom1 in atom_map and atom2 in atom_map:
                    subset_topology.addBond(atom_map[atom1], atom_map[atom2])  # Add the bond to the subset topology
            subset_topology.setPeriodicBoxVectors(topo.getPeriodicBoxVectors())
            subset_topologies.append(subset_topology)
    subset_topologies = sorted(subset_topologies, key=lambda x:x.getNumAtoms())
    return subset_topologies

'''

This part is in order to read cif file with charge information in order to 
generate reasonable PDB files and Lenard-Jones force field file.

Input parameters:

cif_path: the path of the cif file

'''

from ase.io import read

def read_cif_file(file_path):
    atom_info = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        atom_data_started = False
        for line in lines:
            if '_atom_site_label' in line:
                atom_info.append(line.split())
                atom_data_started = True
                continue
            if atom_data_started:
                if line.strip():
                    atom_info.append(line.split())

    return atom_info

def extract_atom_info(atom_info):
    extracted_info = []
    for i, atom in enumerate(atom_info):
        name = f"Al{i+1}" if atom[1] == 'Al' else f"H{i+1}" if atom[1] == 'H' else f"C{i+1}"
        extracted_info.append({
            "name": name,
            "type": atom[1],
            "charge": atom[5]
        })

    return extracted_info

def rename_atoms(cif_info, carterisian_pos):
    # Find the index where the atom information starts
    start_index = next(i for i, sublist in enumerate(cif_info) if len(sublist) > 1)

    # Iterate over the cif_info list starting from the start_index
    for i, atom_info in enumerate(cif_info[start_index:], start=1):
        # Change the atom label to include the index
        # Change the atom element as original lable
        atom_info[1] = atom_info[0]
        atom_info[0] += str(i)
        atom_info[2] = carterisian_pos[i-1][0]
        atom_info[3] = carterisian_pos[i-1][1]
        atom_info[4] = carterisian_pos[i-1][2]


    return cif_info

def transform_cif_info(cif_info):
    # Find the index where the atom information starts
    start_index = next(i for i, sublist in enumerate(cif_info) if len(sublist) > 1)

    # Initialize an empty list to store the dictionaries
    transformed_info = []

    # Iterate over the cif_info list starting from the start_index
    for i, atom_info in enumerate(cif_info[start_index:], start=1):
        # Create a dictionary for each atom and append it to the list
        atom_name = atom_info[0] + str(i)
        if len(atom_name) >4:
            atom_name = atom_name[0]+atom_name[-3:]
        atom_dict = {
            "name": atom_name,
            "type": atom_info[0],
            "charge": atom_info[-1]
        }
        transformed_info.append(atom_dict)

    return transformed_info
# Function to pretty print XML
def prettify(elem, level=0):
    """Indentation function"""
    indent = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for subelem in elem:
            prettify(subelem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent
import xml.etree.ElementTree as ET

def write_force_field(frame_info, gas_info,path, suplement_info=None):
    '''

    This function is only suitable for Al, C, H, O system, default value is suitable for co2 adsorption 
    HarmonicAngleForce is only suitable for CO2, if necessary, please modify this part
    atom type data only suitable for Al, C, H, O, if necessary, please add new atoms
    '''
    # Create the root element
    forcefield = ET.Element("ForceField")

    # Add AtomTypes
    atomtypes = ET.SubElement(forcefield, "AtomTypes")
    atom_type_data = [
        {"class": "Al", "element": "Al", "mass": "0.0", "name": "Al"},
        {"class": "C", "element": "C", "mass": "0.0", "name": "C"},
        {"class": "H", "element": "H", "mass": "0.0", "name": "H"},
        {"class": "O", "element": "O", "mass": "0.0", "name": "O"},
        {"class": "C1_co2", "element": "C", "mass": "12.010", "name": "C_co2"},
        {"class": "O1_co2", "element": "O", "mass": "15.999", "name": "O_co2"},
    ]
    if suplement_info:
        atom_type_data += suplement_info["AtomTypes"]
    for atom_type in atom_type_data:
        ET.SubElement(atomtypes, "Type", **atom_type)

    # Add Residues
    residues = ET.SubElement(forcefield, "Residues")
    residue_data = [
        {
            "name": "MOL",
            "atoms": frame_info
        },
        {
            "name": "GAS",
            "atoms": gas_info
        },
    ]
    for residue_info in residue_data:
        residue = ET.SubElement(residues, "Residue", name=residue_info["name"])
        for atom in residue_info["atoms"]:
            ET.SubElement(residue, "Atom", **atom)

    # Add HarmonicBondForce for CO2
    harmonic_bond_force = ET.SubElement(forcefield, "HarmonicBondForce")
    ET.SubElement(
        harmonic_bond_force,
        "Bond",
        class1="C1_co2",
        class2="O1_co2",
        length="0.115999",
        k="943153.3808",
        mask="true"
    )

    # Add HarmonicAngleForce for CO2
    harmonic_angle_force = ET.SubElement(forcefield, "HarmonicAngleForce")
    ET.SubElement(
        harmonic_angle_force,
        "Angle",
        class1="O1_co2",
        class2="C1_co2",
        class3="O1_co2",
        angle="3.141593",
        k="817.5656",
        mask="true"
    )

    # Add NonbondedForce
    nonbonded_force = ET.SubElement(forcefield, "NonbondedForce", coulomb14scale="0.8333333333333334", lj14scale="0.5")
    ET.SubElement(nonbonded_force, "UseAttributeFromResidue", name="charge")
    atom_data = [
        {"epsilon": "2.11525", "sigma": "0.40082", "type": "Al"},
        {"epsilon": "0.43979", "sigma": "0.34309", "type": "C"},
        {"epsilon": "0.18436", "sigma": "0.25711", "type": "H"},
        {"epsilon": "0.25079", "sigma": "0.31181", "type": "O"},
        {"epsilon": "0.65757", "sigma": "0.305", "type": "O_co2"},
        {"epsilon": "0.22469", "sigma": "0.28", "type": "C_co2"},
    ]
    if suplement_info:
        atom_data += suplement_info["NonbonedForce"]
    for atom_info in atom_data:
        ET.SubElement(nonbonded_force, "Atom", **atom_info)
    prettify(forcefield)
    xml_str = ET.tostring(forcefield, encoding="utf-8")
    with open(path, "wb") as f:
        f.write(xml_str)

'''

The Protein Data Bank (PDB) format has specific requirements for the atom name and coordinates in the ATOM records:

Atom Name (columns 13-16): This field should contain the atom name. The atom name should be right-justified within these columns. For example, the carbon alpha atom of alanine would have the atom name " CA " in columns 13-16.

Atom Coordinates (columns 31-54): These columns should contain the x, y, and z coordinates of the atom in Ångströms. The format is as follows:

x-coordinate (columns 31-38)
y-coordinate (columns 39-46)
z-coordinate (columns 47-54)
Each coordinate should be right-justified and fit within its respective columns. The coordinates are typically represented as floating-point numbers with a precision of 3 decimal places.

# the code to check whether the atom names and coordinates are in correct format

frame = app.PDBFile(Framework_path)
atom_names = [atom.name for atom in frame.topology.atoms()]
print(atom_names)

'''

def write_pdb_file(cif_info, cell_parameters, filename):
    with open(filename, 'w') as f:
        f.write("REMARK   For DMFF workflow PDB file\n")
        
        
        # Format and write the crystal parameters
        try:
            a, b, c, alpha, beta, gamma = [float(param) for param in cell_parameters]
        except ValueError:
            raise ValueError("Invalid crystal parameters provided.")
        
        f.write(f"CRYST1 {a:8.3f} {b:8.3f} {c:8.3f} {alpha:6.2f} {beta:6.2f} {gamma:6.2f}\n")
        
        # Find the index where the atom information starts
        start_index = next(i for i, sublist in enumerate(cif_info) if len(sublist) > 1)

        # Get the maximum length of atom names for consistent padding
        #max_atom_name_length = max(len(atom_info[0]) for atom_info in cif_info[start_index:])
        
        # Iterate over the cif_info list starting from the start_index
        for i, atom_info in enumerate(cif_info[start_index:], start=1):
            # PDB only allow 4 characters for atom name
            atom_name = atom_info[0]
            if len(atom_name) >4:
                atom_name = atom_name[0]+atom_name[-3:]
            atom_name = f"{atom_name:<4}"
            serial_number = f"{i:>5}"
            # Write the atom information to the file
            # atom_info[0] is the atom name, atom_info[1] is the element symbol
            f.write(f"ATOM  {serial_number} {atom_name} MOL A   1    {atom_info[2]:8.3f}{atom_info[3]:8.3f}{atom_info[4]:8.3f}  1.00  1.00          {atom_info[1]:>2}\n")
        f.write("END\n")

# co2 form TraPPE File, O17, C18 are just inherited from the first example: MIL-120 
co2_info = [{"name": "O17", "type": "O_co2", "charge": "-0.35"},
            {"name": "C18", "type": "C_co2", "charge": "0.70"}]


import os

if __name__ == '__main__':
    os.chdir("/home/yutao/project/github/DMFF/UFF_opt")
    pdb=app.PDBFile("MIL120_loading.pdb") #the working directory must be the UFF_opt directory
    frame_top, gas_top = cutoff_topology(pdb.topology)
    Example_cif_path = "/home/yutao/project/Al-MOF/nott300/RSM0516.cif"
    atoms = read(Example_cif_path)
    cell_parameters = atoms.get_cell_lengths_and_angles() # Get the cell parameters
    carterisian_pos = atoms.get_positions()
    cif_info = read_cif_file(Example_cif_path)
    transformed_info = transform_cif_info(cif_info)
    pos_info = rename_atoms(cif_info, carterisian_pos)

    write_force_field(transformed_info, co2_info, "/home/yutao/project/Al-MOF/nott300/forcefield.xml")
    write_pdb_file(pos_info,cell_parameters, "/home/yutao/project/Al-MOF/nott300/structure.pdb")
    app.PDBFile("/home/yutao/project/Al-MOF/nott300/structure.pdb")