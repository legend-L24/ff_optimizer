# -*- coding: utf-8 -*-
"""ff_builder calcfunction."""
import tempfile
import shutil
import os
from math import sqrt
import ruamel.yaml as yaml
from aiida.orm import SinglefileData
from aiida.engine import calcfunction
from .ff_data_schema import FF_DATA_SCHEMA

THISDIR = os.path.dirname(os.path.abspath(__file__))


def check_ff_list(inp_list):
    """Check a list of atom types:
    1) Remove duplicates, preserving the order of the elements.
    2) Warn if there are atom types with the same name but different parameters
    3) If a shorter atom type comes later, swap the order # TODO!
    """
    out_list = []
    for item in inp_list:
        if item.split()[0] not in [x.split()[0] for x in out_list]:  # atom type label is unique
            out_list.append(item)
        else:
            if item in out_list:  # Two atom types are exactly the same
                pass
            else:
                raise ValueError('Two atom types with same name but different parameters are used.')
    return out_list


def load_yaml():
    """Load the ff_data.yaml as a dict.

    Includes validation against schema.
    """
    yamlfullpath = os.path.join(THISDIR, 'ff_data.yaml')

    with open(yamlfullpath, 'r') as stream:
        ff_data = yaml.safe_load(stream)

    FF_DATA_SCHEMA(ff_data)
    return ff_data


def get_ase_charges(cifdata):
    """Given a CifData, get an ASE object with charges."""
    charges_list = []
    for line in cifdata.get_content().split('\n'):  # QUITE FRAGILE, but should always work well with aiida-ddec CIFs
        line_split = line.split()
        if len(line_split) == 6:
            charges_list.append(float(line_split[-1]))
    ase_atoms = cifdata.get_ase()  # It does not get the charges: need to parse and add them.
    ase_atoms.set_initial_charges(charges=charges_list)
    ase_atoms.center(vacuum=0.1)  # move the molecule closer to the origin (it was in the center of the box for CP2K)
    return ase_atoms


def append_cif_molecule(ff_data, mol_cif):
    """Append the FF parameters generated from the CifData, to the ff_loaded from the yaml"""
    mol_ase = get_ase_charges(mol_cif)
    mol_name, ff_name = 'MOL', 'on-the-fly'
    ff_data[mol_name] = {
        'critical_constants': {
            'tc': 999.99,
            'pc': 9999999,
            'af': 9.9,
        },
        ff_name: {
            'description': 'Force field generated on-the-fly with standard LJ parameters, and DFT geometry and charges',
            'atom_types': {},
            'atomic_positions': []
        }
    }

    for atom in mol_ase:
        at_label = f'{atom.symbol}_{atom.index}'
        ff_data[mol_name][ff_name]['atom_types'][at_label] = {
            'force_field':
                'use-framework-ff',
            'pseudo_atom': [
                'yes', atom.symbol, atom.symbol, 0,
                float(atom.mass),
                float(atom.charge), 0.0, 1.0, 1.0, 0, 0, 'relative', 0
            ]
        }
        ff_data[mol_name][ff_name]['atomic_positions'].append([at_label, float(atom.x), float(atom.y), float(atom.z)])
    return ff_data


def string_to_singlefiledata(string, filename):
    """Convert a string to a SinglefileData."""
    tempdir = tempfile.mkdtemp(prefix='aiida_ff-builder_')
    filepath = os.path.join(tempdir, filename)
    with open(filepath, 'w') as fobj:
        fobj.write(string)
    singlefiledata = SinglefileData(file=filepath)
    shutil.rmtree(tempdir)
    return singlefiledata

def render_ff_mixing_def(ff_data, params):
    """Render the force_field_mixing_rules.def file."""
    output = []
    output.append('# general rule for shifted vs truncated (file generated by aiida-raspa)')
    output.append(['truncated', 'shifted'][params['shifted']])
    output.append('# general rule tail corrections')
    output.append(['no', 'yes'][params['tail_corrections']])
    output.append('# number of defined interactions')
    if params['ff_framework']=='Local':
        with open(os.getcwd()+"/data/force_field_mixing_rules.def", "r") as f:
            string = f.read()    
        return string_to_singlefiledata(string, 'force_field_mixing_rules.def'), False
    '''
    
    
    # I change the force filed but I don't want to don't know how to change the 
    
    '''
    
    print("I want to start optimize")
    try:
        if params['ff_optim']:
            ff_data['framework'][params['ff_framework']]['atom_types'].update(params['ff_optim'])
    except:
        refined_ff = {
        "Al_": [
            "lennard-jones",
            254.152,
            4.0082
        ],
        "C_": [
            "lennard-jones",
            34.788010961163,
            3.4309
        ],
        "H_": [
            "lennard-jones",
            22.151264729937,
            2.5711
        ],
        "O_": [
            "lennard-jones",
            4.1285586546958,
            3.1181
        ]}
        ff_data['framework'][params['ff_framework']]['atom_types'].update(refined_ff)
    
    #ff_data['framework'][params['ff_framework']]['atom_types'].update(params['ff_optim'])
    #print(params.attributes)
    force_field_lines = []
    ff_mix_found = False  #If becomes True, needs to handle the mixing differently
    # TODO: this needs to be sorted for python versions where dictionaries are not sorted! #pylint: disable=fixme
    # If separate_interactions==True, prints only "none" interactions for the molecules
    #print("This is changed ff parameters",params['optim_ff'])
    for atom_type, ff_pot in ff_data['framework'][params['ff_framework']]['atom_types'].items():
        force_field_lines.append(' '.join([str(x) for x in [atom_type] + ff_pot]))
    for molecule, ff_name in params['ff_molecules'].items():
        for atom_type, val in ff_data[molecule][ff_name]['atom_types'].items():
            if 'force_field_mix' in val:
                ff_mix_found = True
                ff_pot = val['force_field_mix']
            elif val['force_field'] == 'use-framework-ff':
                continue
            else:
                ff_pot = val['force_field']
            # In case of "separate_interactions" write the ff only if none particle
            if not params['separate_interactions'] or ff_pot[0].lower() == 'none':
                force_field_lines.append(' '.join([str(x) for x in [atom_type] + ff_pot]))

    force_field_lines = check_ff_list(force_field_lines)
    output.append(len(force_field_lines))
    output.append('# atom_type, interaction, parameters')
    output.extend(force_field_lines)
    output.append('# general mixing rule for Lennard-Jones')
    output.append(params['mixing_rule'])
    string = '\n'.join([str(x) for x in output]) + '\n'
    return string_to_singlefiledata(string, 'force_field_mixing_rules.def'), ff_mix_found


def mix_molecule_ff(ff_list, mixing_rule):
    """Mix molecule-molecule interactions in case of separate_interactions: return mixed ff_list"""
    ff_mix = []
    for i, ffi in enumerate(ff_list):
        for ffj in ff_list[i:]:
            if ffi[1].lower() == ffj[1].lower() == 'lennard-jones':
                eps_mix = sqrt(ffi[2] * ffj[2])
                if mixing_rule == 'lorentz-berthelot':
                    sig_mix = 0.5 * (ffi[3] + ffj[3])
                elif mixing_rule == 'jorgensen':
                    sig_mix = sqrt(ffi[3] * ffj[3])
                ff_mix.append('{} {} lennard-jones {:.5f} {:.5f}'.format(ffi[0], ffj[0], eps_mix, sig_mix))
            elif 'none' in [ffi[1], ffj[1]]:
                ff_mix.append('{} {} none'.format(ffi[0], ffj[0]))
            elif ffi[1].lower() == ffj[1].lower() == 'feynman-hibbs-lennard-jones':
                eps_mix = sqrt(ffi[2] * ffj[2])
                if mixing_rule == 'lorentz-berthelot':
                    sig_mix = 0.5 * (ffi[3] + ffj[3])
                elif mixing_rule == 'jorgensen':
                    sig_mix = sqrt(ffi[3] * ffj[3])
                reduced_mass = ffi[4]  # assuming that ffi==ffj, for the moment
                ff_mix.append('{} {} feynman-hibbs-lennard-jones {:.5f} {:.5f} {:.5f}'.format(
                    ffi[0], ffj[0], eps_mix, sig_mix, reduced_mass))
            else:
                raise NotImplementedError('FFBuilder is not able to mix different/unknown potentials.')
    return ff_mix


def render_ff_def(ff_data, params, ff_mix_found):
    """Render the force_field.def file."""
    output = []
    output.append('# rules to overwrite (file generated by aiida-raspa)')
    output.append(0)
    output.append('# number of defined interactions')
    if params['separate_interactions'] or ff_mix_found:
        ff_list = []
        for molecule, ff_name in params['ff_molecules'].items():
            for atom_type, val in ff_data[molecule][ff_name]['atom_types'].items():
                ff_pot = val['force_field']
                if ff_pot == 'dummy_separate':  # Exclude molatoms-moldummy interactions
                    ff_list.append([atom_type] + ['none'])
                else:
                    ff_list.append([atom_type] + ff_pot)
        mixing_rule = params['mixing_rule'].lower()
        ff_mix = mix_molecule_ff(ff_list, mixing_rule)
        output.append(len(ff_mix))
        output.append('# type1 type2 interaction')
        output.extend(ff_mix)
    else:
        output.append(0)
    output.append('# mixing rules to overwrite')
    output.append(0)
    string = '\n'.join([str(x) for x in output]) + '\n'
    return string_to_singlefiledata(string, 'force_field.def')


def render_pseudo_atoms_def(ff_data, params):
    """Render the pseudo_atoms.def file."""
    output = []
    output.append('# number of pseudo atoms')

    pseudo_atoms_lines = []
    for molecule, ff_name in params['ff_molecules'].items():
        for atom_type, val in ff_data[molecule][ff_name]['atom_types'].items():
            type_settings = val['pseudo_atom']
            pseudo_atoms_lines.append(' '.join([str(x) for x in [atom_type] + type_settings]))

    pseudo_atoms_lines = check_ff_list(pseudo_atoms_lines)
    output.append(len(pseudo_atoms_lines))
    output.append('#type print as chem oxidation mass charge polarization B-factor radii connectivity ' +
                  'anisotropic anisotropic-type tinker-type')
    output.extend(pseudo_atoms_lines)
    string = '\n'.join([str(x) for x in output]) + '\n'
    return string_to_singlefiledata(string, 'pseudo_atoms.def')


def render_molecule_def(ff_data, params, molecule_name):
    """Render the molecule.def file containing the thermophysical data, geometry and intramolecular force field."""
    ff_name = params['ff_molecules'][molecule_name]
    ff_dict = ff_data[molecule_name][ff_name]
    output = []
    output.append('# critical constants: Temperature [T], Pressure [Pa], and Acentric factor [-] ' +
                  '(file generated by aiida-raspa)')
    output.append(ff_data[molecule_name]['critical_constants']['tc'])
    output.append(ff_data[molecule_name]['critical_constants']['pc'])
    output.append(ff_data[molecule_name]['critical_constants']['af'])
    if ff_dict['atomic_positions'] == 'flexible':  # read intermolecular forcefield from file
        ff_intermol_path = os.path.join(THISDIR, 'ff_flexible', '{}_{}.def'.format(molecule_name, ff_name))
        with open(ff_intermol_path, 'r') as ff_intermol:
            output += [line.strip() for line in ff_intermol.readlines()]
    else:  # rigid molecules: atomic positions provided as list of coordinates
        natoms = len(ff_dict['atomic_positions'])
        output.append('# Number Of atoms')
        output.append(natoms)
        output.append('# Number of groups (only whole molecule)')
        output.append(1)
        output.append('# Group-1: rigid/flexible')
        output.append('rigid')
        output.append('# Group-1: Number of atoms')
        output.append(natoms)
        output.append('# Group-1: Atomic positions')
        for i, line in enumerate(ff_dict['atomic_positions']):
            output.append(' '.join([str(x) for x in [i] + line]))
        output.append('# Chiral centers Bond  BondDipoles Bend  UrayBradley InvBend  Torsion Imp. Torsion Bond/Bond ' +
                      'Stretch/Bend Bend/Bend Stretch/Torsion Bend/Torsion IntraVDW IntraCoulomb')
        output.append(' '.join([str(x) for x in [0] + [natoms - 1] + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        if natoms > 1:
            output.append('# Bond stretch: atom n1-n2, type, parameters')
            for i in range(1, natoms):
                output.append('0 {} RIGID_BOND'.format(i))
        output.append('# Number of config moves')
        output.append(0)
    string = '\n'.join([str(x) for x in output]) + '\n'
    return string_to_singlefiledata(string, molecule_name + '.def')


@calcfunction
def ff_builder(params, cif_molecule=None):
    """AiiDA calcfunction to assemble force filed parameters into SinglefileData for Raspa."""

    # PARAMS_EXAMPLE = Dict( dict = {
    #   'ff_framework': 'UFF',              # See force fields available in ff_data.yaml as framework.keys()
    #   'ff_molecules': {                   # See molecules available in ff_data.yaml as ff_data.keys()
    #       'CO2': 'TraPPE',                    # See force fields available in ff_data.yaml as {molecule}.keys()
    #       'N2': 'TraPPE',
    #   },
    #   'shifted': True,                    # If True shift despersion interactions, if False simply truncate them.
    #   'tail_corrections': False,          # If True apply tail corrections based on homogeneous-liquid assumption
    #   'mixing_rule': 'Lorentz-Berthelot', # Options: 'Lorentz-Berthelot' or 'Jorgensen'
    #   'separate_interactions': True       # If True use framework's force field for framework-molecule interactions
    # })

    ff_data = load_yaml()
    if cif_molecule:
        ff_data = append_cif_molecule(ff_data, cif_molecule)
    out_dict = {}
    out_dict['ff_mixing_def'], ff_mix_found = render_ff_mixing_def(ff_data, params)
    out_dict['ff_def'] = render_ff_def(ff_data, params, ff_mix_found)
    out_dict['pseudo_atoms_def'] = render_pseudo_atoms_def(ff_data, params)
    for molecule_name in params['ff_molecules']:
        out_dict['molecule_{}_def'.format(molecule_name)] = render_molecule_def(ff_data, params, molecule_name)

    return out_dict