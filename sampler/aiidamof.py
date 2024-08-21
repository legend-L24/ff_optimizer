'''
Hello, this part is to analysis the output information from aiida workchain and 
compare them with DFT-level binding energy and experiental isotherms

The input is *.log file, which format looks like
The tested force field is:  try_0226.json
The tested cif is:  /home/yutao/project/Al-MOF/WOJJOV/RSM2706.cif
The simulation temperature is:  195
This is the final pk values for isotherm workflow:  781611
This is the final pk values for binding sites workflow:  781629
The file is stored in aiida_path (default)
'''

# The aiida working directory, it changes when it moves to other computer
aiida_path = "/home/yutao/project/aiida/applications/"



R = 8.31446261815324 # J/(mol K)

from aiida.orm import QueryBuilder, Group, WorkChainNode, Dict, StructureData, CifData
import aiida
import numpy as np
import pytest
import os
from ase.io import read, write
aiida.load_profile()
import pytest
import matplotlib.pyplot as plt
import csv

# Default global variable "mol/kg" or "per unit cell" or "per metal site"
#Isotherm_unit = "mol/kg" #"per metal site"  #"per unit cell" 
# Ioshterm_unit is controlled by Isotherm_unit, energy_unit = binding_energy

# some constant will be used 
metal_elements = ['Ac','Ag','Al','Am','Au','Ba','Be','Bi',
				  'Bk','Ca','Cd','Ce','Cf','Cm','Co','Cr',
				  'Cs','Cu','Dy','Er','Es','Eu','Fe','Fm',
				  'Ga','Gd','Hf','Hg','Ho','In','Ir',
				  'K','La','Li','Lr','Lu','Md','Mg','Mn',
				  'Mo','Na','Nb','Nd','Ni','No','Np','Os',
				  'Pa','Pb','Pd','Pm','Pr','Pt','Pu','Ra',
				  'Rb','Re','Rh','Ru','Sc','Sm','Sn','Sr',
				  'Ta','Tb','Tc','Th','Ti','Tl','Tm','U',
				  'V','W','Y','Yb','Zn','Zr']

def count_metal_atoms(atoms):
    # List of symbols for common transition metals
    global metal_elements

    count = 0
    for atom in atoms:
        if atom.symbol in metal_elements:
            count += 1

    return count
class AiidaMof: 
    # forcefield must be the first line. folder mofname are also necessary for a AiidaMof object
    def __init__(self, folder, cifname, mofname,forcefield, isotherm_unit="mol/kg", temperature=None, isotherm_pk=None, binding_pk=None):
        # there are some default choice you can change in a class
        self.folder = folder
        self.mofname = mofname
        self.cifname = cifname
        self.temperature = temperature
        self.isotherm_pk = isotherm_pk
        self.binding_pk = binding_pk
        self.forcefield = forcefield
        self.isotherm_unit = isotherm_unit
        self.exp_pressure = None
        self.exp_loading = None # this is a list
        self.ff_pressure = None
        self.ff_loading = None # this is a list
        self.ff_energy = {} 
        self.vdw_energy = {}
        self.coul_energy = {}
        self.dft_energy = None
        self.co_isotherms = None # this is a directories included points with the same pressure from simulated isotherms and experimental isotherms
        self.isotherm_enthalpy = None # this is enthalpy of adsorption while a certain pressure and uptake
        self.isotherm_enthalpy_dev = None # this is the deviation of enthalpy of adsorption because of many configurations
        self.ff_loading_dev = None # this is the deviation of loading because of many configurations
        self.dft_dict = {} # when diffferent force field generate different DFT value, I store it. This property only work in merge function
        self.pova = None # this is the pova value for the mof
    @staticmethod
    def compute_molar_mass(atoms):
        return sum(atom.mass for atom in atoms)
    @staticmethod
    def extract_same_pressure_points(exp_pressure, exp_loading, ff_pressure, ff_loading, error=0.01):

        co_pressures = []
        exp_part = []
        ff_part = []

        for p1, l1 in zip(exp_pressure, exp_loading):
            for p2, l2 in zip(ff_pressure, ff_loading):
                if abs(p1 - p2) / p1 < error:
                    co_pressures.append(p1)
                    exp_part.append(l1)
                    ff_part.append(l2)
                    break
        return co_pressures, exp_part, ff_part
    def set_isotherm(self):
        try:
            qb = QueryBuilder()
            qb.append(WorkChainNode, filters={'id':self.isotherm_pk}, tag="workchain")
            qb.append(Dict, with_incoming="workchain")
            outdict_ls = qb.all()[:]
            self.ff_pressure = np.array(outdict_ls[0][0].get_dict()['isotherm']['pressure'])
            self.pova = outdict_ls[0][0].get_dict()['POAV_A^3']
            if self.isotherm_unit == "mol/kg":
                self.ff_loading = outdict_ls[0][0].get_dict()['isotherm']['loading_absolute_average']
                self.ff_loading_dev = np.array(outdict_ls[0][0].get_dict()['isotherm']['loading_absolute_dev'])
            elif self.isotherm_unit == "per unit cell":
                atoms = read(os.path.join(self.folder, self.cifname))
                molar_mass = self.compute_molar_mass(atoms)
                self.ff_loading = np.array(outdict_ls[0][0].get_dict()['isotherm']['loading_absolute_average'])*molar_mass/1000
                self.ff_loading_dev = np.array(outdict_ls[0][0].get_dict()['isotherm']['loading_absolute_dev'])*molar_mass/1000
            elif self.isotherm_unit == "per metal site":
                atoms = read(os.path.join(self.folder, self.cifname))
                molar_mass = self.compute_molar_mass(atoms)
                numberofmetal = count_metal_atoms(atoms)
                self.ff_loading = np.array(outdict_ls[0][0].get_dict()['isotherm']['loading_absolute_average'])*molar_mass/1000/numberofmetal
                self.ff_loading_dev = np.array(outdict_ls[0][0].get_dict()['isotherm']['loading_absolute_dev'])*molar_mass/1000/numberofmetal
            else:
                raise ValueError("Wrong global unit for isotherm")
            self.isotherm_enthalpy = np.array(outdict_ls[0][0].get_dict()['isotherm']['enthalpy_of_adsorption_average'])
            self.isotherm_enthalpy_dev = np.array(outdict_ls[0][0].get_dict()['isotherm']['enthalpy_of_adsorption_dev'])
        except:
            print("Isotherm workflow failed for ", self.isotherm_pk, self.mofname)
        
        try:
            # read experimental isotherms from {Temperature}K.csv
            exp_path = os.path.join(self.folder, f"{self.temperature}K.csv")
            exp_isotherm = np.loadtxt(exp_path, delimiter=',')
            exp_isotherm = np.atleast_2d(exp_isotherm)
            if self.isotherm_unit == "mol/kg":
                transfer_unit = 1/22.4 #from STP to mol/Kg
            elif self.isotherm_unit == "per unit cell":
                transfer_unit = 1/22.4*molar_mass/1000
            elif self.isotherm_unit == "per metal site":
                transfer_unit = 1/22.4*molar_mass/1000/numberofmetal
            else:
                raise ValueError("Wrong global unit for isotherm")
            self.exp_loading = exp_isotherm[:,1]*transfer_unit
            self.exp_pressure = exp_isotherm[:,0] # unit bar
        except:
            print("No experimental isotherm found for ", self.mofname)
    def set_binding(self):
        binding_pk = self.binding_pk
        qb = QueryBuilder()
        qb.append(WorkChainNode, filters={'id':binding_pk}, tag="workchain")
        workchain = qb.all()[:][0][0]
        if not workchain.is_finished:
            print(f"WorkChain with pk={binding_pk} has not finished yet.")
        qb.append(Dict, with_incoming="workchain")
        outdict_ls = qb.all()[:]
        #temperatue_minimization = 50 # this value comes from default value in binding site workchain, the temperatue last step is 50K
        if len(outdict_ls)==2:
            try:
                self.ff_energy[self.forcefield] = outdict_ls[0][0]["energy_host/ads_tot_final"][-1] #+ R*temperatue_minimization/1000 # temperature correction
                self.coul_energy[self.forcefield] = outdict_ls[0][0]["energy_host/ads_coulomb_final"][-1]
                self.vdw_energy[self.forcefield] = outdict_ls[0][0]["energy_host/ads_vdw_final"][-1]
                self.dft_energy = outdict_ls[1][0]["binding_energy_corr"]
            except:
                self.ff_energy[self.forcefield] = outdict_ls[1][0]["energy_host/ads_tot_final"][-1] #+ R*temperatue_minimization/1000 # temperature correction
                self.coul_energy[self.forcefield] = outdict_ls[0][0]["energy_host/ads_coulomb_final"][-1]
                self.vdw_energy[self.forcefield] = outdict_ls[0][0]["energy_host/ads_vdw_final"][-1]
                self.dft_energy = outdict_ls[0][0]["binding_energy_corr"]
        else:
            print(f"Without DFT-Binding energy, num: {len(outdict_ls)}, in binding site workchain {binding_pk}, {self.mofname}")
            try:
                self.ff_energy[self.forcefield] = outdict_ls[0][0]["energy_host/ads_tot_final"][-1] #+ R*temperatue_minimization/1000 # temperature correction
            except:
                self.ff_energy[self.forcefield] = outdict_ls[1][0]["energy_host/ads_tot_final"][-1] #+ R*temperatue_minimization/1000 # temperature correction
    def compare_isotherm(self):
        co_pressures, exp_part, ff_part = self.extract_same_pressure_points(self.exp_pressure, self.exp_loading, self.ff_pressure, self.ff_loading)
        self.co_isotherms = {"pressure": co_pressures, "experiment": exp_part, self.forcefield: ff_part}
        
    def extractdata(self):
        #print("This is", self.isotherm_pk, self.binding_pk)
        if self.isotherm_pk:
            self.set_isotherm()
            self.compare_isotherm()
        if self.binding_pk: 
            self.set_binding()
        

class AiidaMofs:
    def __init__(self, log_name,isotherm_unit, isdefaultpath=True):
        if isdefaultpath:
            log_path = os.path.join(aiida_path, log_name) # os.path.join can delete overlap of path automatically
        else:
            log_path = log_name
        self.mofs = self.log_file(log_path, isotherm_unit)
        
        self.extractdata()
    def __len__(self):
        return len(self.mofs)
    def __getitem__(self, index):
        return self.mofs[index]

    def __setitem__(self, index, value):
        self.mofs[index] = value

    def __delitem__(self, index):
        del self.mofs[index]
    def extractdata(self):
        for mof in self.mofs:
            #mof.extractdata()
            try:
                mof.extractdata()
            except:
                print(f"meet error in extract data from structure {mof.mofname}")
            
    @staticmethod
    def log_file(log_path, isotherm_unit="mol/kg"):
        mofs = []
        aiida_mof = None
        with open(log_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "The tested force field is:" in line:
                    forcefield = os.path.splitext(line.split()[-1])[0]
                if "The tested cif is:" in line:
                    if aiida_mof is not None:
                        mofs.append(aiida_mof)
                    path = line.split()[-1]
                    folder = os.path.dirname(path)
                    parts = os.path.normpath(path).split(os.sep)
                    cifname = parts[-2]
                    mofname = parts[-1]
                    aiida_mof = AiidaMof(folder, mofname, cifname,  forcefield, isotherm_unit)
                if aiida_mof is not None:
                    if "The simulation temperature is:" in line:
                        try:
                            aiida_mof.temperature = int(line.split()[-1])
                        except:
                            aiida_mof.temperature = float(line.split()[-1])
                    if "This is the final pk values for isotherm workflow:" in line:
                        aiida_mof.isotherm_pk = int(line.split()[-1])
                    if "This is the final pk values for binding sites workflow:" in line:
                        aiida_mof.binding_pk = int(line.split()[-1])
            if aiida_mof is not None:
                mofs.append(aiida_mof)
        return mofs

'''
Hello, this part to input AiidaMofs object and output the results, I do not would like to get a very long class and seperate the plot function part
Basically, I wish to plot results from one kinds of force field.
Furthermore, I wish to put results from different force field to one figure is a very easy things
'''


def create_fig_axs(N):
    # some default parameters for the plot function
    if N == 1:
        scale = 3
    elif 1 < N < 4:
        scale = 2
    else:
        scale = 1

    subplot_width = 5*scale
    subplot_height = 5*scale
    hspace = 0.3
    wspace = 0.3

    rows = int(np.ceil(N / 3))
    cols = min(3, N)

    fig_width = (subplot_width * cols) + (wspace * (cols+1))  # 总宽度包括子图和间隙
    fig_height = (subplot_height * rows) + (hspace * (rows+1))  # 总高度包括子图和间隙
    fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    fig.subplots_adjust(hspace=hspace, wspace=wspace)
    
    if N == 1:
        axs = np.array([[axs]])
    elif N <= 3:
        axs = np.array([axs])
    else:
        axs = np.array(axs)

    # Hide unused subplots
    for idx in range(N, rows*cols):
        i, j = divmod(idx, 3)
        axs[i, j].axis('off')

    return fig, axs

def create_fig_axs_small(N):
    # some default parameters for the plot function
    if N == 1:
        scale = 3
    elif 1 < N < 4:
        scale = 2
    else:
        scale = 1

    subplot_width = 4*scale
    subplot_height = 4*scale
    hspace = 0.1
    wspace = 0.2

    rows = int(np.ceil(N / 3))
    cols = min(3, N)

    fig_width = (subplot_width * cols) + (wspace * (cols+1))  # 总宽度包括子图和间隙
    fig_height = (subplot_height * rows) + (hspace * (rows+1))  # 总高度包括子图和间隙
    fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    fig.subplots_adjust(hspace=hspace, wspace=wspace)
    
    if N == 1:
        axs = np.array([[axs]])
    elif N <= 3:
        axs = np.array([axs])
    else:
        axs = np.array(axs)

    # Hide unused subplots
    for idx in range(N, rows*cols):
        i, j = divmod(idx, 3)
        axs[i, j].axis('off')

    return fig, axs

def plot_isotherm(aiidamofs,pressure_unit="bar"):
    # global parameter
    fz = 18
    aiidamofs = [aiidamof for aiidamof in aiidamofs if aiidamof.co_isotherms]
    
    fig, axs = create_fig_axs(len(aiidamofs))
    idx = 0
    for aiidamof in aiidamofs:
        if aiidamof.co_isotherms: 
            i, j = divmod(idx, 3) # the certain way to calculate positions of figures
            for keys, values in aiidamof.co_isotherms.items():
                if keys == 'pressure':
                    continue
                if pressure_unit == "mbar":
                    transfer_pressure_unit = 1000
                elif pressure_unit == "bar":
                    transfer_pressure_unit = 1
                else:
                    raise ValueError("pressure_unit should be 'mbar' or 'bar'")

                ax = axs[i,j]
                ax.plot(aiidamof.co_isotherms['pressure']*transfer_pressure_unit, aiidamof.co_isotherms[keys], marker='D', label=keys)
                ax.set_title(f"{aiidamof.mofname} at {aiidamof.temperature} K" ,fontsize=fz)
                ax.legend(fontsize=fz)
                ax.set_ylabel(f"{aiidamof.isotherm_unit}", fontsize=fz)
                ax.set_xlabel(f"Pressure ({pressure_unit})", fontsize=fz)
                ax.tick_params(axis='both', which='major', labelsize=fz-2)
                ax.tick_params(axis='both', which='minor', labelsize=fz-8)
                ax.set_ylim([0, max(max(aiidamof.ff_loading), max(aiidamof.exp_loading))*1.2])
            idx += 1
    plt.show()




def plot_isotherm_errorbar(aiidamofs,experiment=True, pressure_unit="mbar", FFname=None):
    # global parameter
    fz = 16
    aiidamofs = [aiidamof for aiidamof in aiidamofs if aiidamof.co_isotherms]
    
    fig, axs = create_fig_axs_small(len(aiidamofs))
    idx = 0
    for aiidamof in aiidamofs:
        if not FFname:
            FFname = aiidamof.forcefield    
        if not aiidamof.co_isotherms:
            continue
        i, j = divmod(idx, 3) # the certain way to calculate positions of figures
        ax = axs[i,j]
        if pressure_unit == "mbar":
            aiidamof.ff_pressure = aiidamof.ff_pressure * 1000
            aiidamof.exp_pressure = aiidamof.exp_pressure * 1000
        elif pressure_unit == "bar":
            aiidamof.ff_pressure = aiidamof.ff_pressure
            aiidamof.exp_pressure = aiidamof.exp_pressure
        else:
            raise ValueError("pressure_unit should be 'mbar' or 'bar'")
        if experiment:
            ax.scatter(aiidamof.exp_pressure, aiidamof.exp_loading, marker='o', color='green', label='experiment')
        ax.errorbar(aiidamof.ff_pressure, aiidamof.ff_loading, yerr=aiidamof.ff_loading_dev, fmt='o', capsize=5, capthick=1, label=FFname)
        ax.set_title(f"{aiidamof.mofname} at {aiidamof.temperature} K" ,fontsize=fz)
        ax.legend(fontsize=fz)
        ax.set_ylabel(f"{aiidamof.isotherm_unit}", fontsize=fz)
        ax.set_xlabel(f"Pressure ({pressure_unit})", fontsize=fz)
        ax.tick_params(axis='both', which='major', labelsize=fz-2)
        ax.tick_params(axis='both', which='minor', labelsize=fz-8)
        ax.set_ylim([0, max(max(aiidamof.ff_loading), max(aiidamof.exp_loading))*1.2])
        idx += 1
    plt.tight_layout()
    plt.legend(fontsize=fz)

# merge two isotherms from two AiidaMofs, which contains the same structures but simulated by different force field
def merge_aiida_mofs(aiida_mofs1, aiida_mofs2):
    mof_dict = {}
    for aiida_mof in aiida_mofs1.mofs+aiida_mofs2.mofs:
        if not aiida_mof.co_isotherms:
            continue
        if aiida_mof.mofname not in mof_dict:
            mof_dict[aiida_mof.mofname] = aiida_mof
        else:
            if mof_dict[aiida_mof.mofname].co_isotherms['pressure'] != aiida_mof.co_isotherms['pressure']:
                raise ValueError("Different legenth of pressure list")
            mof_dict[aiida_mof.mofname].co_isotherms.update(aiida_mof.co_isotherms)
            if mof_dict[aiida_mof.mofname].ff_energy and aiida_mof.ff_energy:
                if mof_dict[aiida_mof.mofname].ff_energy:
                    mof_dict[aiida_mof.mofname].ff_energy.update(aiida_mof.ff_energy)
                elif aiida_mof.ff_energy:
                    mof_dict[aiida_mof.mofname].ff_energy = aiida_mof.ff_energy
            if mof_dict[aiida_mof.mofname].dft_energy and aiida_mof.dft_energy:
                if not mof_dict[aiida_mof.mofname].dft_energy:
                    mof_dict[aiida_mof.mofname].dft_energy = aiida_mof.dft_energy
    return list(mof_dict.values())

def merge_aiida_mofs_list(aiida_mofs1, aiida_mofs2):
    mof_dict = {}
    for aiida_mof in aiida_mofs1+aiida_mofs2:
        if aiida_mof.mofname not in mof_dict:
            mof_dict[aiida_mof.mofname] = aiida_mof
        else:
            if mof_dict[aiida_mof.mofname].ff_energy or aiida_mof.ff_energy:
                if mof_dict[aiida_mof.mofname].ff_energy:
                    mof_dict[aiida_mof.mofname].ff_energy.update(aiida_mof.ff_energy)
                elif aiida_mof.ff_energy:
                    mof_dict[aiida_mof.mofname].ff_energy = aiida_mof.ff_energy
            if mof_dict[aiida_mof.mofname].vdw_energy or aiida_mof.vdw_energy:
                if mof_dict[aiida_mof.mofname].vdw_energy:
                    mof_dict[aiida_mof.mofname].vdw_energy.update(aiida_mof.vdw_energy)
                elif aiida_mof.vdw_energy:
                    mof_dict[aiida_mof.mofname].vdw_energy = aiida_mof.vdw_energy
            if mof_dict[aiida_mof.mofname].coul_energy or aiida_mof.coul_energy:
                if mof_dict[aiida_mof.mofname].coul_energy:
                    mof_dict[aiida_mof.mofname].coul_energy.update(aiida_mof.coul_energy)
                elif aiida_mof.coul_energy:
                    mof_dict[aiida_mof.mofname].coul_energy = aiida_mof.coul_energy
            if mof_dict[aiida_mof.mofname].dft_energy or aiida_mof.dft_energy:
                #if not mof_dict[aiida_mof.mofname].dft_energy:
                if aiida_mof.dft_energy:
                    mof_dict[aiida_mof.mofname].dft_energy = aiida_mof.dft_energy
                    mof_dict[aiida_mof.mofname].dft_dict[aiida_mof.forcefield] = aiida_mof.dft_energy
    return list(mof_dict.values())


def write_to_csv(aiida_mofs, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['cifname',  'ff_pressure(bar)' ,'ff_loading(mol/kg)','ff_loading_dev(mol/kg)','enthalpy_of_adsorption(kJ/mol)','enthalpy_of_adsorption_dev(kJ/mol)','POAV_A^3']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for aiida_mof in aiida_mofs:
            writer.writerow({'cifname': aiida_mof.cifname, 
                             'ff_loading(mol/kg)': list(aiida_mof.ff_loading),
                             'ff_loading_dev(mol/kg)': list(aiida_mof.ff_loading_dev),
                             'ff_pressure(bar)': list(aiida_mof.ff_pressure),
                             'enthalpy_of_adsorption(kJ/mol)': list(aiida_mof.isotherm_enthalpy),
                             'enthalpy_of_adsorption_dev(kJ/mol)': list(aiida_mof.isotherm_enthalpy_dev),
                             'POAV_A^3': aiida_mof.pova})

'''
this is a simple plot function for the data with different elements, 
the function provides choices of units, log scale, and the range of x and y axis
'''

def custom_uppercase(s):
    elements = ['Al', 'Mg', 'Ca','Ga','In','Ba','Sr',"Py"]  # 定义需要保留第二个字母小写的元素列表
    result = []
    i = 0    
    while i < len(s):
        if i + 1 < len(s) and s[i:i+2] in elements:  # 检查是否是元素名称
            result.append(s[i:i+2])  # 将元素名称直接添加到结果中
            i += 2
        else:
            result.append(s[i].upper())  # 否则将字符转换为大写并添加到结果中
            i += 1    
    return ''.join(result)# 测试例子


def plot_single_isotherm(plt,filename, isotherm_unit, suffix="", log_scale=False):
    '''
    plot the isotherm data with different elements
    '''
    # read the data
    aiidamofs = AiidaMofs(filename, isotherm_unit)
    # plot the data
    for aiidamof in aiidamofs:
        if aiidamof.co_isotherms:
            
            labelname = custom_uppercase(aiidamof.mofname)+suffix
            if not log_scale:
                plt.scatter(aiidamof.co_isotherms["experiment"], aiidamof.co_isotherms[aiidamof.forcefield], label=labelname)
            else:
                plt.scatter(np.log10(aiidamof.co_isotherms["experiment"]), np.log10(aiidamof.co_isotherms[aiidamof.forcefield]), label=labelname)

if __name__ == "__main__":

    log_name = "UFF_Mg.log"
    log_path = os.path.join(aiida_path, log_name)
    aiida_mofs = AiidaMofs(log_path)
    # Get an AiidaMof object
    aiida_mof = aiida_mofs[0]
    # Print the folder of the AiidaMof object
    for aiida_mof in aiida_mofs:
        print(aiida_mof.forcefield)
        print(aiida_mof.isotherm_pk)
        print(aiida_mof.folder)
        print(aiida_mof.co_isotherms)
    plot_isotherm(aiida_mofs)
