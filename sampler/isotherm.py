import os, sys
import numpy as np
import subprocess

"""

There are some global parameters and functions to run before starting, this part will be moved to package.

dmff_path = "/home/yutao/project/github/DMFF/UFF_opt/", this parameter will be set in the other scripts about DMFF
"""

# The aiida working directory, it changes when it moves to other computer
aiida_path = "/home/yutao/project/aiida/applications/"


def generateconfig(ff_data, temperature, pressure_list, cif_path, output_name, molecule):
    global aiida_path
    
    with open(os.path.join(aiida_path, 'exp_config.py'), 'w') as f:
        f.write("ff_data = '{}'\n".format(ff_data))
        f.write("Temperature = {}\n".format(temperature))
        f.write("pressure_list = {}\n".format(pressure_list))
        f.write("cif_path = '{}'\n".format(cif_path))
        f.write("output_path = '{}'\n".format(os.path.join(aiida_path, output_name)))
        f.write("molecule = '{}'\n".format(molecule))
def submit_mof(output_name):
    global aiida_path
    script_path = os.path.join(aiida_path, 'submit_ff.sh')
    command = [script_path, aiida_path, output_name]
    completed_process = subprocess.run(command, capture_output=True, cwd=aiida_path,text=True)
    print("As long as it finishes,",completed_process.returncode)
    # Check the return code
    if completed_process.returncode == 0:
        # The script finished successfully
        #print("Script finished successfully!")
        # Display the output in the notebook
        #print("Script output:")
        print(completed_process.stdout)
        # Continue with your program logic here
    else:
        # The script encountered an error
        #print("Script encountered an error:")
        print(completed_process.stderr)
        # Handle the error or exit the program

def submit_mofs(structure_path, ff_data,molecule, suffix = ""):
    basename, _ = os.path.splitext(ff_data)
    output_name = f"{basename}{suffix}.log"
    for mof in os.listdir(structure_path):
        dest_path = os.path.join(structure_path, mof)
        if not os.path.isdir(dest_path) or mof.startswith("."):
            continue
        print("Go to ", dest_path)
        cif_path = [file for file in os.listdir(dest_path) if file.endswith(".cif")][0]
        cif_path = os.path.join(dest_path, cif_path)
        isotherm_path = [file for file in os.listdir(dest_path) if file.endswith("K.csv")]
        if len(isotherm_path) == 0:
            raise ValueError("No well defined isotherm file found in ", dest_path)
        temperature = int(isotherm_path[0].rstrip("K.csv"))
        isotherm_path = os.path.join(dest_path, isotherm_path[0])
        data = np.loadtxt(isotherm_path, delimiter=',')
        pressure_list = list(data[:,0])
        generateconfig(ff_data, temperature, pressure_list, cif_path, os.path.join(aiida_path, output_name),molecule)
        submit_mof(output_name)

def submit_mofs_list(mof_list, structure_path, ff_data,molecule, suffix = ""):
    basename, _ = os.path.splitext(ff_data)
    output_name = f"{basename}{suffix}.log"
    for mof in os.listdir(structure_path):
        if mof not in mof_list:
            continue
        dest_path = os.path.join(structure_path, mof)
        if not os.path.isdir(dest_path) or mof.startswith("."):
            continue
        print("Go to ", dest_path)
        cif_path = [file for file in os.listdir(dest_path) if file.endswith(".cif")][0]
        cif_path = os.path.join(dest_path, cif_path)
        isotherm_path = [file for file in os.listdir(dest_path) if file.endswith("K.csv")]
        if len(isotherm_path) == 0:
            raise ValueError("No well defined isotherm file found in ", dest_path)
        temperature = int(isotherm_path[0].rstrip("K.csv"))
        isotherm_path = os.path.join(dest_path, isotherm_path[0])
        data = np.loadtxt(isotherm_path, delimiter=',')
        pressure_list = list(data[:,0])
        generateconfig(ff_data, temperature, pressure_list, cif_path, os.path.join(aiida_path, output_name),molecule)
        submit_mof(output_name)
    
if __name__ == "__main__":

    '''
        This is a simple example to run isotherm workchain and bindingsite workchain for single strucutre
    '''
    
    ff_data = "UFF.json"
    cif_path = "/home/yutao/project/Al-MOF/mil121/RSM0112.cif"
    isotherm_path = "/home/yutao/project/Al-MOF/mil121/273K.csv"
    output_name = "test.log"
    temperature = 303
    data = np.loadtxt(isotherm_path, delimiter=',')
    pressure_list = list(data[:,0])
    generateconfig(ff_data, temperature, pressure_list, cif_path, output_name)
    submit_mof(output_name)

    '''
    if the structure follow the rule, 
    e.g {Temperature}.csv is the experienmetal isotherms STP vs bar and only one cif file in the directory
    output_path = aiida_path+force field name
    I can use run_simple and only need the cif_path and UFF.json
    '''

    structure_path = "/home/yutao/project/Al-MOF/"
    ff_data = "try_0226.json"
    submit_mofs(structure_path, ff_data)