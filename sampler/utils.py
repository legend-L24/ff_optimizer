from aiida.orm import QueryBuilder, Group, WorkChainNode, Dict, StructureData, CifData
import numpy as np
from aiida import load_profile

import pandas as pd
import os
load_profile()


aiida_path = "/home/yutao/project/aiida/applications/"

# energy transfer from Force field in raspa to openmm
Transfer_energy_unit = 254.152/2.11525
Transfer_length_unit = 10

import json

def write_json(ff_dict, filename):
    with open(filename, 'w') as f:
        json.dump(ff_dict, f, indent=4)
    print(f"{filename} has been written")

import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

def write_xml(ff_dict, path):
    global Transfer_energy_unit, Transfer_length_unit
    # Create the root element
    root = ET.Element("ForceField")

    # Create the LennardJonesForce element
    lj_force = ET.SubElement(root, "LennardJonesForce")
    lj_force.set("lj14scale", "0.50000")
    # Add Atom elements for each atom
    for name in ff_dict.keys():
        atom = ET.SubElement(lj_force,"Atom")
        atom.set("epsilon", str(ff_dict[name][1]/Transfer_energy_unit))
        atom.set("sigma", str(ff_dict[name][2]/Transfer_length_unit))
        atom.set("type", name.rstrip('_'))
        atom.set("mask", "false")


    #<Atom epsilon="0.65757" sigma="0.305"  type="O_co2" mask="true"/>
    #<Atom epsilon="0.22469" sigma="0.28" type="C_co2" mask="true"/>
    atom = ET.SubElement(lj_force, "Atom")
    atom.set("epsilon", "0.65757")
    atom.set("sigma", "0.305")
    atom.set("type", "O_co2")
    atom.set("mask", "true")


    atom = ET.SubElement(lj_force, "Atom")
    atom.set("epsilon", "0.22469")
    atom.set("sigma", "0.28")
    atom.set("type", "C_co2")
    atom.set("mask", "true")


    # Convert the ElementTree to a formatted string with line breaks
    xml_string = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")

    # Remove the XML declaration line
    xml_lines = xml_string.split("\n")[1:]

    # Write the formatted XML string to a file
    with open(path, "w") as xml_file:
        xml_file.write("\n".join(xml_lines))
    print(f"{path} has been written")


def extract_ff(filename, pk, element_list, dest=aiida_path):
    qb = QueryBuilder()
    qb.append(WorkChainNode, filters={'id':pk}, tag="workchain")
    workchain = qb.all()[0][0]
    ff = workchain.inputs.parameters.get_dict()['ff_optim']
    ff_dict = {k: ff[k] for k in element_list}
    write_json(ff_dict, os.path.join(dest, f"{filename}.json"))
    write_xml(ff_dict, os.path.join(dest, f"{filename}.xml"))
    return ff_dict


def select_max_spacing_points(data_points, num_points_to_select):

    sorted_points = data_points[np.argsort(data_points[:, 0])]    # 初始化选择点的列表
    # 排序数据点
    if len(sorted_points) < num_points_to_select:
        num_points_to_select = len(sorted_points)
    selected_indices = [0, len(sorted_points) - 1]  # 选第一个和最后一个点的索引    # 用贪心算法选择剩余的数据点
    for _ in range(num_points_to_select - 2):  # 已经选择了两个点，还需选择 (num_points_to_select - 2) 个
        max_gap = -1
        new_index = None
        # 找到当前选点间距最大的间隔，并插入其中间的点
        for i in range(len(selected_indices) - 1):
            left_index, right_index = selected_indices[i], selected_indices[i + 1]
            left, right = sorted_points[left_index, 0], sorted_points[right_index, 0]
            # 找到 left 和 right 之间最靠近中间的点
            candidate_idx = np.searchsorted(sorted_points[:, 0], (left + right) / 2)
            if candidate_idx >= len(sorted_points):
                candidate_idx = len(sorted_points) - 1
            candidate = sorted_points[candidate_idx]
            gap = min(candidate[0] - left, right - candidate[0])
            if gap > max_gap and candidate_idx not in selected_indices:
                max_gap = gap
                new_index = candidate_idx        
        selected_indices.append(new_index)
        selected_indices.sort()  # 保持索引有序    
    selected_points = sorted_points[selected_indices]    
    return selected_points

"""
this is a example to use it
from utils import process_csv

input_csv = '/home/yutao/project/InN-MOF/MMPF-8/273K_long.csv'  # 输入CSV文件路径
output_csv = '/home/yutao/project/InN-MOF/MMPF-8/273K.csv'  # 输出CSV文件路径
num_points_to_select = 20  # 要选择的数据点数量
process_csv(input_csv, output_csv, num_points_to_select)

"""
def process_csv(input_csv, output_csv, num_points_to_select):
    # 读取CSV文件
    data = pd.read_csv(input_csv).values    # 选择数据点
    selected_points = select_max_spacing_points(data, num_points_to_select)    # 保存结果到新的CSV文件
    pd.DataFrame(selected_points).to_csv(output_csv, header=False, index=False)# 示例调用

from datetime import datetime
def get_time():
    # Get the current date
    current_date = datetime.now()

    # Format the date as MMDD
    formatted_date = current_date.strftime("%m%d")

    # Print the formatted date
    return formatted_date
