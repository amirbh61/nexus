#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 19:34:20 2023

@author: amir
"""

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.style.use('classic')
from tqdm import tqdm
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
import re

# In[0]
# generate new geant4 macros for all geometries and all possible runs

import itertools

# Geometry parameters
geometry_params = {
    'ELGap': ['1 mm', '10 mm'],
    'pitch': ['5 mm', '10 mm', '15.6 mm'],
    'distanceFiberHolder': ['-1 mm', '2 mm', '5 mm'],
    'distanceAnodeHolder': ['2.5 mm', '5 mm', '10 mm'],
    'holderThickness': ['10 mm'],
    'TPBThickness': ['2.2'] # microns
}

# Run parameters
run_params = {
    'x': ['val mm'],
    'y': ['0 mm'],
    'z': ['-5.0023 mm'],
}

unit_cell_source_spacing = 0.5 # mm, spacing between sources in different runs
z = 0 # junk number, value has no meaning but must exist as input for geant
seed = 10000

original_macro_path = r'/home/amir/Products/geant4/geant4-v11.0.1/MySims/nexus/macros/SquareOpticalFiber.config.mac'
output_macro_Mfolder = r'/home/amir/Products/geant4/geant4-v11.0.1/MySims/nexus/SquareFiberMacros/'

if not os.path.isdir(output_macro_Mfolder):
    os.mkdir(output_macro_Mfolder)
    
# Open your macro file and read in the data
with open(original_macro_path, "r") as f:
    template = f.read()

# Get the Cartesian product of all parameter values
geometry_combinations = list(itertools.product(*geometry_params.values()))

for i, combination in tqdm(enumerate(geometry_combinations)):
    
    output_macro_geom_folder = output_macro_Mfolder + \
                                f'ELGap={combination[0].replace(" ","")}_' + \
                                f'pitch={combination[1].replace(" ","")}_' + \
                                f'distanceFiberHolder={combination[2].replace(" ","")}_' + \
                                f'distanceAnodeHolder={combination[3].replace(" ","")}_' + \
                                f'holderThickness={combination[4].replace(" ","")}'                              
    if not os.path.isdir(output_macro_geom_folder):
        os.mkdir(output_macro_geom_folder)
        
    macro = template
    for key, value in zip(geometry_params.keys(), combination):
        if key == 'pitch':
            # NOTE: the source spacing limits depend on the pitch
            pitch_value = float(value.split(' ')[0])
            x = np.arange(-pitch_value/2, pitch_value/2 + unit_cell_source_spacing,
                          unit_cell_source_spacing )
            y = np.arange(-pitch_value/2, pitch_value/2 + unit_cell_source_spacing,
                          unit_cell_source_spacing )
            
            # Update the run parameters
            run_params = {
                'x': [str(val) + ' mm' for val in x],
                'y': [str(val) + ' mm' for val in y],
            }
        macro = macro.replace('${' + key + '}', value)

    # Fill x,y,seed values in macros for each geometry setup
    for x_val in run_params['x']:
        for y_val in run_params['y']:
            macro = macro.replace('${x}', x_val)
            macro = macro.replace('${y}', y_val)
            macro = macro.replace('${seed}', str(seed))
            seed += 1
            # print(type(x_val))
            # print(x_val)
            # convert to float and take 3 first digits
            x = round(float(x_val.replace(" mm","")),3)
            y = round(float(y_val.replace(" mm","")),3)
            
            
            output_macro_path = output_macro_geom_folder + \
            f'/x={x}mm_' + \
            f'y={y}mm.config.mac'
            
            # Write the new macro to a new file
            with open(output_macro_path, "w") as f:
                f.write(macro)


# In[1]













