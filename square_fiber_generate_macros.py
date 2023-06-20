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
import itertools


# In[0]
# generate new geant4 macros for all geometries and all possible runs

# Geometry parameters
geometry_params = {
    'ELGap': ['1', '10'],
    'pitch': ['5', '10', '15.6'],
    'distanceFiberHolder': ['-1', '2', '5'],
    'distanceAnodeHolder': ['2.5', '5', '10'],
    'holderThickness': ['5','10'],
    'TPBThickness': ['2.2'] # microns
}

# Run parameters
run_params = {
    'x': ['0'],
    'y': ['0'],
    'z': ['0'],
}

unit_cell_source_spacing = 0.5 # mm, spacing between sources in different runs
z = 0 # junk number, value has no meaning but must exist as input for geant
seed = 10000

original_macro_path = r'/home/amir/Products/geant4/geant4-v11.0.1/MySims/nexus/macros/SquareOpticalFiberCluster.config.mac'
output_macro_Mfolder = r'/home/amir/Products/geant4/geant4-v11.0.1/MySims/nexus/SquareFiberMacrosAndOutputs/'

if not os.path.isdir(output_macro_Mfolder):
    os.mkdir(output_macro_Mfolder)
    
# Open your macro file and read in the data
with open(original_macro_path, "r") as f:
    template = f.read()

# Get the Cartesian product of all parameter values
geometry_combinations = list(itertools.product(*geometry_params.values()))

for i, combination in tqdm(enumerate(geometry_combinations)):
    
    output_macro_geom_folder = output_macro_Mfolder + \
                                f'ELGap={combination[0].replace(" ","")}mm_' + \
                                f'pitch={combination[1].replace(" ","")}mm_' + \
                                f'distanceFiberHolder={combination[2].replace(" ","")}mm_' + \
                                f'distanceAnodeHolder={combination[3].replace(" ","")}mm_' + \
                                f'holderThickness={combination[4].replace(" ","")}mm'
                                
    if not os.path.isdir(output_macro_geom_folder):
        os.mkdir(output_macro_geom_folder)
        
    # Fill x,y,seed values in macros for each geometry setup
    for x_val in run_params['x']:
        for y_val in run_params['y']:
            x_val = str(round(float(x_val),3))
            y_val = str(round(float(y_val),3))
            
            macro = template # Define a fresh copy of the macro template
            for key, value in zip(geometry_params.keys(), combination):
                if key == 'pitch':
                    pitch_value = float(value.split(' ')[0])
                    x = np.arange(-pitch_value/2, pitch_value/2 + unit_cell_source_spacing,
                                  unit_cell_source_spacing )
                    y = np.arange(-pitch_value/2, pitch_value/2 + unit_cell_source_spacing,
                                  unit_cell_source_spacing )
                    
                    # Update the run parameters
                    run_params = {
                        'x': [str(val) for val in x],
                        'y': [str(val) for val in y],
                    }
                macro = macro.replace('${' + key + '}', value)
                
            macro = macro.replace('${x}', x_val)
            macro = macro.replace('${y}', y_val)
            macro = macro.replace('${seed}', str(seed))
            seed += 1
            
            output_SiPM_path = output_macro_geom_folder + \
                f'/SiPM_hits_x={x_val}mm_y={y_val}mm.txt'                     
            macro = macro.replace('${sipmOutputFile_}', output_SiPM_path)
            
            output_TPB_path = output_macro_geom_folder + \
                f'/TPB_hits_x={x_val}mm_y={y_val}mm.txt'
            macro = macro.replace('${tpbOutputFile_}', output_TPB_path)
            
            output_macro_path = output_macro_geom_folder + \
            f'/x={x_val}mm_y={y_val}mm.config.mac'
                                  
            # Write the new macro to a new file
            with open(output_macro_path, "w") as f:
                f.write(macro)


    











