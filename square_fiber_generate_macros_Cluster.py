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
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
import re
import itertools


# In[0]
# generate new geant4 macros for all geometries and all possible runs

# Geometry parameters
# geometry_params = {
#      'ELGap': ['1', '10'],
#      'pitch': ['5', '10', '15.6'],
#      'distanceFiberHolder': ['-1', '2', '5'],
#      'distanceAnodeHolder': ['2.5', '5', '10'],
#      'holderThickness': ['10'],
#      'TPBThickness': ['2.2'] # microns
# }

 # Geometry parameters - Exansion to NEXT database for studying optimal anode distance at varying fiber depths
geometry_params = {
     'ELGap': ['10'],
     'pitch': ['15.6'],
     'distanceFiberHolder': ['-1','2'],
     'distanceAnodeHolder': ['7.5'],
     'holderThickness': ['10'],
     'TPBThickness': ['2.2']  # microns
}

# # Run parameters
# run_params = {
#     'x': ['0'],
#     'y': ['0'],
#     'z': ['0'],
# }


# Run parameters
run_params = {
    'x': ['0'],
    'y': ['0'],
    'z': ['0'],
}

### IMPORTANT : BOTH MODES ARE NEEDED FOR EACH GEOMETRY !! ###
# Chose mode of source generation

# The Kr events
fixed_intervals = False # edges are included !
if fixed_intervals:
    unit_cell_source_spacing = 0.2 # mm, spacing between sources in different runs
    sub_dir = r'Geant4_Kr_events'

# The PSF events
random_events_xy = True
if random_events_xy:
    num_samples = 10000
    sub_dir = 'Geant4_PSF_events'



seed = 10000

original_config_macro_path = r'/gpfs0/arazi/users/amirbenh/Resolving_Power/nexus/macros/SquareOpticalFiberCluster.config.mac'
original_init_macro_path = r'/gpfs0/arazi/users/amirbenh/Resolving_Power/nexus/macros/SquareOpticalFiberCluster.init.mac'
output_macro_Mfolder = r'/gpfs0/arazi/users/amirbenh/Resolving_Power/nexus/SquareFiberDatabaseExpansion2/'

if not os.path.isdir(output_macro_Mfolder):
    os.mkdir(output_macro_Mfolder)

# Open config macro file and read in the data
with open(original_config_macro_path, "r") as f:
    config_template = f.read()
    
# Open init macro file and read in the data
with open(original_init_macro_path, "r") as f:
    init_template = f.read()
    

# Get the Cartesian product of all parameter values
geometry_combinations = list(itertools.product(*geometry_params.values()))

# Iterate through each geometry combination
for i, combination in enumerate(geometry_combinations):

    # Generate output directory based on geometry
    output_macro_geom_folder = os.path.join(output_macro_Mfolder,
                                            f'ELGap={combination[0]}mm_'
                                            f'pitch={combination[1]}mm_'
                                            f'distanceFiberHolder={combination[2]}mm_'
                                            f'distanceAnodeHolder={combination[3]}mm_'
                                            f'holderThickness={combination[4]}mm')
	
    output_macro_geom_folder = os.path.join(output_macro_geom_folder,sub_dir)

    # Create directory if it doesn't exist
    if not os.path.isdir(output_macro_geom_folder):
        os.makedirs(output_macro_geom_folder, exist_ok=True)

    # Update x and y values in run_params for the 'pitch' parameter
    for key, value in zip(geometry_params.keys(), combination):
        if key == 'pitch':
            pitch_value = float(value)
            
            if fixed_intervals:
                # scan unit cell at fixed intervals
                x = np.arange(-pitch_value / 2, pitch_value / 2 + unit_cell_source_spacing,
                              unit_cell_source_spacing)
                y = np.arange(-pitch_value / 2, pitch_value / 2 + unit_cell_source_spacing,
                              unit_cell_source_spacing)
            
            if random_events_xy:
                # randomize events in the unit cell at random places           
                x = list(np.random.uniform(-(pitch_value / 2) + 0.00001,
                                           pitch_value / 2, num_samples))
                y = list(np.random.uniform(-(pitch_value / 2) + 0.00001,
                                           pitch_value / 2, num_samples))
                if len(x)!=len(y):
                    raise ValueError('x and y vectors must be the same legth!')
            

            
            # Update the run parameters
            run_params = {
                'x': [str(val) for val in x],
                'y': [str(val) for val in y],
            }
            break


    if fixed_intervals:
        # Iterate through each combination of x and y
        for x_val in run_params['x']:
            for y_val in run_params['y']:
                
                # fresh copy of the macro config template
                config_macro = config_template
                
                # Replace geometry parameters in the config macro
                for key, value in zip(geometry_params.keys(), combination):
                    config_macro = config_macro.replace('${' + key + '}', value)
                    
                # Replace x, y and seed in the macro
                config_macro = config_macro.replace('${x}', str(x_val))
                config_macro = config_macro.replace('${y}', str(y_val))
                config_macro = config_macro.replace('${seed}', str(seed))
                seed += 1
    
                # Output paths
                output_SiPM_path = os.path.join(output_macro_geom_folder, f'SiPM_hits_x={x_val}mm_y={y_val}mm.txt')
                config_macro = config_macro.replace('${sipmOutputFile_}', output_SiPM_path)
                
                output_TPB_path = os.path.join(output_macro_geom_folder, f'TPB_hits_x={x_val}mm_y={y_val}mm.txt')
                config_macro = config_macro.replace('${tpbOutputFile_}', output_TPB_path)
                
                output_config_macro_path = os.path.join(output_macro_geom_folder, f'x={x_val}mm_y={y_val}mm.config.mac')
                
                # fresh copy of init macro template
                init_macro = init_template
                init_macro = init_macro.replace('${config_macro}', output_config_macro_path)
                 
                output_init_macro_path = os.path.join(output_macro_geom_folder, f'x={x_val}mm_y={y_val}mm.init.mac')
                
                
                # Write the new config macro to a new file
                with open(output_config_macro_path, "w") as f:
                    f.write(config_macro)
                    
                # Write the new init macro to a new file
                with open(output_init_macro_path, "w") as f:
                    f.write(init_macro)
                
                
    if random_events_xy:
        for i in range(len(x)):
            # fresh copy of the macro config template
            config_macro = config_template
            
            # Replace geometry parameters in the config macro
            for key, value in zip(geometry_params.keys(), combination):
                config_macro = config_macro.replace('${' + key + '}', value)
                
            # Replace x, y and seed in the macro
            x_val = x[i]
            y_val = y[i]
            config_macro = config_macro.replace('${x}', str(x_val))
            config_macro = config_macro.replace('${y}', str(y_val))
            config_macro = config_macro.replace('${seed}', str(seed))
            seed += 1
            
            # Output paths
            output_SiPM_path = os.path.join(output_macro_geom_folder, f'SiPM_hits_x={x_val}mm_y={y_val}mm.txt')
            config_macro = config_macro.replace('${sipmOutputFile_}', output_SiPM_path)
            
            output_TPB_path = os.path.join(output_macro_geom_folder, f'TPB_hits_x={x_val}mm_y={y_val}mm.txt')
            config_macro = config_macro.replace('${tpbOutputFile_}', output_TPB_path)
            
            output_config_macro_path = os.path.join(output_macro_geom_folder, f'x={x_val}mm_y={y_val}mm.config.mac')
            
            # fresh copy of init macro template
            init_macro = init_template
            init_macro = init_macro.replace('${config_macro}', output_config_macro_path)
             
            output_init_macro_path = os.path.join(output_macro_geom_folder, f'x={x_val}mm_y={y_val}mm.init.mac')
            
            
            # Write the new config macro to a new file
            with open(output_config_macro_path, "w") as f:
                f.write(config_macro)
                
            # Write the new init macro to a new file
            with open(output_init_macro_path, "w") as f:
                f.write(init_macro)


    print(f'Finished creating ALL geometry macros for path:\n{output_macro_geom_folder}')


