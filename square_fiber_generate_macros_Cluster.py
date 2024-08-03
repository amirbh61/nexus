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


'''
This script generates new .mac files for Geant4 for all geometries and in all runs.
'''


# Choose geometry parameters - script will create .mac file combinations of these parameters
# Geometry parameters
geometry_params = {
     'ELGap': ['1', '10'],
     'pitch': ['5', '10', '15.6'],
     'distanceFiberHolder': ['-1', '2', '5'],
     'distanceAnodeHolder': ['2.5', '5', '10'],
     'holderThickness': ['10'],
     'TPBThickness': ['2.2'] # microns, FULL thickness (in contrast to half size generally used in geant4 examples)
}


# line source x,y,z
run_params = {
    'x': ['0'],
    'y': ['0'],
    'z': ['0'],
}

### IMPORTANT : BOTH MODES ARE NEEDED FOR EACH GEOMETRY !! ###

'''
Choose mode of source generation.
Explanation:

to run our simulation we need both line source events and PSF events for resolving power study.
The line source events are currently set to be taken at a fixed intervals of the central unit cell "unit_cell_source_spacing".

for PSF generation, we use a different approach, in which we generate .mac files for line source events at random x,y positions. the number of positions it the "num_samples" under the below PSF section. the number of photons per individual run is set in the job.sh file.
NOTE: I have noticed it is generally more efficient to generate more macros with less photons per macro than to generate less macros with more photons per macro.
so for example, 10K macros with 10K photons per macro gives a better PSF performance than 1K macros with 100K photons each. Ideally, one would generate a 1M macros with a single photon each at random x,y , however the huge amount of macros and the inefficincy of loading the entire geometry for a single photon for each core worker will be sub optimal and wasteful. Also, in the current BGU HPC there is a 3M files limit for each user.
Therefore, we need to compromise somewhere in between.

For both, the parameter "sub_dir" specifies the folder name for these files.

At the current state of the code, you need to choose the method to generate .mac files using the below 'x_y_dist_method' parameter.
We use 'fixed_intervals' for creating the line sources, later to be joind to create the twin events. 
We use 'random_events_xy' for creating files for the PSF.

'''


# x,y distribution method
x_y_dist_method = 'fixed_intervals' # or 'random_events_xy'

if x_y_dist_method == 'fixed_intervals':
    unit_cell_source_spacing = 0.2 # in mm, spacing between consecutive sources
    sub_dir = r'Geant4_Kr_events'
    
if x_y_dist_method == 'random_events_xy':
    num_samples = 10000 # how many random x,y positions to take for PSF generation
    sub_dir = 'Geant4_PSF_events'


seed = 10000 # initaial seed. Each .mac will have seed=seed+1 with "seed" being a tunning counter.

template_config_macro_path = r'/gpfs0/arazi/users/amirbenh/Resolving_Power/nexus/macros/SquareOpticalFiberCluster.config.mac'
template_init_macro_path = r'/gpfs0/arazi/users/amirbenh/Resolving_Power/nexus/macros/SquareOpticalFiberCluster.init.mac'
output_macro_Mfolder = r'/gpfs0/arazi/users/amirbenh/Resolving_Power/nexus/SquareFiberDatabaseExpansion2/' # main dataset folder path



if not os.path.isdir(output_macro_Mfolder):
    os.mkdir(output_macro_Mfolder)

# Open config macro file and read in the data
with open(template_config_macro_path, "r") as f:
    config_template = f.read()
    
# Open init macro file and read in the data
with open(template_init_macro_path, "r") as f:
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
            
            if x_y_dist_method == 'fixed_intervals':
                # scan unit cell at fixed intervals
                x = np.arange(-pitch_value / 2, pitch_value / 2 + unit_cell_source_spacing,
                              unit_cell_source_spacing)
                y = np.arange(-pitch_value / 2, pitch_value / 2 + unit_cell_source_spacing,
                              unit_cell_source_spacing)
            
            if x_y_dist_method == 'random_events_xy':
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


    if x_y_dist_method == 'fixed_intervals':
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
                
                
    if x_y_dist_method == 'random_events_xy':
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


