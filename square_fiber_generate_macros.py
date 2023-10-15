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

'''
Option to create events in a circular pattern so each octant is symmetrical.

'''

# import numpy as np
# import matplotlib.pyplot as plt

# def points_on_circle_octant(center, radius):
#     """Generate points on 1/8th of a circle ensuring intersections with axes, diagonals, and 15 and 30-degree lines."""
#     # Angles corresponding to x-axis, 15-degree from x-axis, 30-degree from x-axis, y=x, 
#     # 30-degree from y-axis, 15-degree from y-axis, and y-axis intersections
#     t = [0, np.pi/12, np.pi/6, np.pi/4, np.pi/3, 5*np.pi/12, np.pi/2]
#     x = center[0] + radius * np.cos(t)
#     y = center[1] + radius * np.sin(t)
#     return x, y

# def reflect_points(x, y):
#     """Reflect points across all the symmetry lines of a circle."""
#     points = [(x, y), (x, -y), (-x, y), (-x, -y),
#               (y, x), (y, -x), (-y, x), (-y, -x)]
#     return points

# def is_inside_square(point, square_size):
#     """Check if a point is inside a square centered at the origin."""
#     half_size = square_size / 2.0
#     return -half_size <= point[0] <= half_size and -half_size <= point[1] <= half_size

# # Parameters
# square_size = 10.0  # Side length of the square
# center = (0, 0)  # Center of the square and circles
# radii = np.arange(0, np.sqrt(2)*square_size/2, 0.5)  # Radii of circles increase by 0.5mm

# # Plot
# plt.figure(figsize=(10, 10))
# plt.plot([-square_size/2, square_size/2, square_size/2, -square_size/2, -square_size/2], 
#          [-square_size/2, -square_size/2, square_size/2, square_size/2, -square_size/2], 'g-', label="Square")

# # Adding the origin
# plt.plot(0, 0, 'bo')

# for radius in radii[1:]:
#     x, y = points_on_circle_octant(center, radius)
    
#     # Plot the circle itself
#     circle = plt.Circle(center, radius, color='r', fill=False, linestyle='--')
#     plt.gca().add_patch(circle)
    
#     for px, py in zip(x, y):
#         for ref_pt in reflect_points(px, py):
#             if is_inside_square(ref_pt, square_size):
#                 plt.plot(ref_pt[0], ref_pt[1], 'bo')

# plt.gca().set_aspect('equal', adjustable='box')
# plt.xlabel('X (mm)')
# plt.ylabel('Y (mm)')
# plt.title('Symmetric Points in Circles Inside a Square')
# plt.legend()
# plt.grid(True)
# plt.show()



# '''
# Show square grid with octant of symmetry
# '''

# # Create the square points
# x = np.arange(-5, 25, 1)
# y = np.arange(-5, 25, 1)

# X, Y = np.meshgrid(x, y)

# # Plotting
# fig, ax = plt.subplots()

# # Plot the points
# ax.scatter(X, Y, color='blue', marker='o',s=1.5)

# # Highlight the boundaries of the octant (x >= 0 and y <= x)
# ax.plot([5, 25], [5, 25], alpha=0.5, color='black', linewidth=2) # y=x line
# ax.plot([5, 5], [5, 25], alpha=0.5, color='black', linewidth=2) # Vertical line splitting the octant

# ax.plot([0, 0], [10, 0], alpha=0.5, color='black', linewidth=2) # Vertical line splitting the octant
# ax.plot([10, 0], [10, 10], alpha=0.5, color='black', linewidth=2) # Vertical line splitting the octant
# ax.plot([10, 10], [10, 0], alpha=0.5, color='black', linewidth=2) # Vertical line splitting the octant
# ax.plot([10, 0], [0, 0], alpha=0.5, color='black', linewidth=2) # Vertical line splitting the octant



# # Set the limits and display
# # ax.set_xlim(-1, 11)
# # ax.set_ylim(-1, 11)
# ax.set_aspect('equal')
# plt.show()




# import numpy as np
# import matplotlib.pyplot as plt

# '''
# Show square grid with octant of symmetry
# '''

# # Create the square points
# x = np.arange(-5, 25, 1)
# y = np.arange(-5, 25, 1)

# X, Y = np.meshgrid(x, y)

# # Plotting
# fig, ax = plt.subplots()

# # Plot the points
# ax.scatter(X, Y, color='blue', marker='o', s=1.5)

# # Highlight the boundaries of the octant
# ax.plot([5, 25], [5, 25], alpha=0.5, color='black', linewidth=2) # y=x line
# ax.plot([5, 5], [5, 25], alpha=0.5, color='black', linewidth=2) # Vertical line splitting the octant
# ax.plot([0, 0], [10, 0], alpha=0.5, color='black', linewidth=2)
# ax.plot([10, 0], [10, 10], alpha=0.5, color='black', linewidth=2)
# ax.plot([10, 10], [10, 0], alpha=0.5, color='black', linewidth=2)
# ax.plot([10, 0], [0, 0], alpha=0.5, color='black', linewidth=2)

# # Draw concentric circles inside the octant
# radii = np.arange(1, 20, 1)
# theta = np.linspace(np.pi/4, np.pi/2, 1000)

# for r in radii:
#     x_circle = r * np.cos(theta) + 5
#     y_circle = r * np.sin(theta) + 5
    
#     # Mask to ensure circle segments are inside the octant
#     mask = np.logical_and(y_circle >= x_circle, x_circle >= 5)
#     ax.plot(x_circle[mask], y_circle[mask], color='black', alpha=0.5, linewidth=1)

# # Set square aspect ratio
# ax.set_aspect('equal')
# plt.show()







# In[1]
# generate new geant4 macros for all geometries and all possible runs

# # Geometry parameters, FULL
# geometry_params = {
#     'ELGap': ['1', '10'],
#     'pitch': ['5', '10', '15.6'],
#     'distanceFiberHolder': ['-1', '2', '5'],
#     'distanceAnodeHolder': ['2.5', '5', '10'],
#     'holderThickness': ['5','10'],
#     'TPBThickness': ['2.2'] # microns
# }


# # Geometry parameters - TEST CLUSTER , REMOVE AFTER
# geometry_params = {
#     'ELGap': ['10'],
#     'pitch': ['5'],
#     'distanceFiberHolder': ['2'],
#     'distanceAnodeHolder': ['2.5', '10'],
#     'holderThickness': ['10'],
#     'TPBThickness': ['2.2'] # microns
# }

# Geometry parameters - single geometry , REMOVE AFTER
geometry_params = {
    'ELGap': ['10'],
    'pitch': ['10'],
    'distanceFiberHolder': ['-1'],
    'distanceAnodeHolder': ['2.5'],
    'holderThickness': ['10'],
    'TPBThickness': ['2.2'] # microns
}

# Run parameters
run_params = {
    'x': ['0'],
    'y': ['0'],
    'z': ['0'],
}

# Chose mode of source generation
fixed_intervals = False # edges are included !
if fixed_intervals:
    unit_cell_source_spacing = 0.5 # mm, spacing between sources in different runs
    
random_events_xy = True
if random_events_xy:
    num_samples = 100


z = 0 # junk number, value has no meaning but must exist as input for geant
seed = 10000

original_config_macro_path = r'/home/amir/Products/geant4/geant4-v11.0.1/MySims/nexus/macros/SquareOpticalFiberCluster.config.mac'
original_init_macro_path = r'/home/amir/Products/geant4/geant4-v11.0.1/MySims/nexus/macros/SquareOpticalFiberCluster.init.mac'
output_macro_Mfolder = r'/home/amir/Products/geant4/geant4-v11.0.1/MySims/nexus/SquareFiberMacrosAndOutputsRandomFaceGen/'

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

    # Create directory if it doesn't exist
    if not os.path.isdir(output_macro_geom_folder):
        os.mkdir(output_macro_geom_folder)

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
                print(f'create geometry macro:\n{output_macro_geom_folder}')
                
                
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
            print(f'create geometry macro:\n{output_macro_geom_folder}')
            
    

