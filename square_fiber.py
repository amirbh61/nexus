#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:22:36 2023

@author: amir
"""

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.style.use('classic')
from tqdm import tqdm
from scipy.interpolate import interp2d as interp2d
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)

# In[0]
# Generate shell session to get to sim folder

# import subprocess
# import pexpect
  
# commands = [
#     "cd Products/geant4/geant4-v11.0.1/",
#     "echo 'Current directory:'",
#     "pwd",
#     "source set_source.sh",
#     "cd MySims/nexus/",
#     "echo 'Current directory:'",
#     "pwd"
# ]

# # Start a shell session
# shell = pexpect.spawn("/bin/bash", timeout=None, encoding='utf-8')

# # Execute the commands one by one in the same shell session
# for command in commands:
#     shell.sendline(command)
#     shell.expect_exact("$")

# # Collect the output
# output = shell.before

# # Close the shell session
# # shell.close()

# print("Output:")
# print(output)

# In[1]

# n_photons = 100000
# geant4_command = f"./build/nexus -b -n {n_photons}" # commant to run sim

# # Define the range of values for the two coordinates
# coordinate_values = range(-5, 6, 1) 

# input_macro_path = "/home/amir/Products/geant4/geant4-v11.0.1/MySims/nexus/macros/SquareOpticalFiber.init.mac"
# # Loop through all combinations of coordinate values
# for x in coordinate_values:
#     for y in coordinate_values:
#         # Read the original macro file
#         with open(input_macro_path, "r") as input_macro_file:
#             input_macro = input_macro_file.read()

#         # Modify the specific line with the new x and y values
#         modified_macro = input_macro.replace(
#             "/Geometry/SquareOpticalFiber/specific_vertex 0 0",
#             f"/Geometry/SquareOpticalFiber/specific_vertex {x} {y}",
#         )

#         # Save the modified macro to a temporary file
#         modified_macro_path = f"modified_macro_({x},{y}).txt"
#         with open(modified_macro_path, "w") as modified_macro_file:
#             modified_macro_file.write(modified_macro)

#         # Run the Geant4 simulation with the modified macro file
#         run_command = f'{geant4_command} "{modified_macro_path}"'
#         subprocess.run(run_command, shell=True)

#         # Optionally, remove the temporary modified macro file
#         os.remove(modified_macro_path)



# In[2]
# load square fiber
import tables as tb
from invisible_cities.io.dst_io import df_writer
from invisible_cities.reco.tbl_functions import filters


def shrink_replace_data_file(filename_in,filename_out):
    
    # Ensure that the directory containing the output file exists
    os.makedirs(os.path.dirname(filename_out), exist_ok=True)
    
    # Read the DataFrame from the HDF5 file
    with pd.HDFStore(filename_in, mode="r") as store:
        df = store.select("/DEBUG/steps")
        
    # Truncate string columns to a maximum length of 256 bytes
    max_str_len = 1024
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].str.slice(0, max_str_len)
    
    # Write the DataFrame to a CSV file
    df.to_csv(filename_out, index=False)
    
    # Optional: Write the DataFrame to a new HDF5 file with compression
    with tb.open_file("output_file.h5", mode="w", filters=filters("ZLIB4")) as file:
        df_writer(file, df, "DEBUG", "steps")
        file.root.DEBUG.steps.colinstances["event_id"].create_index()
        
    # Check file generated is not empty and remove initial LARGE data file
    if os.path.isfile(filename_out) and os.path.getsize(filename_out) > 0:
        print(f"File {filename_out} generated successfully.")
        # os.remove(filename_in)
        # print(f"File {filename_in} deleted successfully.")
       
    return filename_out


# Set the input and output file paths
# filename_in  = '/media/amir/9C33-6BBD/NEXT_work/Geant4/nexus/SquareFiber_big_run.next.h5'
filename_out = "/media/amir/9C33-6BBD/NEXT_work/Geant4/nexus/results_gonzalo_code_big_run.csv"




# filename_out = shrink_replace_data_file(filename_in,filename_out)
df = pd.read_csv(filename_out)


# hits = df.loc[df['final_volume'].str.contains("SiPM_")]
#make sure each row is really a new particle
if df.event_id.duplicated().sum() == 0:
    print("No duplicates in data frame!", end="\n")
unique_primary_photons = df['event_id'].nunique()
print(f'Unique primary photons in dataframe = {unique_primary_photons}/{len(df)}')

max_daughters = df.groupby('event_id')['particle_id'].nunique().max()
print(f'max daughters found = {max_daughters-1}')


# In[3.1]
### Figure for Paula about TPB ###
# make plot of number of primaries vs max number of secondaries
import matplotlib.pyplot as plt
n_daugters = np.arange(2,max_daughters,1)

daughters_df = []
for i in range(2,max_daughters):
    daughters_df.append(df[df['particle_id'] == i])

counts = [daughters_df[i]["event_id"].nunique() for i in range(len(daughters_df))]
                                  
plt.plot(n_daugters-np.ones(len(n_daugters)),np.divide(counts,unique_primary_photons),'-*k')
plt.xticks(n_daugters-np.ones(len(n_daugters)))
plt.xlabel("Max number of secondary photons")
plt.ylabel("Fraction of primary photons")
plt.title("Breakdown of primary vs max secondary photons")

grouped = df.groupby('event_id')["particle_id"].max()
did_WLS_number = sum(grouped > 1)
did_not_WLS_number = sum(grouped == 1)


did_not_WLS_Fraction = did_not_WLS_number/unique_primary_photons
legend = f'Fraction that did not produce secondaries={round(did_not_WLS_Fraction,3)}'
plt.text(2.5, 0.42, legend, bbox=dict(facecolor='blue', alpha=0.5))
plt.grid()
plt.show()

print(f'recorded WLS efficiency = {round(did_WLS_number/unique_primary_photons,3)}')


# In[3.2]
### Figure for Lior , with TPB###
# TPB, plot of the absorbed WLS blue photons in sipm vs fiber coating reflectivity

n_UV = 100000
# Note: in reality only 1K photons for 100% reflectivity
vikuiti_ref = [96, 97, 98, 99, 100]
# SiPM_hits = [5699, 7841, 11488, 19013, 56640] # results from point source (0,0,-1.1um)
SiPM_hits = [6006, 8092, 11806, 19365, 55300] # Randomly on face
SiPM_hits = [x / n_UV for x in SiPM_hits]
plt.plot(vikuiti_ref, SiPM_hits, '-^', color='rebeccapurple')
text = "100K UV photons at 7.21[eV] per sample\n" + \
       "Randomly generated in TPB center, facing forward\n" + \
       "1K UV photons for 100% reflectivity due to runtime"
plt.text(96, 0.52, text, bbox=dict(facecolor='rebeccapurple', alpha=0.5))
plt.xlabel("Fiber Coating Reflectivity [%]")
plt.ylabel("Fraction of photons absorbed in SiPM")
plt.title("WLS blue photons absorbed in SiPM vs fiber coating reflectivity")
plt.grid()
plt.xlim(min(vikuiti_ref) - 0.1, max(vikuiti_ref) + 0.1)
plt.gca().ticklabel_format(style='plain', useOffset=False)
save_path = r'/home/amir/Desktop/Sipm_hits_vs_coating_reflectivity_TPB.jpg'
plt.savefig(save_path, dpi=600)
plt.show()
# In[3.3]
### Figure for Lior , with TPB###
# No TPB, plot of the absorbed WLS blue photons in sipm vs fiber coating reflectivity

n_UV = 100000
# Note: in reality only 1K photons for 100% reflectivity
vikuiti_ref = [96, 97, 98, 99, 100]

SiPM_hits = [ 15294, 20062, 27828, 42391, 95800]
SiPM_hits = [x / n_UV for x in SiPM_hits]
plt.plot(vikuiti_ref, SiPM_hits, '-^b')
text = "100K blue photons at 2.883[eV] per sample\n" + \
       "Randomly generated on face of fiber, facing forward, no TPB\n" + \
       "1K blue photons for 100% reflectivity due to runtime"
plt.text(96, 0.88, text, bbox=dict(facecolor='blue', alpha=0.5))
plt.xlabel("Fiber Coating Reflectivity [%]")
plt.ylabel("Fraction of photons absorbed in SiPM")
plt.title("WLS blue photons absorbed in SiPM vs fiber coating reflectivity")
plt.grid()
plt.xlim(min(vikuiti_ref) - 0.1, max(vikuiti_ref) + 0.1)
plt.gca().ticklabel_format(style='plain', useOffset=False)
save_path = r'/home/amir/Desktop/Sipm_hits_vs_coating_reflectivity_no_TPB.jpg'
plt.savefig(save_path, dpi=600)
plt.show()





# In[4]

n_particles = 462500
edge = 75 #Create a 2*edge by 2*edge matrix
# x_hitpoints = df["final_x"]
# y_hitpoints = df["final_y"]

SiPM_hit_path = r'/media/amir/9C33-6BBD/NEXT_work/Geant4/nexus/SiPM_hits.txt'
hit_points = np.genfromtxt(SiPM_hit_path)
x_hitpoints = hit_points[:,0]
y_hitpoints = hit_points[:,1]
print('\n' + f'percent SiPM hits = {(len(hit_points)/(n_particles)) * 100}')



bins = 2*edge
SR = np.zeros((bins,bins))
SR, x_hist, y_hist = np.histogram2d(x_hitpoints, y_hitpoints,
                     range=[[-edge, edge], [-edge, edge]],bins=bins)
plt.imshow(SR)
plt.xlabel('x [mm]')
plt.ylabel('y [mm]')
plt.colorbar()
plt.show()

# In[5]
# import os
# import pandas as pd
# import tables as tb
# from invisible_cities.io  .dst_io        import df_writer
# from invisible_cities.reco.tbl_functions import filters


# import pandas as pd
# import tables as tb
# from invisible_cities.io.dst_io import df_writer
# from invisible_cities.reco.tbl_functions import filters

# # Set the input and output file paths
# filename_in  = path
# filename_out = "/media/amir/9C33-6BBD/NEXT_work/Geant4/nexus/results_gonzalo_code.csv"

# # Ensure that the directory containing the output file exists
# os.makedirs(os.path.dirname(filename_out), exist_ok=True)

# # Read the DataFrame from the HDF5 file
# with pd.HDFStore(filename_in, mode="r") as store:
#     df = store.select("/DEBUG/steps")

# # Write the DataFrame to a CSV file
# df.to_csv(filename_out, index=False)

# # Optional: Write the DataFrame to a new HDF5 file with compression
# with tb.open_file("output_file.h5", mode="w", filters=filters("ZLIB4")) as file:
#     df_writer(file, df, "DEBUG", "steps")
#     file.root.DEBUG.steps.colinstances["event_id"].create_index()

# # if os.path.exists(filename_in):
# #     os.remove(filename_in)
# #     print(f"File {filename_in} deleted successfully.")
    
# out_data = pd.read_csv(filename_out)

# In[6]
# checks if photons hit SiPMs
SiPM_hits = 0
back_fired = 0
n_particles = df['event_id'].max()
edge = 100 #Create a 2*edge by 2*edge matrix
x_final = []
y_final = []
z_final = []
for row in tqdm(range(len(df)-1)):
    cell = df.iloc[row,5]
    if df.iloc[row,0] != df.iloc[row+1,0]:
        x = df.iloc[row,-3]
        y = df.iloc[row,-2]
        if cell.startswith(("SiPM_")):
            x_final.append(df.iloc[row,-3])
            y_final.append(df.iloc[row,-2])
            SiPM_hits += 1
        if cell.startswith(("WORLD")):
            back_fired+=1

print('\n' + f'percent SiPM hits = {(SiPM_hits/(n_particles+1)) * 100}')
print(f'Back fired = {back_fired}')



# # Filter for SiPM and World cells only
# df_filtered = df[df['cell_name'].str.startswith(('SiPM_', 'World_'))]

# # Group by event ID and cell name
# grouped = df_filtered.groupby(['event_id', 'cell_name'])

# # Get the last coordinates of each cell for each event
# last_coords = grouped.tail(1)[['x', 'y', 'z']].reset_index(drop=True)

# # Filter for SiPM hits
# SiPM_hits = last_coords[last_coords['cell_name'].str.startswith('SiPM_')]

# # Filter for back-fired photons
# back_fired = last_coords[last_coords['cell_name'].str.startswith('World_')]

# # Count the number of events and calculate the percentage of SiPM hits
# n_particles = df['event_id'].nunique()
# percent_SiPM_hits = len(SiPM_hits) / n_particles * 100

# # Print the results
# print(f"Percent SiPM hits = {percent_SiPM_hits:.2f}")
# print(f"Back fired = {len(back_fired)}")



# In[2]
# Sensor response
bins = 2*edge
SR = np.zeros((bins,bins))
SR, x_hist, y_hist = np.histogram2d(x_final, y_final,
                     range=[[-edge, edge], [-edge, edge]],bins=bins)
plt.imshow(SR)
plt.xlabel('x [mm]')
plt.ylabel('y [mm]')
plt.colorbar()
plt.show()





# path_to_save ='/home/amir/Desktop/hit_map_right_off_center.npy'
# np.save(path_to_save,SR)
# # interpolate

# interp = interp2d(x_hist[:-1], y_hist[:-1], SR, 'cubic')
# z = interp(x_final, y_final)
# plt.imshow(z)
# plt.show()


# here we will save the PSF

# In[5]
# left = np.load('/home/amir/Desktop/hit_map_left_off_center.npy')
# right = np.load('/home/amir/Desktop/hit_map_right_off_center.npy')
# combined = left+right
# plt.imshow(combined)
# plt.xlabel('x [mm]')
# plt.ylabel('y [mm]')
# plt.colorbar()
# plt.show()

# path_to_save ='/home/amir/Desktop/hit_map_combined.npy'
# np.save(path_to_save,combined)

# In[6]






# In[3]
#plot results for lior 8.3.2023
d2 = [-5, 0, 2.5, 4, 5] - 5*np.ones(5)
photons_hit_SiPM = [0.4394, 0.5984, 0.8173, 1.0218, 1.4175]
title = "5 mm distance between EL gap and holder, line source, 100K ph, 2pi, TPB roughness = 0.01"


plt.plot(d2,photons_hit_SiPM,"*-b")
plt.title(title,fontdict={'fontsize': 10})
plt.xticks(np.arange(d2.min(), d2.max()+1, 1))
plt.xlabel("Distance from fiber entrance to hole entrance [mm]")
plt.ylabel("photons hitting the SiPMs [%]")
plt.grid()
plt.figure(dpi=600)
plt.show()





























