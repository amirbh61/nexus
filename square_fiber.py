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
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
import glob
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage
import time
from scipy.signal           import fftconvolve
from scipy.signal           import convolve
from invisible_cities.reco.deconv_functions     import richardson_lucy
from scipy.interpolate import interp2d
from scipy.interpolate import griddata
from scipy.interpolate import RectBivariateSpline
from scipy.signal import find_peaks, peak_widths
import random
# Global settings #


n_sipms = 25 # DO NOT CHANGE THIS VALUE
n_sipms_per_side = (n_sipms-1)/2
size = 100
bins = 100


path_to_dataset = '/media/amir/Extreme Pro/SquareFiberDatabase'

# List full paths of the Geant4_PSF_events folders inside SquareFiberDatabase
geometry_dirs = [os.path.join(path_to_dataset, d) for d in os.listdir(path_to_dataset)
                 if os.path.isdir(os.path.join(path_to_dataset, d))]


# In[0]
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


# In[0.1]
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


# In[0.2]
### Figure for Lior , with TPB ###
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
# In[0.3]
### Figure for Lior , without TPB ###
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





# In[1]
# Create PSFs

def smooth_PSF(PSF):
    '''
    Smoothes the PSF matrix based on the fact the ideal PSF should have radial
    symmetry.
    Receives:
        PSF: ndarray, the PSF 2d array
    Returns:
        smooth_PSF: ndarray, the smoothed PSF 2d array
    '''

    x_cm, y_cm = ndimage.measurements.center_of_mass(PSF)
    # x_cm, y_cm = int(x_cm), int(y_cm)
    psf_size = PSF.shape[0]
    x,y = np.indices((psf_size,psf_size))
    smooth_PSF = np.zeros((psf_size,psf_size))
    r = np.arange(0,(1/np.sqrt(2))*psf_size,1)
    for radius in r:
        R = np.full((1, psf_size), radius)
        circle = (np.abs(np.hypot(x-x_cm, y-y_cm)-R) <= 0.6).astype(int)
        circle_ij = np.where(circle==1)
        smooth_PSF[circle_ij] = np.mean(PSF[circle_ij])
    return smooth_PSF


# #original
# def assign_hit_to_SiPM_original(hit, pitch, n):
#     """
#     Assign a hit to a SiPM based on its coordinates.
    
#     Args:
#     - hit (tuple): The (x, y) coordinates of the hit.
#     - pitch (float): The spacing between SiPMs.
#     - n (int): The number of SiPMs on one side of the square grid.
    
#     Returns:
#     - (int, int): The assigned SiPM coordinates.
#     """
    
#     half_grid_length = (n-1) * pitch / 2

#     x, y = hit

#     # First, check the central SiPM
#     for i in [0, -pitch, pitch]:
#         for j in [0, -pitch, pitch]:
#             if -pitch/2 <= x - i < pitch/2 and -pitch/2 <= y - j < pitch/2:
#                 return (i, j)

#     # If not found in the central SiPM, search the rest of the grid
#     for i in np.linspace(-half_grid_length, half_grid_length, n):
#         for j in np.linspace(-half_grid_length, half_grid_length, n):
#             if abs(i) > pitch or abs(j) > pitch:  # Skip the previously checked SiPMs
#                 if i - pitch/2 <= x < i + pitch/2 and j - pitch/2 <= y < j + pitch/2:
#                     return (i, j)
    
#     # Return None if hit doesn't belong to any SiPM
#     return None


def assign_hit_to_SiPM(hit, pitch, n):
    half_grid_length = (n-1) * pitch / 2
    x, y = hit

    # Direct calculation to find the nearest grid point
    nearest_x = round((x + half_grid_length) / pitch) * pitch - half_grid_length
    nearest_y = round((y + half_grid_length) / pitch) * pitch - half_grid_length

    # Check if the hit is within the bounds of the SiPM
    if -half_grid_length <= nearest_x <= half_grid_length and -half_grid_length <= nearest_y <= half_grid_length:
        return (np.around(nearest_x,1), np.around(nearest_y,1))
    else:
        return None





def psf_creator(directory, create_from, to_plot=True,to_smooth=True):
    '''
    Gives a picture of the SiPM (sensor wall) response to an event involving 
    the emission of light in the EL gap.
    Receives:
        directory: str, path to geometry directory
        create_from: str, surface to create PSF from. Either "SiPM" or "TPB".
        to_plot: bool, flag - if True, plots the PSF
        to_smooth: bool, flag - if True, smoothes the PSF
    Returns:
        PSF: ndarray, the Point Spread Function
    '''
    
    os.chdir(directory)
    working_dir = r'Working on directory:'+f'\n{os.getcwd()}'
    print(working_dir)
    
    files = glob.glob(f'{create_from}*')
    
    # pattern to extract x,y values of each event from file name
    pattern = r"-?\d+.\d+"

    PSF_list = []
    size = 100 # keep the same for all future histograms
    bins = 100 # keep the same for all future histograms
    
    # For DEBUGGING
    plot_event = False
    plot_sipm_assigned_event = False
    plot_shifted_event = False
    plot_accomulated_events = False
    
    
    # Search for the pitch value pattern
    match = re.search(r"_pitch=(\d+(?:\.\d+)?)mm", working_dir)
    pitch = float(match.group(1))

    ### assign each hit to its corresponding SiPM ###
    for filename in tqdm(files):

        # Check if file is empty and skip if it is
        if os.path.getsize(filename) == 0:
            continue
        
        # Load hitmap from file
        hitmap = np.genfromtxt(filename)
        
        # Check if hitmap is empty
        if hitmap.size == 0:
            continue
        
        # If hitmap has a single line, it's considered a 1D array
        if len(hitmap.shape) == 1:
            hitmap = np.array([hitmap[0:2]])  # Convert to 2D array with single row
        else:
            # Multiple lines in hitmap
            hitmap = hitmap[:, 0:2]

        
        # Store x,y values of event
        matches = re.findall(pattern, filename)
        x_event = float(matches[0])
        y_event = float(matches[1])
        
        if plot_event:
            single_event, x_hist, y_hist = np.histogram2d(hitmap[:,0], hitmap[:,1],
                                                  range=[[-size/2,size/2],[-size/2,size/2]],
                                                  bins=bins)
            plt.imshow(single_event,extent=[-size/2, size/2, -size/2, size/2])
            plt.title("Single Geant4 event")
            plt.xlabel('x [mm]');
            plt.ylabel('y [mm]');
            plt.colorbar()
            plt.show()
        
        # Assign each hit to a SiPM before shifting the hitmap
        new_hitmap = []
        for hit in hitmap:
            sipm = assign_hit_to_SiPM(hit=hit, pitch=pitch, n=n_sipms)
            if sipm:  # if the hit belongs to a SiPM
                new_hitmap.append(sipm)
       
        new_hitmap = np.array(new_hitmap)
        
        
        if plot_sipm_assigned_event:
            plot_sensor_response(new_hitmap, bins, size)
        
        
        
        # Now, shift each event to center
        shifted_hitmap = new_hitmap - [x_event, y_event]
    
        
        if plot_shifted_event:
            shifted_event, x_hist, y_hist = np.histogram2d(shifted_hitmap[:,0], shifted_hitmap[:,1],
                                                  range=[[-size/2,size/2],[-size/2,size/2]],
                                                  bins=bins)     
            plt.imshow(shifted_event,extent=[-size/2, size/2, -size/2, size/2])
            plt.title("Shifted Geant4 event, after assigned to SiPMs")
            plt.xlabel('x [mm]');
            plt.ylabel('y [mm]');
            plt.colorbar()
            plt.show()
        
        
        PSF_list.append(shifted_hitmap)
        
        if plot_accomulated_events:
            PSF = np.vstack(PSF_list)
            PSF, x_hist, y_hist = np.histogram2d(PSF[:,0], PSF[:,1],
                                                  range=[[-size/2,size/2],[-size/2,size/2]],
                                                  bins=bins)
            plt.imshow(PSF,extent=[-size/2, size/2, -size/2, size/2])
            plt.title("Accumulated Geant4 events, after assigned to SiPMs and shifted")
            plt.xlabel('x [mm]');
            plt.ylabel('y [mm]');
            plt.colorbar()
            plt.show()
        
        

    # Concatenate ALL shifted hitmaps into a single array
    PSF = np.vstack(PSF_list)
    PSF, x_hist, y_hist = np.histogram2d(PSF[:,0], PSF[:,1],
                                         range=[[-size/2,size/2],[-size/2,size/2]],
                                         bins=bins)
    

    if to_smooth:
        #Smoothes the PSF
        PSF = smooth_PSF(PSF)
        
    if to_plot:        
        _ = plot_PSF(PSF=PSF,size=size)
        
    return PSF


def plot_sensor_response(event, bins, size, noise=False):
    sipm_assigned_event, x_hist, y_hist = np.histogram2d(event[:,0], event[:,1],
                                                         range=[[-size/2, size/2], [-size/2, size/2]],
                                                         bins=bins)  
    if noise:
        sipm_assigned_event = np.random.poisson(sipm_assigned_event)

    # Transpose the array to correctly align the axes
    sipm_assigned_event = sipm_assigned_event.T

    plt.imshow(sipm_assigned_event,
               extent=[-size/2, size/2, -size/2, size/2], vmin=0,origin='lower')
    plt.title("Sensor Response")
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    plt.colorbar(label='Photon hits')
    plt.show()


def plot_PSF(PSF,size=100):
    total_TPB_photon_hits = int(np.sum(PSF))
    
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16.5,8), dpi=600)
    title = f'{total_TPB_photon_hits}/100M TPB hits PSF, current geometry:' + f'\n{os.path.basename(os.getcwd())}'
    fig.suptitle(title, fontsize=15)
    im = ax0.imshow(PSF, extent=[-size/2, size/2, -size/2, size/2])
    ax0.set_xlabel('x [mm]');
    ax0.set_ylabel('y [mm]');
    ax0.set_title('PSF image')
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    y = PSF[int(size/2),:]
    peaks, _ = find_peaks(y)
    fwhm = np.max(peak_widths(y, peaks, rel_height=0.5)[0])
    ax1.plot(np.arange(-size/2,size/2,1), y, linewidth=2) #normalize
    ax1.set_xlabel('mm')
    ax1.set_ylabel('Charge')
    ax1.set_title('Charge profile')
    ax1.grid(linewidth=1)
    fwhm_text = f"FWHM = {fwhm:.3f}"  # format to have 3 decimal places
    ax1.text(0.95, 0.95, fwhm_text, transform=ax1.transAxes, 
             verticalalignment='top', horizontalalignment='right', 
             color='red', fontsize=12, fontweight='bold',
             bbox=dict(facecolor='white', edgecolor='red',
                       boxstyle='round,pad=0.5'))

    fig.tight_layout()
    plt.show()
    return fig



# # test assign_hit_to_SiPM
# test_cases = [
#     ((x, y), pitch, n)
#     for x in np.linspace(-10, 10, 20)
#     for y in np.linspace(-10, 10, 20)
#     for pitch in [1, 2, 3]
#     for n in [25]
# ]

# # Compare the outputs of the two functions
# for hit, pitch, n in test_cases:
#     result_original = assign_hit_to_SiPM_original(hit, pitch, n)
#     result_optimized = assign_hit_to_SiPM_optimized(hit, pitch, n)

#     if result_original != result_optimized:
#         print(f"Discrepancy found for hit {hit}, pitch {pitch}, n {n}:")
#         print(f"  Original: {result_original}, Optimized: {result_optimized}")

# # If no output, then the two functions are consistent for the test cases
# print("Test completed.")

# In[2]

# Generate PSF and save
path_to_dataset = r'/media/amir/9C33-6BBD/NEXT_work/Geant4/nexus/' + \
                   r'small_cluster_hitpoints_dataset/SquareFiberMacrosAndOutputsCloseEL10KSources'
                  
# path_to_dataset = r'/media/amir/Extreme Pro/SquareFiberDatabase'
geometry_dirs = os.listdir(path_to_dataset)
size = 100
bins = 100
MC_folder = '/home/amir/Desktop/NEXT_work/Resolving_Power/Results/'
plot_MC_PSF = True

# maybe remove
# Load theoretical MC PSF
# anode_track_gap = 2.5
# el_gap = 10
# # Enter parameters manually
# anode_track_gap = float(input("Please enter the value for anode_track_gap (in mm): "))
# el_gap = float(input("Please enter the value for el_gap (in mm): "))

# # Output the entered values to confirm
# print(f"Value entered for anode_track_gap: {anode_track_gap} mm")
# print(f"Value entered for el_gap: {el_gap} mm")



# create full paths
for i in range(len(geometry_dirs)):
    geometry_dirs[i] = os.path.join(path_to_dataset, geometry_dirs[i])
    
# Generate a PSF for each geometry   
for dir in geometry_dirs:
    
    # Search for the geometry parameter patterns in dir name
    match = re.search(r"ELGap=(\d+(?:\.\d+)?)mm", dir)
    el_gap = float(match.group(1))
    match = re.search(r"distanceAnodeHolder=(\d+(?:\.\d+)?)mm", dir)
    anode_track_gap = float(match.group(1))
    match = re.search(r"_pitch=(\d+(?:\.\d+)?)mm", dir)
    pitch = float(match.group(1))
    
    # Load generic Monte Carlo PSF
    folder = MC_folder + f'Resolving_Power_EL_gap{el_gap}mm_Tracking_Plane_Gap{anode_track_gap}mm/'
    Save_PSF  = f'{folder}PSF/'  
    psf_file_name = 'PSF_matrix'
    evt_PSF_output = Save_PSF + psf_file_name + '.npy'
    MC_PSF = np.load(evt_PSF_output)
    if plot_MC_PSF:
        plt.imshow(MC_PSF)
        plt.colorbar()
        plt.title(f"Generic MC PSF EL={el_gap}, anode distance={anode_track_gap}, 10M photons")
        plt.show()
    
    
    
    
    # PSF_SiPM = psf_creator(dir,create_from="SiPM",to_plot=True,to_smooth=True)
    PSF_TPB = psf_creator(dir,create_from="TPB",to_plot=True,to_smooth=True)
     
    # # save PSFs
    # save_PSF_SiPM = r'PSF_SiPM.npy'
    # np.save(save_PSF_SiPM,PSF_SiPM)
    # save_PSF_TPB = r'PSF_TPB.npy'
    # np.save(save_PSF_TPB,PSF_TPB)
    
    
    #### Show TPB hits vs theoretical monte carlo ####
    
    MC_y = MC_PSF[int(size/2),:]
    
    # plot
    PSF = PSF_TPB
    # PSF = PSF_SiPM
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16.5,8))
    fig.suptitle(r'Fiber TPB PSF, 100M photons fired forward', fontsize=15)
    im = ax0.imshow(PSF, extent=[-size/2, size/2, -size/2, size/2])
    ax0.set_xlabel('x [mm]')
    ax0.set_ylabel('y [mm]')
    ax0.set_title('PSF image')
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    y = PSF[int(size/2),:]
    peaks, _ = find_peaks(y)
    fwhm = np.max(peak_widths(y, peaks, rel_height=0.5)[0])

    # Adding labels and colors to the plots
    ax1.plot(np.arange(-size/2,size/2,1), y/np.max(y), 
             linewidth=2, color='blue', label='Geant4 TPB hits')  # normalize cross section and set color to blue
    ax1.plot(np.arange(-size/2,size/2,1), MC_y/np.max(MC_y), 
             linewidth=2, color='green', label='MC hits')  # normalize cross section and set color to red

    ax1.set_xlabel('mm')
    ax1.set_ylabel('Charge')
    ax1.set_title('Charge profile')
    ax1.grid(linewidth=1)
    fwhm_text = f"FWHM = {fwhm:.3f}"  # format to have 3 decimal places
    ax1.text(0.95, 0.95, fwhm_text, transform=ax1.transAxes, 
              verticalalignment='top', horizontalalignment='right', 
              color='red', fontsize=12, fontweight='bold',
              bbox=dict(facecolor='white', edgecolor='red',
                        boxstyle='round,pad=0.5'))

    ax1.legend(loc='upper left')  # Display the legend
    ax1.set_ylim([0,1.1])
    fig.tight_layout()
    plt.show()
    
    print('\n')
    print(f'area TPB = {np.trapz(y/np.sum(PSF))}')
    print(f'area MC = {np.trapz(MC_y/np.sum(MC_PSF))}',end='\n\n\n')


    # # plot around center unit cell
    # plt.imshow(PSF[40:60,40:60],extent=[-10,10,-10,10])
    # plt.title("Fiber TPB PSF, zoom")
    # plt.colorbar()
    # plt.show()
    
    
    
    
    
    
    

    
    
# load_PSF = glob.glob(path_to_dataset + r'/*/PSF.npy')[0]
# PSF = np.load(load_PSF)
# # Show TPB Teflon hits on a single text file, can be removed later


# file = r'/media/amir/9C33-6BBD/NEXT_work/Geant4/nexus/' +\
#         r'small_cluster_hitpoints_dataset/fibers_6mm_in_10M/TPB_hits.txt'
# hitmap = np.array(np.genfromtxt(file)[:,0:2])
# PSF, x_hist, y_hist = np.histogram2d(hitmap[:,0], hitmap[:,1],
#                                       range=[[-size/2,size/2],[-size/2,size/2]],
#                                       bins=bins)
# fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16.5,8))
# fig.suptitle(r'TPB hit map, 10M photons fired forward', fontsize=15)
# im = ax0.imshow(PSF, extent=[-size/2, size/2, -size/2, size/2])
# ax0.set_xlabel('x [mm]');
# ax0.set_ylabel('y [mm]');
# ax0.set_title('PSF image')
# # fig.colorbar(im, orientation='vertical', location='left')
# divider = make_axes_locatable(ax0)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im, cax=cax)

# # y = PSF.sum(axis=0) 
# y = PSF[int(size/2),:]
# peaks, _ = find_peaks(y)
# fwhm = np.max(peak_widths(y, peaks, rel_height=0.5)[0])
# ax1.plot(np.arange(-size/2,size/2,1),y/np.sum(PSF[int(size/2),:]),
#          linewidth=2) # normalize cross section
# ax1.set_xlabel('mm')
# ax1.set_ylabel('Charge')
# ax1.set_title('Charge profile')
# ax1.grid(linewidth=1)
# fwhm_text = f"FWHM = {fwhm:.3f}"  # format to have 3 decimal places
# ax1.text(0.95, 0.95, fwhm_text, transform=ax1.transAxes, 
#           verticalalignment='top', horizontalalignment='right', 
#           color='red', fontsize=12, fontweight='bold',
#           bbox=dict(facecolor='white', edgecolor='red',
#                     boxstyle='round,pad=0.5'))

# fig.tight_layout()
# plt.show()



# ## Show TPB hits vs theoretical monte carlo
# # Load theoretical MC PSF
# anode_track_gap = 2.5
# el_gap = 10

# # Load theoretical PSF
# Mfolder = '/home/amir/Desktop/NEXT_work/Resolving_Power/Results/'
# folder = Mfolder + f'Resolving_Power_EL_gap{el_gap}mm_Tracking_Plane_Gap{anode_track_gap}mm/'
# Save_PSF  = f'{folder}PSF/'  
# psf_file_name = 'PSF_matrix'
# evt_PSF_output = Save_PSF + psf_file_name + '.npy'
# MC_PSF = np.load(evt_PSF_output)

# MC_y = MC_PSF[int(size/2),:]



# # plot
# fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16.5,8))
# fig.suptitle(r'TPB hit map, 10M photons fired forward', fontsize=15)
# im = ax0.imshow(PSF, extent=[-size/2, size/2, -size/2, size/2])
# ax0.set_xlabel('x [mm]')
# ax0.set_ylabel('y [mm]')
# ax0.set_title('PSF image')
# divider = make_axes_locatable(ax0)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im, cax=cax)

# y = PSF[int(size/2),:]
# peaks, _ = find_peaks(y)
# fwhm = np.max(peak_widths(y, peaks, rel_height=0.5)[0])

# # Adding labels and colors to the plots
# ax1.plot(np.arange(-size/2,size/2,1), y/np.sum(PSF[int(size/2),:]), 
#          linewidth=2, color='blue', label='Geant4 TPB hits, 10M')  # normalize cross section and set color to blue
# ax1.plot(np.arange(-size/2,size/2,1), MC_y/np.sum(MC_PSF[int(size/2),:]), 
#          linewidth=2, color='green', label='MC hits, 10M')  # normalize cross section and set color to red

# ax1.set_xlabel('mm')
# ax1.set_ylabel('Charge')
# ax1.set_title('Charge profile')
# ax1.grid(linewidth=1)
# fwhm_text = f"FWHM = {fwhm:.3f}"  # format to have 3 decimal places
# ax1.text(0.95, 0.95, fwhm_text, transform=ax1.transAxes, 
#           verticalalignment='top', horizontalalignment='right', 
#           color='red', fontsize=12, fontweight='bold',
#           bbox=dict(facecolor='white', edgecolor='red',
#                     boxstyle='round,pad=0.5'))

# ax1.legend(loc='upper left')  # Display the legend

# fig.tight_layout()
# plt.show()



# # plot around center uit cell
# plt.imshow(PSF[40:60,40:60],extent=[-10,10,-10,10])
# plt.colorbar()
# plt.show()


# In[4]
# Generate all PSFs (of geant4 TPB hits) from SquareFiberDataset

TO_GENERATE = False
TO_PLOT = False
TO_SAVE = False

if TO_GENERATE:
    for dir in geometry_dirs:
        PSF_TPB = psf_creator(dir,create_from="TPB",to_plot=TO_PLOT,to_smooth=False)
        os.chdir(dir)
        os.chdir('..')
        
        if TO_SAVE:
            save_PSF = r'PSF.npy'
            np.save(save_PSF,PSF_TPB)
            
# In[5]
# plot and save all TPB PSFs from SquareFiberDataset in their respective folders
TO_PLOT = False
TO_SAVE = False

if TO_PLOT:
    for dir in tqdm(geometry_dirs):
        os.chdir(dir)
        working_dir = r'Working on directory:'+f'\n{os.getcwd()}'
        print(working_dir)
        PSF = np.load(r'PSF.npy')
        fig = plot_PSF(PSF=PSF)
        
        if TO_SAVE:
            save_path = r'PSF_plot.jpg'
            fig.savefig(save_path)  
        plt.close(fig)  
        
# In[6]
'''
Generate and save twin events dataset, after shifting, centering and rotation
'''

def find_highest_number(directory):
    '''
    For the case more data is generated and added to existing data,
    finds the last data sample created by finding the max data number.

    Parameters
    ----------
    directory : str
        The directory of data to search in.

    Returns
    -------
    highest_number : int
        the number of the last data sample created.
        if not found , returns -1

    '''
    highest_number = -1  # Start with a default value

    for entry in os.scandir(directory):
        if entry.is_dir():  # Check if the entry is a directory
            for file in os.scandir(entry.path):
                if file.is_file() and file.name.endswith('.npy'):
                    # Extracting the number from the file name
                    number = int(file.name.split('.')[0])
                    highest_number = max(highest_number, number)

    return highest_number


TO_GENERATE = False
TO_SAVE = False
samples_per_geometry = 100

if TO_GENERATE:
    x_match_str = r"_x=(-?\d+(?:\.\d+)?(?:e-?\d+)?)mm"
    y_match_str = r"_y=(-?\d+(?:\.\d+)?(?:e-?\d+)?)mm"

    for i,geo_dir in tqdm(enumerate(geometry_dirs[0:1])):
        
        # find min distance that will be of interest
        match = re.search(r"_pitch=(\d+(?:\.\d+)?)mm", geo_dir)
        pitch = float(match.group(1))
        dist_min_threshold = pitch # mm

        # assign input and output directories
        print(geo_dir)
        working_dir = geo_dir + r'/Geant4_Kr_events'
        save_dir = geo_dir + r'/combined_event_SR' 
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        os.chdir(working_dir)
        PSF = np.load(geo_dir + '/PSF.npy')
        PSF = smooth_PSF(PSF)

        event_pattern = "SiPM_hits"
        event_list = [entry.name for entry in os.scandir() if entry.is_file() 
                      and entry.name.startswith(event_pattern)]
        
        # in case we add more samples, find highest existing sample number
        highest_existing_data_number = find_highest_number(save_dir)
        if highest_existing_data_number == -1:
            highest_existing_data_number = 0

        for j in tqdm(range(samples_per_geometry+1)):
            event_pair = random.sample(event_list, k=2)
    
            # grab event 0 x,y original generation coordinates
            x0_match = re.search(x_match_str, event_pair[0])
            x0 = float(x0_match.group(1))
            y0_match = re.search(y_match_str, event_pair[0])
            y0 = float(y0_match.group(1))
    
            # grab event 1 x,y original generation coordinates
            x1_match = re.search(x_match_str, event_pair[1])
            x1 = float(x1_match.group(1))
            y1_match = re.search(y_match_str, event_pair[1])
            y1 = float(y1_match.group(1))
    
                    
            event_to_stay, event_to_shift = np.genfromtxt(event_pair[0]), np.genfromtxt(event_pair[1])
    
            # Assign each hit to a SiPM
            event_to_stay_SR = []
            for hit in event_to_stay:
                sipm = assign_hit_to_SiPM(hit=hit, pitch=pitch, n=n_sipms)
                if sipm:  # if the hit belongs to a SiPM
                    event_to_stay_SR.append(sipm)
               
            event_to_stay_SR = np.array(event_to_stay_SR)
    
    
            # Assign each hit to a SiPM
            event_to_shift_SR = []
            for hit in event_to_shift:
                sipm = assign_hit_to_SiPM(hit=hit, pitch=pitch, n=n_sipms)
                if sipm:  # if the hit belongs to a SiPM
                    event_to_shift_SR.append(sipm)
               
            event_to_shift_SR = np.array(event_to_shift_SR)
    
            # shift "event_shift_SR"
            m, n = np.random.randint(1,4), np.random.randint(1,4) 
            # m , n = 2,1
            shifted_event_SR = event_to_shift_SR + [m*pitch, n*pitch]
    
            # Combine the two events
            combined_event_SR = np.concatenate((event_to_stay_SR, shifted_event_SR))
            shifted_event_coord = np.array([x1, y1]) + [m*pitch, n*pitch]
    
            # get distance between stay and shifted
            dist = (np.sqrt((x0-shifted_event_coord[0])**2+(y0-shifted_event_coord[1])**2))
            if dist < dist_min_threshold:
                samples_per_geometry -= 1
                continue
            # get midpoint of stay and shifted
            midpoint = [(x0+shifted_event_coord[0])/2,(y0+shifted_event_coord[1])/2]
            # print(f'distance = {dist}mm')
            # print(f'midpoint = {midpoint}mm')
    
            # center combined event using midpoint
            centered_combined_event_SR = combined_event_SR - midpoint
    
            # # Save combined centered event to suitlable folder according to sources distance
            # save_dir = save_dir + f'/{int(dist)}mm'
            # if not os.path.isdir(save_dir):
            #     os.mkdir(save_dir)
            # save_path = save_dir + f'/{i}.npy'
            # np.save(save_path,centered_combined_event_SR)
    
    
            # rotate combined event 
            theta = np.arctan2(y0-shifted_event_coord[1],x0-shifted_event_coord[0])
            # theta = np.arctan((y0-shifted_event_coord[1]) / (x0-shifted_event_coord[0]))
            rot_matrix = np.array([[np.cos(theta),-np.sin(theta)],
                                    [np.sin(theta),np.cos(theta)]])
            combined_rotated_event_SR = np.matmul(centered_combined_event_SR,rot_matrix)
    
            # Save combined centered ROTATED event to suitlable folder according to sources distance
            if TO_SAVE:
                save_to_dir = save_dir + f'/{int(dist)}_mm'
                if not os.path.isdir(save_to_dir):
                    os.mkdir(save_to_dir)
                save_path = save_to_dir + f'/{highest_existing_data_number+j}.npy'
                np.save(save_path,combined_rotated_event_SR)
                
                       
# In[7]
'''
Load twin events after shifted, centered and rotated. interpolate and deconv.
'''






# In[8]
'''
combine events, interpolate and RL
This shows 1 sample at a time for a chosen m,n shift values for different geometries
for each geometry:
sample 2 events -> shift 1 of them to (randint(0,max_n),randint(0,max_n))*(x2,y2)
-> make sensor response, save distance (example 16-17mm, 17-18mm), rotate,
interpolate, RL and Peak to Valley (P2V)
'''

def peaks(array):
    fail = 0
    hight_threshold = 0.1*max(array)
    peak_idx, properties = find_peaks(array, height = hight_threshold)
    if len(peak_idx) != 2:
        fail = 1
    return fail, peak_idx, properties['peak_heights']

def find_min_between_peaks(array, left, right):
    return min(array[left:right])


def richardson_lucy(image, psf, iterations=50, iter_thr=0.):
    """Richardson-Lucy deconvolution (modification from scikit-image package).

    The modification adds a value=0 protection, the possibility to stop iterating
    after reaching a given threshold and the generalization to n-dim of the
    PSF mirroring.

    Parameters
    ----------
    image : ndarray
       Input degraded image (can be N dimensional).
    psf : ndarray
       The point spread function.
    iterations : int, optional
       Number of iterations. This parameter plays the role of
       regularisation.
    iter_thr : float, optional
       Threshold on the relative difference between iterations to stop iterating.

    Returns
    -------
    im_deconv : ndarray
       The deconvolved image.
    Examples
    --------
    >>> from skimage import color, data, restoration
    >>> camera = color.rgb2gray(data.camera())
    >>> from scipy.signal import convolve2d
    >>> psf = np.ones((5, 5)) / 25
    >>> camera = convolve2d(camera, psf, 'same')
    >>> camera += 0.1 * camera.std() * np.random.standard_normal(camera.shape)
    >>> deconvolved = restoration.richardson_lucy(camera, psf, 5)
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
    """
    # compute the times for direct convolution and the fft method. The fft is of
    # complexity O(N log(N)) for each dimension and the direct method does
    # straight arithmetic (and is O(n*k) to add n elements k times)
    direct_time = np.prod(image.shape + psf.shape)
    fft_time    = np.sum([n*np.log(n) for n in image.shape + psf.shape])

    # see whether the fourier transform convolution method or the direct
    # convolution method is faster (discussed in scikit-image PR #1792)
    time_ratio = 40.032 * fft_time / direct_time
    if time_ratio <= 1 or len(image.shape) > 2:
        convolve_method = fftconvolve
    else:
        convolve_method = convolve

    image      = image.astype(np.float)
    psf        = psf.astype(np.float)
    im_deconv  = 0.5 * np.ones(image.shape)
    s          = slice(None, None, -1)
    psf_mirror = psf[(s,) * psf.ndim] ### Allow for n-dim mirroring.
    eps        = np.finfo(image.dtype).eps ### Protection against 0 value
    ref_image  = image/image.max()
    # lowest_value = 4.94E-324
    lowest_value = 4.94E-20
    
    
    for i in range(iterations):
        x = convolve_method(im_deconv, psf, 'same')
        np.place(x, x==0, eps) ### Protection against 0 value
        relative_blur = image / x
        im_deconv *= convolve_method(relative_blur, psf_mirror, 'same')
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_diff = np.sum(np.divide(((im_deconv/im_deconv.max() - ref_image)**2), ref_image))
        # if i>50:
        #     print(f'{i},rel_diff={rel_diff}')     
        if rel_diff < iter_thr: ### Break if a given threshold is reached.
            break  
        ref_image = im_deconv/im_deconv.max()     
        ref_image[ref_image<=lowest_value] = lowest_value      
        rel_diff_checkout = rel_diff # Store last value of rel_diff before it becomes NaN
    return rel_diff_checkout, i, im_deconv


def P2V(vector):
    fail, peak_idx, heights = peaks(vector)
    if fail:
        print(r'Could not find any peaks for event!')
        return 0
    else:
        # Combine peak indices and heights into a list of tuples
        peaks_with_heights = list(zip(peak_idx, heights))
     
        # Sort by height in descending order and select the top two
        top_two_peaks = sorted(peaks_with_heights, key=lambda x: x[1], reverse=True)[:2]

        # Extract heights of the top two peaks
        top_heights = [peak[1] for peak in top_two_peaks]

        # Calculate the average height of the top two peaks
        avg_peak = np.average(top_heights)

        # Ensure indices are in ascending order for slicing
        left_idx, right_idx = sorted([top_two_peaks[0][0], top_two_peaks[1][0]])


        # print(f'left idx = {left_idx}')
        # print(f'right idx = {right_idx}')
        # Find the valley height between the two strongest peaks
        valley_height = find_min_between_peaks(vector, left_idx, right_idx)
        if valley_height <= 0 and avg_peak > 0:
            return float('inf')

        avg_P2V = avg_peak / valley_height
        return avg_P2V

    

TO_PLOT = True

# override previous bins/size settings
bins = 250
size = bins
# seed = random.randint(0,10**9)
seed = 322414211
random.seed(seed)
np.random.seed(seed)

x_match_str = r"_x=(-?\d+(?:\.\d+)?(?:e-?\d+)?)mm"
y_match_str = r"_y=(-?\d+(?:\.\d+)?(?:e-?\d+)?)mm"

# for i,geo_dir in tqdm(enumerate(geometry_dirs)):
# geo_dir = geometry_dirs[-1]


# # works bad for this geometry
# geo_dir = ('/media/amir/Extreme Pro/SquareFiberDatabase/' +
#             'ELGap=1mm_pitch=15.6mm_distanceFiberHolder=2mm_distanceAnodeHolder=10mm_holderThickness=10mm')

# geo_dir = ('/media/amir/Extreme Pro/SquareFiberDatabase/' +
#             'ELGap=10mm_pitch=15.6mm_distanceFiberHolder=-1mm_distanceAnodeHolder=10mm_holderThickness=10mm')

# geo_dir = ('/media/amir/Extreme Pro/SquareFiberDatabase/' +
#             'ELGap=10mm_pitch=15.6mm_distanceFiberHolder=5mm_distanceAnodeHolder=10mm_holderThickness=10mm')

geo_dir = str(random.sample(geometry_dirs, k=1)[0])

# grab pitch value as a reference to minimum distance between sources
match = re.search(r"_pitch=(\d+(?:\.\d+)?)mm", geo_dir)
pitch = float(match.group(1))
dist_min_threshold = pitch #mm

# assign input and output directories
print(geo_dir)
working_dir = geo_dir + r'/Geant4_Kr_events'
save_dir = geo_dir + r'/combined_event_SR' 
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

os.chdir(working_dir)
PSF = np.load(geo_dir + '/PSF.npy')
PSF = smooth_PSF(PSF)

event_pattern = "SiPM_hits"

# for j in tqdm(range(100)):
event_list = [entry.name for entry in os.scandir() if entry.is_file() 
              and entry.name.startswith(event_pattern)]
# second for
event_pair = random.sample(event_list, k=2)

# grab event 0 x,y original generation coordinates
x0_match = re.search(x_match_str, event_pair[0])
x0 = float(x0_match.group(1))
y0_match = re.search(y_match_str, event_pair[0])
y0 = float(y0_match.group(1))

# grab event 1 x,y original generation coordinates
x1_match = re.search(x_match_str, event_pair[1])
x1 = float(x1_match.group(1))
y1_match = re.search(y_match_str, event_pair[1])
y1 = float(y1_match.group(1))

        
event_to_stay, event_to_shift = np.genfromtxt(event_pair[0]), np.genfromtxt(event_pair[1])

# Assign each hit to a SiPM
event_to_stay_SR = []
for hit in event_to_stay:
    sipm = assign_hit_to_SiPM(hit=hit, pitch=pitch, n=n_sipms)
    if sipm:  # if the hit belongs to a SiPM
        event_to_stay_SR.append(sipm)
   
event_to_stay_SR = np.array(event_to_stay_SR)


# Assign each hit to a SiPM
event_to_shift_SR = []
for hit in event_to_shift:
    sipm = assign_hit_to_SiPM(hit=hit, pitch=pitch, n=n_sipms)
    if sipm:  # if the hit belongs to a SiPM
        event_to_shift_SR.append(sipm)
   
event_to_shift_SR = np.array(event_to_shift_SR)

# shift "event_shift_SR"
m, n = np.random.randint(1,3), np.random.randint(1,3) 
# m , n = 2,1
shifted_event_SR = event_to_shift_SR + [m*pitch, n*pitch]

# Combine the two events
combined_event_SR = np.concatenate((event_to_stay_SR, shifted_event_SR))
shifted_event_coord = np.array([x1, y1]) + [m*pitch, n*pitch]

# get distance between stay and shifted
dist = (np.sqrt((x0-shifted_event_coord[0])**2+(y0-shifted_event_coord[1])**2))
# if dist < dist_min_threshold:
#     continue
# get midpoint of stay and shifted
midpoint = [(x0+shifted_event_coord[0])/2,(y0+shifted_event_coord[1])/2]
print(f'distance = {dist}mm')
# print(f'midpoint = {midpoint}mm')

# center combined event using midpoint
centered_combined_event_SR = combined_event_SR - midpoint

# # Save combined centered event to suitlable folder according to sources distance
# save_dir = save_dir + f'/{int(dist)}mm'
# if not os.path.isdir(save_dir):
#     os.mkdir(save_dir)
# save_path = save_dir + f'/{i}.npy'
# np.save(save_path,centered_combined_event_SR)


# rotate combined event 
theta = np.arctan2(y0-shifted_event_coord[1],x0-shifted_event_coord[0])
# theta = np.arctan((y0-shifted_event_coord[1]) / (x0-shifted_event_coord[0]))
rot_matrix = np.array([[np.cos(theta),-np.sin(theta)],
                        [np.sin(theta),np.cos(theta)]])
combined_rotated_event_SR = np.matmul(centered_combined_event_SR,rot_matrix)

# # Save combined centered rotated event to suitlable folder according to sources distance
# save_dir = save_dir + f'/{int(dist)}mm'
# if not os.path.isdir(save_dir):
#     os.mkdir(save_dir)
# save_path = save_dir + f'/{i}.npy'
# np.save(save_path,combined_rotated_event_SR)


#### interpolation ####

# # Create a 2D histogram
hist, x_edges, y_edges = np.histogram2d(combined_rotated_event_SR[:,0],
                                        combined_rotated_event_SR[:,1],
                                        range=[[-size/2, size/2], [-size/2, size/2]],
                                        bins=bins)



# Compute the centers of the bins
x_centers = (x_edges[:-1] + x_edges[1:]) / 2
y_centers = (y_edges[:-1] + y_edges[1:]) / 2

hist_hits_x_idx, hist_hits_y_idx = np.where(hist>0)
hist_hits_x, hist_hits_y = x_centers[hist_hits_x_idx], y_centers[hist_hits_y_idx]

hist_hits_vals = hist[hist>0]


# Define the interpolation grid
x_range = np.linspace(-size/2, size/2, num=bins)
y_range = np.linspace(-size/2, size/2, num=bins)
x_grid, y_grid = np.meshgrid(x_range, y_range)
# Perform the interpolation
interp_img = griddata((hist_hits_x, hist_hits_y), hist_hits_vals, (x_grid, y_grid),
              method='cubic', fill_value=0)

# optional, cut interp image values below 0
interp_img[interp_img<0] = 0

# try P2V without deconv
x_cm, y_cm = ndimage.measurements.center_of_mass(interp_img)
x_cm, y_cm = int(x_cm), int(y_cm)
interp_img_1d = interp_img[y_cm,:]
avg_P2V_interp = P2V(interp_img_1d)
print(f'avg_P2V_interp={avg_P2V_interp}')


# RL deconvolution
# rel_diff_checkout, cutoff_iter, deconv = richardson_lucy(interp_img, PSF,
#                                                   iterations=75, iter_thr=0.05)
rel_diff_checkout, cutoff_iter, deconv = richardson_lucy(interp_img, PSF,
                                                  iterations=75, iter_thr=0.01)

print(f'rel_diff = {rel_diff_checkout}')
print(f'cut off iteration = {cutoff_iter}')


# try P2V with deconv
# print(f'min deconv = {np.around(np.min(deconv),3)}')
x_cm, y_cm = ndimage.measurements.center_of_mass(deconv)
x_cm, y_cm = int(x_cm), int(y_cm)
deconv_1d = deconv[y_cm,:]
avg_P2V_deconv = P2V(deconv_1d)
print(f'avg_P2V_deconv = {avg_P2V_deconv}')
print(f'seed = {seed}')

# deconvolution diverges
if rel_diff_checkout >= 1 :
    chosen_avg_P2V = np.around(avg_P2V_interp,3)
    print('\n\nDeconvolution process status: FAIL - diverging' +
          f'\nInterpolation P2V outperforms, avg_P2V={chosen_avg_P2V}')
# deconvolution converges
if rel_diff_checkout < 1:
    if avg_P2V_deconv >= avg_P2V_interp:
        chosen_avg_P2V = np.around(avg_P2V_deconv,3)
        print('\n\nDeconvolution process status: SUCCESS - converging' + 
              f'\nDeconvolution P2V outperforms, avg_P2V={chosen_avg_P2V}')
    if avg_P2V_deconv < avg_P2V_interp: # deconvolution converges but didn't outperform interp P2V
        chosen_avg_P2V = np.around(avg_P2V_interp,3)
        print('\n\nDeconvolution process status: SUCCEED - converging' + 
              f'\nInterpolation P2V outperforms, avg_P2V={chosen_avg_P2V}')
    if avg_P2V_deconv == 0 and avg_P2V_interp == 0:
        chosen_avg_P2V = -1
        print('Could not find a P2V value. Check sensor response image.')
        



if TO_PLOT:
    # plot sensor responses
    plot_sensor_response(event_to_stay_SR,bins,size)
    plot_sensor_response(event_to_shift_SR,bins,size)
    plot_sensor_response(combined_event_SR, bins, size)
    plot_sensor_response(centered_combined_event_SR, bins, size)
    plot_sensor_response(combined_rotated_event_SR, bins, size, noise=True)
    
    # plot interpolated combined event
    plt.imshow(interp_img, extent=[-size/2, size/2, -size/2, size/2], vmin=0)
    plt.colorbar(label='Photon hits')
    plt.title('Cubic Interpolation of Combined Rotated event')
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    plt.show()
    
    # plot PSF
    plt.imshow(PSF,vmin=0)
    plt.colorbar()
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    plt.title('PSF')
    plt.show()
    
    ## plot deconv + deconv profile

    fig, ax = plt.subplots(2,2,figsize=(15,13))
    im = ax[0,0].imshow(interp_img, extent=[-size/2, size/2, -size/2, size/2])
    divider = make_axes_locatable(ax[0,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax[0,0].set_xlabel('x [mm]')
    ax[0,0].set_ylabel('y [mm]')
    ax[0,0].set_title('Interpolated image')

    legend = f'Avg P2V={np.around(avg_P2V_interp,3)}'
    ax[0,1].plot(np.arange(-size/2, size/2), interp_img_1d,label=legend)
    ax[0,1].set_xlabel('x [mm]')
    ax[0,1].set_ylabel('photon hits')
    ax[0,1].set_title('Interpolated image profile')
    ax[0,1].grid()
    ax[0,1].legend(fontsize=10)
    # ax[0,1].set_ylim([0,None])
    
    # deconv
    im = ax[1,0].imshow(deconv, extent=[-size/2, size/2, -size/2, size/2])
    divider = make_axes_locatable(ax[1,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax[1,0].set_xlabel('x [mm]')
    ax[1,0].set_ylabel('y [mm]')
    ax[1,0].set_title('RL deconvolution')
    # deconv profile
    legend = f'Avg P2V={np.around(avg_P2V_deconv,3)}'
    ax[1,1].plot(np.arange(-size/2, size/2), deconv_1d,label=legend)
    ax[1,1].set_xlabel('x [mm]')
    ax[1,1].set_ylabel('photon hits')
    ax[1,1].set_title('RL deconvolution profile')
    ax[1,1].grid()
    ax[1,1].legend(fontsize=10)
    # ax[1,1].set_ylim([0,None])
    geo_params = geo_dir.split('/SquareFiberDatabase/')[-1]
    fig.suptitle(f'{geo_params}\nEvent spacing = {np.around(dist,3)}[mm]',fontsize=15)
    fig.tight_layout()
    plt.show()

# In[7]

    



















# In[6]










