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
import glob
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage




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
    x_cm, y_cm = int(x_cm), int(y_cm)
    psf_size = PSF.shape[0]
    x,y = np.indices((psf_size,psf_size))
    smooth_PSF = np.zeros((psf_size,psf_size))
    r = np.arange(0,(1/np.sqrt(2))*psf_size,1)
    for radius in r:
        R = np.full((1, psf_size), radius)
        circle = (np.abs(np.hypot(x-x_cm, y-y_cm)-R) < 0.6).astype(int)
        circle_ij = np.where(circle==1)
        smooth_PSF[circle_ij] = np.mean(PSF[circle_ij])
    
    return smooth_PSF



def psf_creator(geo_directory,to_plot=True,to_smooth=True):
    '''
    Gives a picture of the SiPM (sensor wall) response to an event involving 
    the emission of light in the EL gap.
    Receives:
        geo_directory: str, path to geometry directory
        to_plot: bool, flag - if True, plots the PSF
        to_smooth: bool, flag - if True, smoothes the PSF
    Returns:
        PSF: ndarray, the Point Spread Function
    '''
    
    os.chdir(path_to_dataset + "/" + dir)
    print('\nWorking on directory:'+f'\n{os.getcwd()}')
    SiPM_files = glob.glob(r'SiPM*')
    
    # pattern to extract x,y values of each event from file name
    pattern = r"-?\d+.\d+"

    PSF_list = []
    bins = 500
    for filename in SiPM_files:
        # Load hitmap
        hitmap = np.array(np.genfromtxt(filename)[:,0:2])
        # Store x,y values of event
        matches = re.findall(pattern, filename)
        x_event = float(matches[0])
        y_event = float(matches[1])
        # shift each event to center        
        shifted_hitmap = hitmap - [x_event, y_event]
        # Add all shifted maps to create the geometry's PSF
        PSF_list.append(shifted_hitmap)

    # Concatenate all shifted hitmaps into a single array
    PSF = np.vstack(PSF_list)
    PSF, x_hist, y_hist = np.histogram2d(PSF[:,0], PSF[:,1],
                                         range=[[-100,100],[-100,100]],
                                         bins=bins)
    
    if to_plot:
        plt.hist2d(PSF[:,0],PSF[:,1],
                   bins=(bins, bins), cmap=plt.cm.jet)
        plt.xlabel('[mm]')
        plt.ylabel('[mm]')
        plt.title('PSF')
        plt.colorbar()
        plt.show()

    if to_smooth:
        #Smooth the PSF
        PSF = smooth_PSF(PSF)
        plt.imshow(PSF)
        plt.show()
        # np.save(evt_PSF_output,smoothed_PSF)
        # np.save(evt_PSF_output,PSF) #unsmoothed
    return PSF



# estimates entire dataset size on disk
def estimate_dataset_size_on_disk(factor):
    '''
    Estimates entire dataset size on disk

    Parameters
    ----------
    factor : float, dataset factor - 1 for 100K photons per sim, 4.62 for 
    462K per sim, etc.

    Returns
    -------
    None.

    '''
    datset_size_factor_per_100K = 1 # 1 is for 100K photons per sim
    single_sipm_file_size = datset_size_factor_per_100K*0.008 # mb
    single_tpb_file_size = datset_size_factor_per_100K*2 # mb
    geometry_group = [18,18,18]
    sims_per_geom = [100,400,961]
    total_files_files = 2*np.sum(np.multiply(geometry_group, sims_per_geom))
    total_sipm_txt_files_size = 0.5*total_files_files * single_sipm_file_size     
    total_tpb_txt_files_size = 0.5*total_files_files * single_tpb_file_size   
    print(f'Total estimated Sipm text files size on disk = {int(total_sipm_txt_files_size)} MB')
    print(f'Total estimated TPB text files size on disk = {int(total_tpb_txt_files_size)} MB')


estimate_dataset_size_on_disk(factor=1)

# In[2]
# plot SiPM and TPB hits, basically just show a sample of database
path_to_dataset = r'/media/amir/9C33-6BBD/NEXT_work/Geant4/nexus/' + \
                  r'small_cluster_hitpoints_dataset/SquareFiberMacrosAndOutputs'

n_sipms = 25
n_sipms_per_side = (n_sipms-1)/2

geometry_dirs = os.listdir(path_to_dataset)
for geometry in geometry_dirs:
    os.chdir(path_to_dataset + "/" + geometry)
    print(os.getcwd())
    
    # get the pitch value
    match = re.search('pitch=(.*?)_', geometry)
    if match:
        pitch = match.group(1)
        pitch = float(pitch.split('mm')[0]) # stored value in mm
        
    SiPM_files = glob.glob(r'SiPM*')
    TPB_files = glob.glob(r'TPB*')
    for i in range(10,12):
        SR_response = np.genfromtxt(SiPM_files[i])[:,0:2]
        sipm_x_coords = SR_response[:,0]
        sipm_y_coords = SR_response[:,1]
        
        TPB_response = np.genfromtxt(TPB_files[i])[:,0:2]
        tpb_x_coords = TPB_response[:,0]
        tpb_y_coords = TPB_response[:,1]
        
        fig, (ax0,ax1) = plt.subplots(1,2,figsize=(20,10))
        
        # Save the mappable object for the colorbar
        hist_sipm = ax0.hist2d(sipm_x_coords,sipm_y_coords,
                               bins=(300, 300), cmap=plt.cm.jet)
        ax0.set_title("SiPM hits")
        
        divider0 = make_axes_locatable(ax0)
        cax0 = divider0.append_axes("right", size="5%", pad=0.05)
        ax0.set_xlabel('[mm]')
        ax0.set_ylabel('[mm]')
        # Pass the mappable object (hist0[3]) to colorbar
        fig.colorbar(hist_sipm[3], cax=cax0)
        
        hist_tpb = ax1.hist2d(tpb_x_coords, tpb_y_coords,
                              bins=(300, 300), cmap=plt.cm.jet)
        ax1.set_xlim([-n_sipms_per_side*pitch,n_sipms_per_side*pitch])
        ax1.set_ylim([-n_sipms_per_side*pitch,n_sipms_per_side*pitch])
        ax1.set_title("TPB hits")
        ax1.set_xlabel('[mm]')
        ax1.set_ylabel('[mm]')
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        
        # Pass the mappable object (hist1[3]) to colorbar
        fig.colorbar(hist_tpb[3], cax=cax1)
        
        fig.suptitle(geometry + r'/' + SiPM_files[i])
        plt.show()


# In[3]
# Generate PSF and save
path_to_dataset = r'/media/amir/9C33-6BBD/NEXT_work/Geant4/nexus/' + \
                  r'small_cluster_hitpoints_dataset/SquareFiberMacrosAndOutputs'

geometry_dirs = os.listdir(path_to_dataset)
PSFS = []
for dir in geometry_dirs:
    PSF = psf_creator(dir,to_plot=True,to_smooth=False)
    
    # save_PSF = r'PSF.npy'
    # np.save(save_PSF,PSF)
    
# In[4]
# combine 2 events

def combine_PSFs(filename1, filename2):
    '''
    This function take 2 sets of hit points from two files, simulating the light from 2 sources hitting
    the tracking plane and merges them into one.
    '''
    pattern = r"[-+]?\d*\.\d+|\d+"
    
    # Load hitpoints from files
    PSF1 = np.genfromtxt(filename1)[:,0:2]
    PSF2 = np.genfromtxt(filename2)[:,0:2]
    
    # Extract positions from filenames
    matches1 = re.findall(pattern, filename1)
    x1_event = float(matches1[0])
    y1_event = float(matches1[1])

    matches2 = re.findall(pattern, filename2)
    x2_event = float(matches2[0])
    y2_event = float(matches2[1])
    
    # Shift hitpoints so that events are at their declared positions
    PSF1 += [x1_event, y1_event]
    PSF2 += [x2_event, y2_event]
    
    # Calculate distance and angle between events
    distance_between_sources = np.hypot(x2_event - x1_event, y2_event - y1_event)
    angle = np.arctan2(y2_event - y1_event, x2_event - x1_event)
    
    # Compute the vector from center of mass of PSF1 to PSF2
    cm_vector_x = np.cos(angle)
    cm_vector_y = np.sin(angle)
    
    # Calculate the point where PSF2 should be shifted to
    shift_point_x = x1_event + cm_vector_x * distance_between_sources
    shift_point_y = y1_event + cm_vector_y * distance_between_sources
    
    # Shift PSF2 to the desired position
    shift_x = shift_point_x - x2_event
    shift_y = shift_point_y - y2_event
    shifted_PSF2 = PSF2 + [shift_x, shift_y]
    
    # Combine the shifted PSF2 and PSF1
    combined_PSF = np.concatenate([PSF1, shifted_PSF2], axis=0)
    
    # Calculate center of mass of the combined PSF
    combined_cm_x, combined_cm_y = np.mean(combined_PSF, axis=0)
    
    # Shift the combined PSF so that the center of mass is at (0, 0)
    centered_combined_PSF = combined_PSF - [combined_cm_x, combined_cm_y]
    
    # Create a 2D histogram
    plt.hist2d(centered_combined_PSF[:, 0], centered_combined_PSF[:, 1], bins=(100, 100), cmap=plt.cm.jet)
    plt.colorbar()
    plt.show()


filename1 = r'/media/amir/9C33-6BBD/NEXT_work/Geant4/nexus/' + \
r'small_cluster_hitpoints_dataset/SquareFiberMacrosAndOutputs/' + \
r'ELGap=10mm_pitch=5mm_distanceFiberHolder=2mm_distanceAnodeHolder=2.5mm_holderThickness=10mm/' + \
r'SiPM_hits_x=-2.5mm_y=-2.0mm.txt'

filename2 = r'/media/amir/9C33-6BBD/NEXT_work/Geant4/nexus/' + \
r'small_cluster_hitpoints_dataset/SquareFiberMacrosAndOutputs/' + \
r'ELGap=10mm_pitch=5mm_distanceFiberHolder=2mm_distanceAnodeHolder=2.5mm_holderThickness=10mm/' + \
r'SiPM_hits_x=2.5mm_y=1.5mm.txt'

combine_PSFs(filename1, filename2)


