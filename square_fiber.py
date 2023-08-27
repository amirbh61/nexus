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
import time
from scipy.signal           import fftconvolve
from scipy.signal           import convolve
from invisible_cities.reco.deconv_functions     import richardson_lucy
from scipy.interpolate import interp2d

# Global settings #
n_sipms = 25
n_sipms_per_side = (n_sipms-1)/2



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
    factor : float, dataset factor - equals 1 for 100K photons per sim, 4.62 for 
    462K per sim, etc.

    Returns
    -------
    None.

    '''
    single_sipm_file_size = factor*0.008 # mb
    single_tpb_file_size = factor*2 # mb
    geometry_group = [18,18,18]
    sims_per_geom = [100,400,961]
    total_files_files = 2*np.sum(np.multiply(geometry_group, sims_per_geom))
    total_sipm_txt_files_size = 0.5*total_files_files * single_sipm_file_size     
    total_tpb_txt_files_size = 0.5*total_files_files * single_tpb_file_size   
    print(f'Total estimated Sipm text files size on disk = {int(total_sipm_txt_files_size)} MB')
    print(f'Total estimated TPB text files size on disk = {int(total_tpb_txt_files_size)} MB')


estimate_dataset_size_on_disk(factor=4.7)

# In[2]
# plot SiPM and TPB hits, basically just show a sample of database
path_to_dataset = r'/media/amir/9C33-6BBD/NEXT_work/Geant4/nexus/' + \
                  r'small_cluster_hitpoints_dataset/SquareFiberMacrosAndOutputs'

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
        
        # bins_sipm = int(np.sqrt(len(sipm_x_coords)))
        # bins_tpb = int(np.sqrt(len(tpb_x_coords)))
        bins_sipm = 300
        bins_tpb = 300
              
        
        hist_sipm = ax0.hist2d(sipm_x_coords,sipm_y_coords,
                               bins=(bins_sipm, bins_sipm), cmap=plt.cm.jet)
        ax0.set_title("SiPM hits")
        
        divider0 = make_axes_locatable(ax0)
        cax0 = divider0.append_axes("right", size="5%", pad=0.05)
        ax0.set_xlabel('[mm]')
        ax0.set_ylabel('[mm]')
        # Pass the mappable object (hist0[3]) to colorbar
        fig.colorbar(hist_sipm[3], cax=cax0)
        
        hist_tpb = ax1.hist2d(tpb_x_coords, tpb_y_coords,
                              bins=(bins_tpb, bins_tpb), cmap=plt.cm.jet)
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
import itertools
import random

'''
Create sensor response image for 2 events + interpolation of the signal.
'''

def find_equivalent_lattice_points(filename, shift):
    '''
    This function receives a text file of hitpoints, and looks for symmetrical 
    lattice points in the surrounding unit cells.
    
    
    Receives:
        filename: str, path to an event text file
        shift: double, distance to shift the event from its original position
        
    Returns:
        lattice_point: tuple, a lattice point sampled from all possible lattice points
    '''
    # need to be on geometry dir 
    matches = re.findall(xy_pattern, filename)
    x_event = float(matches[0][0])  # the first element in the first tuple
    y_event = float(matches[0][1])  # the second element in the first tuple
    
    
    x_lattice_points = [x_event+shift, x_event-shift]
    y_lattice_points = [y_event+shift, y_event-shift]
    possible_lattice_points = list(itertools.product(x_lattice_points,y_lattice_points))
    lattice_point = random.sample(possible_lattice_points, k=1)
    lattice_point = np.reshape(lattice_point,-1)
    if shift == 0:
        lattice_point = [x_event,y_event]
    return lattice_point
    
    
def shift_event_to_lattice_point(event_file, lattice_point):
    matches1 = re.findall(xy_pattern, event_file)
    x_event = float(matches1[0][0])
    y_event = float(matches1[0][1])
    event = np.genfromtxt(event_file)[:,0:2]
    shifted_event = event + [lattice_point[0]-x_event, lattice_point[1]-y_event]
    return np.array(shifted_event)

    
def sample_2_events():
    event_list = glob.glob(r'SiPM_hits*')
    events = random.sample(event_list, k=2)
    return events


def combine_events(event1_file, event2_file, shift, to_plot=True):
    '''
    This function take 2 sets of hit points from two files, simulating the light
    from 2 sources hitting the tracking plane and merges them into one.
    '''
    # Load hitpoints from files
    event1 = np.genfromtxt(event1_file)[:,0:2]
    
    # Extract positions from filenames
    matches1 = re.findall(xy_pattern, event1_file)
    x1_event = float(matches1[0][0])
    y1_event = float(matches1[0][1])

    matches2 = re.findall(xy_pattern, event2_file)
    x2_event = float(matches2[0][0])
    y2_event = float(matches2[0][1])
    
    ### The second event will be have its original x,y shifted  ###
    # Find its new equivalent point in a nearby cell
    lattice_point = find_equivalent_lattice_points(event2_file, shift=shift)
    # Shift the event to the new lattice point
    shifted_event2 = shift_event_to_lattice_point(event2_file, lattice_point)
    
    # Combine the shifted PSF2 and PSF1
    combined_event = np.concatenate([event1, shifted_event2], axis=0)
    
    # Calculate center of mass of the combined PSF
    combined_cm_x, combined_cm_y = np.mean(combined_event, axis=0)
    
    distance_between_events = np.hypot(x1_event - lattice_point[0],
                                        y1_event - lattice_point[1])
    
    # Shift the combined PSF so that the center of mass is at (0, 0)
    centered_combined_event = combined_event - [combined_cm_x, combined_cm_y]
    
    
    bins = int(np.sqrt(len(event1)+len(shifted_event2)))
    # bins = 300
    H, x_hist, y_hist = np.histogram2d(centered_combined_event[:, 0],
                                       centered_combined_event[:, 1], bins=bins)
      
    if to_plot:
        # Create a 2D histogram
        plt.figure(figsize=(15,10))

        plt.hist2d(centered_combined_event[:, 0], centered_combined_event[:, 1],
                    bins=((bins, bins)), cmap=plt.cm.jet)
        # plt.imshow(H)   
       
        plt.colorbar()
        
        title = f'Event spacing={np.round(distance_between_events,3)}mm\n'+ \
                  f'event1: (x,y)={(np.round(x1_event,3),np.round(y1_event,3))}, ' + \
                  f'event2: (x,y)={(np.round(lattice_point[0],3),np.round(lattice_point[1],3))}'
    
        plt.title(title)
        plt.show()
        
        

    return H, x_hist, y_hist, title



## run just one prechosen pair

# filename1 = r'/media/amir/9C33-6BBD/NEXT_work/Geant4/nexus/' + \
# r'small_cluster_hitpoints_dataset/SquareFiberMacrosAndOutputs/' + \
# r'ELGap=10mm_pitch=5mm_distanceFiberHolder=2mm_distanceAnodeHolder=2.5mm_holderThickness=10mm/' + \
# r'SiPM_hits_x=-2.5mm_y=-2.0mm.txt'

# filename2 = r'/media/amir/9C33-6BBD/NEXT_work/Geant4/nexus/' + \
# r'small_cluster_hitpoints_dataset/SquareFiberMacrosAndOutputs/' + \
# r'ELGap=10mm_pitch=5mm_distanceFiberHolder=2mm_distanceAnodeHolder=2.5mm_holderThickness=10mm/' + \
# r'SiPM_hits_x=2.5mm_y=1.5mm.txt'

# combine_events(filename1, filename2)



# define regex pattern for file name
xy_pattern = r"x=(-?\d*\.\d+)mm_y=(-?\d*\.\d+)mm" # gets x,y of each event
pitch_pattern = r'pitch=(.*?)_'

path_to_dataset = r'/media/amir/9C33-6BBD/NEXT_work/Geant4/nexus/' + \
                  r'small_cluster_hitpoints_dataset/SquareFiberMacrosAndOutputs'


geometry_dirs = os.listdir(path_to_dataset)
for geometry in geometry_dirs:
    folder = path_to_dataset + "/" + geometry
    os.chdir(folder)
    print(f'Current working directory:\n{os.getcwd()}',end='\n')
    
    # store pitch value from file name
    pitch_match = re.search(pitch_pattern, geometry)
    pitch = pitch_match.group(1)
    pitch = float(pitch.split('mm')[0]) # stored value in mm
    print(f'pitch={pitch}mm',end='\n\n')
    
    # create save paths
    Save_SR  = f'{folder}/Sensor_Response_2_events_raw'
    Save_SR_interpolate  = f'{folder}/Sensor_Response_2_events_interpolated'
    if not os.path.isdir(Save_SR):
        os.mkdir(Save_SR)
    if not os.path.isdir(Save_SR_interpolate):
        os.mkdir(Save_SR_interpolate)
    source_spacing=2*pitch

        
    for attempt in range(8):
        events = sample_2_events()

        H, x_hist, y_hist, title = combine_events(events[0], events[1],
                                           shift=source_spacing,to_plot=True)
        # Create a meshgrid for interpolation
        x_centers = 0.5 * (x_hist[1:] + x_hist[:-1])
        y_centers = 0.5 * (y_hist[1:] + y_hist[:-1])
        
        # Create the interpolation function
        interp_f = interp2d(x_centers, y_centers, H, kind='cubic')
        
        # Create a finer meshgrid for the interpolated data
        x_fine = np.linspace(x_centers.min(), x_centers.max(), 100)
        y_fine = np.linspace(y_centers.min(), y_centers.max(), 100)
        
        hist_interpolated = interp_f(x_fine, y_fine)
               
        ### SAVE ###
        SR_output = Save_SR + f'/source_spacing={source_spacing}mm_pair_id={attempt}'
        SR_interpolation_output = Save_SR_interpolate + \
        f'/source_spacing={source_spacing}mm_pair_id={attempt}'
        
        # Save H (2D histogram)
        np.save(SR_output+"_hist.npy", H)
        with open(SR_output+"_hist_details.txt", "w") as file:
            file.write(title)
        np.save(SR_output+"_x_bins.npy", x_hist)
        np.save(SR_output+"_y_bins.npy", y_hist)
        # Save interpolation
        np.save(SR_interpolation_output+"_hist.npy", hist_interpolated)
        with open(SR_interpolation_output+"_hist_details.txt", "w") as file:
            file.write(title)
        np.save(SR_interpolation_output+"_x_fine.npy", x_fine)
        np.save(SR_interpolation_output+"_y_fine.npy", y_fine)
    
# In[5]
'''
Plot sensor response and interpolated data
'''

def file_content_to_string(filename):
    with open(filename, 'r') as f:
        return f.read().strip().replace("\n", ", ")



xy_pattern = r"x=(-?\d*\.\d+)mm_y=(-?\d*\.\d+)mm" # gets x,y of each event
pitch_pattern = r'pitch=(.*?)_'

path_to_dataset = r'/media/amir/9C33-6BBD/NEXT_work/Geant4/nexus/' + \
                  r'small_cluster_hitpoints_dataset/SquareFiberMacrosAndOutputs'


geometry_dirs = os.listdir(path_to_dataset)
for geometry in geometry_dirs:
    folder = path_to_dataset + "/" + geometry
    os.chdir(folder)
    print(f'Current working directory:\n{os.getcwd()}',end='\n')
    
    Load_SR  = f'{folder}/Sensor_Response_2_events_raw'
    Load_SR_interpolate  = f'{folder}/Sensor_Response_2_events_interpolated'
    
    for i in range(int(len(glob.glob(Load_SR+r'/*.npy'))/3)):
        
        two_events_raw_hist = np.load(glob.glob(Load_SR + f'/*id={i}_hist.npy')[0])
        two_events_x_bins = np.load(glob.glob(Load_SR + f'/*id={i}_x_bins.npy')[0])
        two_events_y_bins = np.load(glob.glob(Load_SR + f'/*id={i}_y_bins.npy')[0])
    
        two_events_interpolated_hist = np.load(glob.glob(
            Load_SR_interpolate + f'/*id={i}_hist.npy')[0])
        two_events_x_fine = np.load(glob.glob(Load_SR_interpolate +
                                              f'/*id={i}_x_fine.npy')[0])
        two_events_y_fine = np.load(glob.glob(Load_SR_interpolate +
                                              f'/*id={i}_y_fine.npy')[0])
        
        title_file = glob.glob(Load_SR_interpolate + f'/*id={i}_hist_details.txt')[0]
        title = file_content_to_string(title_file)
    
    
        fig, (ax0,ax1) = plt.subplots(1,2,figsize=(17,7), dpi=300)
        im = ax0.imshow(two_events_raw_hist,interpolation='none', origin='lower',
                        extent=[two_events_x_fine.min(), two_events_x_fine.max(),
                                two_events_y_fine.min(), two_events_y_fine.max()])
        ax0.set_title('Sensor response')
        ax0.set_xlabel('x [mm]')
        ax0.set_ylabel('y [mm]')
        divider = make_axes_locatable(ax0)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
    
    
        im = ax1.imshow(two_events_interpolated_hist, origin='lower',
                        extent=[two_events_x_fine.min(), two_events_x_fine.max(),
                                two_events_y_fine.min(), two_events_y_fine.max()])
        
        ax1.set_title('Sensor response after interpolation')
        ax1.set_xlabel('x[mm]')
        ax1.set_ylabel('y [mm]')
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        fig.suptitle(title)
        fig.tight_layout()
        plt.show()

        save_SR_plots = f'{folder}/SR_plots'
        if not os.path.isdir(save_SR_plots):
            os.mkdir(save_SR_plots)
        save_path = save_SR_plots + f'/pair_id={i}'
        fig.savefig(save_path, format='jpg', dpi=300, bbox_inches='tight' )




    
    
# In[6]
'''
RL deconvolution calculation and save
'''
section_clock_start = time.process_time()

def richardson_lucy(image, psf, iterations=75, iter_thr=0.):
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
    lowest_value = 4.94E-324
    
    for i in range(iterations):
        x = convolve_method(im_deconv, psf, 'same')
        np.place(x, x==0, eps) ### Protection against 0 value
        relative_blur = image / x
        im_deconv *= convolve_method(relative_blur, psf_mirror, 'same')
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_diff = np.sum(np.divide(((im_deconv/im_deconv.max() - ref_image)**2), ref_image))
        if i>50:
            print(f'{i},rel_diff={rel_diff}')     
        if rel_diff < iter_thr: ### Break if a given threshold is reached.
            break   
        ref_image = im_deconv/im_deconv.max()     
        ref_image[ref_image<=lowest_value] = lowest_value      
        rel_diff_checkout = rel_diff # Store last value of rel_diff before it becomes NaN
    return rel_diff_checkout, i, im_deconv


def remove_digits_beyond_first_significant(num):
    '''
    This function cuts a float when when the first significant digit is shown
    after the decimal point.
    for example: 
        if given 0.00001023, returns 0.00001
    '''
    new_string = ''
    string = '{:.15f}'.format(num)
    for counter,element in enumerate(string):
        if element == '0' or element == '.':
            new_string = new_string + element
        else: break
    return float(new_string + element) 


















