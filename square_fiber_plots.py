#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 20:17:15 2024

@author: amir
"""

'''
plots for square_fiber.py
'''


import os
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.style.use('classic')
from tqdm import tqdm
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
import glob
import re
# from invisible_cities.reco.deconv_functions     import richardson_lucy
import random

import sys
sys.path.append('/home/amir/Products/geant4/geant4-v11.0.1/MySims/nexus')
from square_fiber_functions import *


# Global settings #
n_sipms = 25 # DO NOT CHANGE THIS VALUE
n_sipms_per_side = (n_sipms-1)/2
size = 250
bins = size


path_to_dataset = '/media/amir/Extreme Pro/SquareFiberDatabase'
# path_to_dataset = '/media/amir/Extreme Pro/SquareFiberDatabaseExpansion'

# List full paths of the Geant4_PSF_events folders inside SquareFiberDatabase
geometry_dirs = [os.path.join(path_to_dataset, d) for d in os.listdir(path_to_dataset)
                 if os.path.isdir(os.path.join(path_to_dataset, d))]
delta_r = '\u0394' + 'r'

def sort_ascending_dists_P2Vs(dists, P2Vs):
    combined = list(zip(dists, P2Vs))
    combined.sort(key=lambda x: x[0])
    sorted_dists, sorted_P2Vs = zip(*combined)
    sorted_dists, sorted_P2Vs = list(sorted_dists), list(sorted_P2Vs)
    return sorted_dists, sorted_P2Vs


# In[0]
### Figure for Lior , with coating, with TPB ###
# TPB, plot of the absorbed WLS blue photons in sipm vs fiber coating reflectivity
# geant seed was set to 10002
n_photons = 100000

vikuiti_ref = [95, 96, 97, 98, 99, 99.9] # % reflection, in geant optical materials file
SiPM_hits = [2475, 3241,  4366, 6828 , 11546 , 25786 ] # Random XY in TPB center z=-0.0023


SiPM_hits = [x / n_photons for x in SiPM_hits]

fig, ax = plt.subplots(1,figsize=(10,8),dpi=600)
fig.patch.set_facecolor('white')
ax.plot(vikuiti_ref, SiPM_hits, '-*', color='rebeccapurple',linewidth=3,
         markersize=12)
text = ("100K VUV photons facing forward at 7.21 eV per reflectivity\n" + \
       "Random XY generation on TPB face")
ax.text(95.1, 0.27, text, bbox=dict(facecolor='rebeccapurple', alpha=0.5),
        fontsize=15)
ax.set_xlabel("Fiber Coating Reflectivity [%]")
ax.set_ylabel("Fraction of photons absorbed in SiPM")
ax.grid()
ax.set_xlim(min(vikuiti_ref) - 0.1, max(vikuiti_ref) + 0.1)
plt.gca().ticklabel_format(style='plain', useOffset=False)
# fig.suptitle("Absorbed WLS photons in SiPM vs fiber coating reflectivity")
# save_path = r'/home/amir/Desktop/Sipm_hits_vs_coating_reflectivity_TPB.jpg'
# plt.savefig(save_path, dpi=600)
plt.show()

# In[0.1]
### Figure for Lior , with coating, without TPB ###
# No TPB, plot of the absorbed WLS blue photons in sipm vs fiber coating reflectivity
# geant seed was set to 10002

n_photons = 100000
# DO NOT CHANGE !!

vikuiti_ref = [95, 96, 97, 98, 99, 99.9] # % reflection, in geant optical materials file
SiPM_hits = [ 24721, 29228 , 34895 , 41901, 50794, 60509] # Random XY pn fiber face (inside)

SiPM_hits = [x / n_photons for x in SiPM_hits]

fig, ax = plt.subplots(1,figsize=(10,8),dpi=600)
fig.patch.set_facecolor('white')
plt.plot(vikuiti_ref, SiPM_hits, '-*b',linewidth=3,markersize=12)
text = ("100K blue photons facing forward at 2.883 eV per reflectivity\n" + \
       "Random XY generation on PMMA fiber face")
plt.text(95.1, 0.6, text, bbox=dict(facecolor='blue', alpha=0.5),
         fontsize=15)
plt.xlabel("Fiber Coating Reflectivity [%]")
plt.ylabel("Fraction of photons absorbed in SiPM")
plt.grid()
plt.xlim(min(vikuiti_ref) - 0.1, max(vikuiti_ref) + 0.1)
plt.gca().ticklabel_format(style='plain', useOffset=False)
# fig.suptitle("Absorbed photons in SiPM vs fiber coating reflectivity")
# save_path = r'/home/amir/Desktop/Sipm_hits_vs_coating_reflectivity_no_TPB.jpg'
# plt.savefig(save_path, dpi=600)
plt.show()

# In[0.2]
# plot both fiber and fiber+TPB on a single graph

n_photons = 100000

# no TPB data
vikuiti_ref = [95, 96, 97, 98, 99, 99.9] # % reflection, in geant optical materials file
SiPM_hits = [ 24721, 29228 , 34895 , 41901, 50794, 60509] # Random XY pn fiber face (inside)
SiPM_hits_NO_TPB = [x / n_photons for x in SiPM_hits]

# TPB data 
SiPM_hits = [2475, 3241,  4366, 6828 , 11546 , 25786 ] # Random XY in TPB center z=-0.0023

SiPM_hits_TPB = [x / n_photons for x in SiPM_hits]

fig, ax = plt.subplots(1,figsize=(9,7),dpi=600)
fig.patch.set_facecolor('white')
ax.plot(vikuiti_ref, SiPM_hits_NO_TPB, '-*g',linewidth=3,markersize=12,label="Fiber only")
ax.plot(vikuiti_ref, SiPM_hits_TPB, '-*r', linewidth=3,
         markersize=12, label="fiber+TPB")

text = ("100K photons facing forward vs. reflectivity\n" + \
       "Random XY generation on external face")

ax.text(95.1, 0.63, text, bbox=dict(facecolor='blue', alpha=0.5),
        fontsize=15)
ax.set_xlabel("Fiber Coating Reflectivity [%]",fontweight='bold')
ax.set_ylabel("Fraction of photons absorbed in SiPM",fontweight='bold')
ax.grid()
ax.set_xlim(min(vikuiti_ref) - 0.1, max(vikuiti_ref) + 0.1)
plt.gca().ticklabel_format(style='plain', useOffset=False)
plt.legend(fontsize=13)
# fig.suptitle("Absorbed WLS photons in SiPM vs fiber coating reflectivity")
# save_path = r'/home/amir/Desktop/Sipm_hits_vs_coating_reflectivity_TPB.jpg'
# plt.savefig(save_path, dpi=600)
plt.show()


# In[0.3]
# How fiber length affects the light fraction detected , 98 % vikuiti reflectivity
# 2.2 um TPB thickness
# geant seed was set to 10002

n_photons = 100000


# New results 1.4.24, NO holder here
fiber_length = np.arange(5,105,5).tolist() # mm

# insert additional data point
# fiber_length.insert(0,0.01) # no tpb gives 81641, with tpb gives 38322
fiber_length.insert(0,1) # no tpb gives 61522, with tpb gives 31651

# DO NOT CHANGE the following fixed values!!
# here z=-0.0023 mm, tiny bit outside the TPB
SiPM_hits_with_TPB = [ 31651, 24846, 20754, 17705, 15620, 14166, 12725 ,
             11763, 10581, 9785, 9295, 8435,
             7735, 7266, 6828, 6351, 6037, 5737,
             5281, 4913, 4660]

# here z=-0.0001 mm, tiny bit outside the fiber
SiPM_hits_without_TPB = [ 61522, 60057, 58224, 56775, 55260, 53510,  51941,
             50504, 49392, 48701, 46655, 45600,
             44078, 43071, 42013, 40846, 39746, 38652,
             37777, 36881, 35650]

SiPM_hits_with_TPB = [x / n_photons for x in SiPM_hits_with_TPB]
SiPM_hits_without_TPB = [x / n_photons for x in SiPM_hits_without_TPB]


fig, ax1 = plt.subplots(figsize=(10, 8), dpi=600)
fig.patch.set_facecolor('white')

# Plotting on the primary y-axis
ax1.plot(fiber_length, SiPM_hits_with_TPB, '-ok', linewidth=3, markersize=10, label='Fiber+TPB')
ax1.plot(fiber_length, SiPM_hits_without_TPB, '--^k', linewidth=3, markersize=10, label='Fiber only')
ax1.set_xlabel("Fiber length [mm]",fontsize=15,fontweight='bold')
ax1.set_ylabel("Fraction of photons absorbed in SiPM",fontsize=15,fontweight='bold')
ax1.tick_params(axis='y', labelcolor='black')

# Adding the text box
text = ("100K photons facing forward vs. length" +
        "\nRandom XY generation on external face" +
        "\nFiber coating reflectivity set to 98%")
ax1.text(12, 0.615, text, bbox=dict(facecolor='magenta', alpha=0.5), fontsize=14)

# Setting x and primary y-axis scales and labels
ax1.set_xticks(np.arange(0, 110, 10))
ax1.set_xlim([0,None])
ax1.grid()
ax1.ticklabel_format(style='plain', useOffset=False)

# Creating a secondary y-axis
ax2 = ax1.twinx()

# Match the y-range and y-ticks of ax2 to ax1
ax2.set_ylim(ax1.get_ylim())  # Set y-range of ax2 to match ax1
ax2.set_yticks(ax1.get_yticks())  # Set y-ticks of ax2 to match ax1

# Plotting the ratio on the secondary y-axis
ax2.plot(fiber_length, np.divide(SiPM_hits_with_TPB, SiPM_hits_without_TPB), '-sr', linewidth=3, markersize=10, label='Fiber+TPB / Fiber Ratio')
ax2.set_ylabel("Fiber+TPB / Fiber Ratio", color='red',fontsize=15,fontweight='bold')
ax2.tick_params(axis='y', labelcolor='red')

# Adding a legend that includes all plots
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, fontsize=12)

plt.show()


# In[1]
### PSF comparison plots ###

# profile plots, select comparison test
immersion_test = False
anode_distance_test = False
EL_gap_test = False
pitch_test = True


if immersion_test:
    fiber_immersion_choice = [0,3,6]
    pitch_choice = [10]
    el_gap_choice = [1]
    anode_distance_choice = [2.5]
    holder_thickness_choice = [10]
    color = iter(['red', 'green', 'blue'])
    marker = iter(['^', '*', 's'])
    
if anode_distance_test:
    fiber_immersion_choice = [0]
    pitch_choice = [10]
    el_gap_choice = [1]
    anode_distance_choice = [2.5,5,10]
    holder_thickness_choice = [10]
    color = iter(['red', 'green', 'blue'])
    marker = iter(['^', '*', 's'])

if EL_gap_test:
    fiber_immersion_choice = [3]
    pitch_choice = [5]
    el_gap_choice = [1,10]
    anode_distance_choice = [2.5]
    holder_thickness_choice = [10]
    color = iter(['red', 'green'])
    marker = iter(['^', '*'])
    
if pitch_test:
    fiber_immersion_choice = [6]
    pitch_choice = [5,10,15.6]
    el_gap_choice = [1]
    anode_distance_choice = [2.5]
    holder_thickness_choice = [10]
    color = iter(['red', 'green', 'blue'])
    marker = iter(['^', '*', 's'])


psf_profiles = []

# Loop through each directory to compute and store the PSF profiles
for directory in tqdm(geometry_dirs):
    geo_params = directory.split('/SquareFiberDatabase/')[-1]
    
    
    el_gap = float(re.search(r"ELGap=(-?\d+\.?\d*)mm",
                             geo_params).group(1))
    pitch = float(re.search(r"pitch=(-?\d+\.?\d*)mm",
                            geo_params).group(1))
    anode_distance = float(re.search(r"distanceAnodeHolder=(-?\d+\.?\d*)mm",
                                     geo_params).group(1))
    holder_thickness = float(re.search(r"holderThickness=(-?\d+\.?\d*)mm",
                                       geo_params).group(1))
    fiber_immersion = float(re.search(r"distanceFiberHolder=(-?\d+\.?\d*)mm",
                                      geo_params).group(1))
    fiber_immersion = 5 - fiber_immersion

    if (el_gap not in el_gap_choice or
        pitch not in pitch_choice or
        anode_distance not in anode_distance_choice or
        fiber_immersion not in fiber_immersion_choice or
        holder_thickness not in holder_thickness_choice):
        continue
    
    PSF = np.load(directory + '/PSF.npy')
    psf_profile = PSF.mean(axis=0)
    psf_profiles.append((psf_profile))

max_psf_profile = max([np.max(profile) for profile in psf_profiles])


fig, ax = plt.subplots(figsize=(9,7), dpi = 600)
fig.patch.set_facecolor('white')
for directory in tqdm(geometry_dirs):
    
    geo_params = directory.split('/SquareFiberDatabase/')[-1]
    
    
    el_gap = float(re.search(r"ELGap=(-?\d+\.?\d*)mm",
                             geo_params).group(1))
    pitch = float(re.search(r"pitch=(-?\d+\.?\d*)mm",
                            geo_params).group(1))
    anode_distance = float(re.search(r"distanceAnodeHolder=(-?\d+\.?\d*)mm",
                                     geo_params).group(1))
    holder_thickness = float(re.search(r"holderThickness=(-?\d+\.?\d*)mm",
                                       geo_params).group(1))
    fiber_immersion = float(re.search(r"distanceFiberHolder=(-?\d+\.?\d*)mm",
                                      geo_params).group(1))
    fiber_immersion = 5 - fiber_immersion

    if (el_gap not in el_gap_choice or
        pitch not in pitch_choice or
        anode_distance not in anode_distance_choice or
        fiber_immersion not in fiber_immersion_choice or
        holder_thickness not in holder_thickness_choice):
        continue
    
    # normalize other PSFs to max PSF

    PSF = np.load(directory + '/PSF.npy')
    psf_profile = PSF.mean(axis=0)
   
    normalized_profile = psf_profile * (max_psf_profile/np.max(psf_profile))
    x_vec = np.arange(-PSF.shape[0]/2, PSF.shape[0]/2)+0.5 # shift to bin center
    
    if immersion_test:
        label = f'Fiber immersion={int(fiber_immersion)} mm'
        
        text = (f'Pitch={pitch_choice[-1]} mm' +
                f'\nEL gap={el_gap_choice[-1]} mm' +
                f'\nAnode distance={anode_distance_choice[-1]} mm')
        
        ax.text(0.03, 0.97, text, ha='left', va='top',
                 transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.5,
                           boxstyle='round,pad=0.3'),
                 fontsize=12, weight='bold', linespacing=1.5)
        
    if anode_distance_test:
        label = f'Anode distance={anode_distance} mm'
        
        text = (f'Pitch={pitch_choice[-1]} mm' +
                f'\nEL gap={el_gap_choice[-1]} mm' +
                f'\nFiber immersion={fiber_immersion_choice[-1]} mm')
        
        ax.text(0.03, 0.97, text, ha='left', va='top',
                 transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.5,
                           boxstyle='round,pad=0.3'),
                 fontsize=12, weight='bold', linespacing=1.5)
        
    if EL_gap_test:
        label = f'EL gap={int(el_gap)} mm'
        
        text = (f'Pitch={pitch_choice[-1]} mm' +
                f'\nFiber immersion={fiber_immersion_choice[-1]} mm' +
                f'\nAnode distance={anode_distance_choice[-1]} mm')
        
        ax.text(0.03, 0.97, text, ha='left', va='top',
                 transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.5,
                           boxstyle='round,pad=0.3'),
                 fontsize=12, weight='bold', linespacing=1.5)
        
    if pitch_test:
        label = f'pitch={int(pitch)} mm'
        
        text = (f'EL gap={el_gap_choice[-1]} mm' +
                f'\nFiber immersion={fiber_immersion_choice[-1]} mm' +
                f'\nAnode distance={anode_distance_choice[-1]} mm')
        
        ax.text(0.03, 0.97, text, ha='left', va='top',
                 transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.5,
                           boxstyle='round,pad=0.3'),
                 fontsize=12, weight='bold', linespacing=1.5)
        
    ax.plot(x_vec, normalized_profile, color=next(color), ls='-', 
            alpha=0.7, label=label, linewidth=3, markersize=7)


plt.xlabel('x [mm]', fontsize=14)
plt.ylabel('Photon hits', fontsize=14)
# plt.xlim([0,30])
plt.grid()
ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
plt.legend(loc='upper right',fontsize=12)
fig.suptitle(f'PSF profiles',fontsize=15, fontweight='bold')
plt.show()

# In[1.1]
# PSF image plots - show square PSF
pitch = 5
el_gap = 1
anode_distance = 2.5

geo_dir_1 = ('/media/amir/Extreme Pro/SquareFiberDatabase/' +
            f'ELGap={el_gap}mm_pitch={pitch}mm_distanceFiberHolder=5mm_' +
            f'distanceAnodeHolder={anode_distance}mm_holderThickness=10mm')

geo_dir_2 = ('/media/amir/Extreme Pro/SquareFiberDatabase/' +
            f'ELGap={el_gap}mm_pitch={pitch}mm_distanceFiberHolder=2mm_' +
            f'distanceAnodeHolder={anode_distance}mm_holderThickness=10mm')

geo_dir_3 = ('/media/amir/Extreme Pro/SquareFiberDatabase/' +
            f'ELGap={el_gap}mm_pitch={pitch}mm_distanceFiberHolder=-1mm_' +
            f'distanceAnodeHolder={anode_distance}mm_holderThickness=10mm')

geo_dirs = [geo_dir_1,geo_dir_2,geo_dir_3]

fig, ax = plt.subplots(1,3, figsize=(18,6), dpi = 200)
fig.patch.set_facecolor('white')

for i,geo_dir in enumerate(geo_dirs):
    
    geo_params = geo_dir.split('/SquareFiberDatabase/')[-1]
    
    fiber_immersion = float(re.search(r"distanceFiberHolder=(-?\d+\.?\d*)mm",
                                      geo_params).group(1))
    fiber_immersion = 5 - fiber_immersion
    
    PSF = np.load(geo_dir + '/PSF.npy')
    PSF_sum = np.sum(PSF)
    PSF = PSF[25:75,25:75]
    PSF_size = PSF.shape[0] / 2
    
    title = f'Immersion = {int(fiber_immersion)} mm'
    im = ax[i].imshow(PSF, extent=[-PSF_size,PSF_size,-PSF_size,PSF_size]
                      ,vmin = 5, vmax=485969.0)
    
    divider = make_axes_locatable(ax[i])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar_min = np.min(PSF)
    cbar_max = np.max(PSF)
    print(f'cbar_min = {cbar_min}, cbar_max = {cbar_max}')
    cbar = plt.colorbar(im, cax=cax)
    
    # Format colorbar tick labels in scientific notation
    formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    cbar.ax.yaxis.set_major_formatter(formatter)
    
    ax[i].set_xlabel('x [mm]', fontsize=17)
    ax[i].set_ylabel('y [mm]', fontsize=17)
    ax[i].set_title(title, fontsize=18, fontweight='bold')
    # print(f'PSF sum for immersion={int(fiber_immersion)} mm is {int(PSF_sum)}')
    

title = (f'Geometry parameters: pitch = {pitch} mm, EL gap = {el_gap} mm,' +
         f'anode distance = {anode_distance} mm')
fig.suptitle(title,fontsize=22, fontweight='bold')
fig.tight_layout()
plt.show()


## absorption length for Silicon
# E =  [1.5498 ,  1.937 ,   2.479 , 
#   3.099 ,   3.874,   4.959]   

# abs_length = [11.764,   3.289,   0.9,
#   0.105,     0.0078125,     0.00543]
# plt.plot(E,abs_length)
# plt.yticks(np.arange(0,12,0.5))
# plt.axvline(2.883,color='red')
# plt.xlabel('E [ev]')
# plt.ylabel('abs length [um]')
# plt.show()


# In[2]
### GRAPHS - compare P2Vs of different geometries ###

import matplotlib.cm as cm
# Number of unique colors needed
num_colors = len(geometry_dirs)

cmap = cm.get_cmap('hsv', num_colors)

# Generate colors from the colormap
colors = [cmap(i) for i in range(num_colors)]
random.shuffle(colors)
markers = ['^', '*', 's', 'o', 'x', '+', 'D']

# Parameter choices to slice to
fiber_immersion_choice = [0,3,6]
pitch_choice = [15.6]
el_gap_choice = [10]
anode_distance_choice = [10]
holder_thickness_choice = [10]
count = 0
SHOW_ORIGINAL_SPACING = False
if SHOW_ORIGINAL_SPACING is False:
    custom_spacing = 4
POLYNOMIAL_FIT = False


fig, ax = plt.subplots(figsize=(10,7), dpi = 600)

for i, geo_dir in tqdm(enumerate(geometry_dirs)):
    
    working_dir = geo_dir + r'/P2V'
    geo_params = geo_dir.split('/SquareFiberDatabase/')[-1]
    
    
    el_gap = float(re.search(r"ELGap=(-?\d+\.?\d*)mm",
                             geo_params).group(1))
    pitch = float(re.search(r"pitch=(-?\d+\.?\d*)mm",
                            geo_params).group(1))
    anode_distance = float(re.search(r"distanceAnodeHolder=(-?\d+\.?\d*)mm",
                                     geo_params).group(1))
    holder_thickness = float(re.search(r"holderThickness=(-?\d+\.?\d*)mm",
                                       geo_params).group(1))
    fiber_immersion = float(re.search(r"distanceFiberHolder=(-?\d+\.?\d*)mm",
                                      geo_params).group(1))
    fiber_immersion = 5 - fiber_immersion
    
    
    if (el_gap not in el_gap_choice or
        pitch not in pitch_choice or
        anode_distance not in anode_distance_choice or
        fiber_immersion not in fiber_immersion_choice or
        holder_thickness not in holder_thickness_choice):
        # print(i)
        continue
    
    
    dir_data = glob.glob(working_dir + '/*/*.txt')
    dists = []
    P2Vs = []
        
    if SHOW_ORIGINAL_SPACING:
        for data in dir_data:
            dist, P2V = np.loadtxt(data)
            if P2V > 100 or P2V == float('inf'):
                P2V = 100
            dists.append(dist)
            P2Vs.append(P2V)
            
        # zip and sort in ascending order
        sorted_dists, sorted_P2Vs = sort_ascending_dists_P2Vs(dists,P2Vs)

            
            
    else:
        averaged_dists = []
        averaged_P2Vs = []
        for j in range(0, len(dir_data), 2):  # Step through dir_data two at a time
            data_pairs = dir_data[j:j+2]  # Get the current pair of data files
            dists = []
            P2Vs = []
            for data in data_pairs:
                dist, P2V = np.loadtxt(data)
                if P2V > 100 or P2V == float('inf'):
                    P2V = 100
                dists.append(dist)
                P2Vs.append(P2V)
                # print(f'data_pair dist={dist}, P2V={P2V}')
            
            # Only proceed if we have a pair, to avoid index out of range errors
            if len(dists) == 2 and len(P2Vs) == 2:
                averaged_dist = sum(dists) / 2
                averaged_P2V = sum(P2Vs) / 2
                # print(f'averaged_dist={averaged_dist}, averaged_P2V={averaged_P2V}',end='\n\n')
                
                '''
                the 0.25 acts to fix the fact the dist dirs are sorted
                to 0.5mm intervals of source spaceing. So when you just average
                Them out, you get a missing 0.25 missing.
                Example:
                    dist dirs are:
                        3+3.5+4+4.5  -> avg is (3+3.5+4+4.5)/4=3.75, however 
                        these dist dirs show an actual range of 3-5 mm, so the 
                        real average should be 4, hence the addition of 0.25.
                        Same logic works for 2 or 3 dist dirs.
                '''
                averaged_dists.append(averaged_dist+0.25)
                averaged_P2Vs.append(averaged_P2V)
        
        # zip and sort in ascending order
        sorted_dists, sorted_P2Vs = sort_ascending_dists_P2Vs(averaged_dists,
                                                              averaged_P2Vs)
        

    label = (f'EL gap={el_gap}mm,pitch={pitch}mm,fiber immersion={fiber_immersion}mm,' +
             f'anode distance={anode_distance}mm,holder thickness={holder_thickness}mm')
    
    if POLYNOMIAL_FIT:
        # Fit the data to a 2nd degree polynomial
        coefficients = np.polyfit(sorted_dists, sorted_P2Vs, 2)
        
        # Generate a polynomial function from the coefficients
        polynomial = np.poly1d(coefficients)
        
        # Generate x values for the polynomial curve (e.g., for plotting)
        sorted_dists_fit = np.linspace(sorted_dists[0], sorted_dists[-1], 100)
        
        # Generate y values for the polynomial curve
        sorted_P2V_fit = polynomial(sorted_dists_fit)
        
        ax.plot(sorted_dists_fit, sorted_P2V_fit, color=colors[i], ls='-', 
                    alpha=0.5, label=label)
        
    else:
        ax.plot(sorted_dists, sorted_P2Vs, color=colors[i], ls='-', 
                    marker=random.choice(markers),alpha=0.5, label=label)
        
    count += 1
        
xlim = 30
plt.axhline(y=1,color='red',alpha=0.7)
plt.xticks(np.arange(0,xlim),rotation=45, size=10)
plt.xlabel('Distance [mm]', fontweight='bold')
plt.ylabel('P2V', fontweight='bold')
plt.ylim([-1,5])
plt.xlim([0,xlim])
plt.grid()
plt.legend(loc='upper left', bbox_to_anchor=(1, 1),
           title='Datasets', fontsize='small', ncol=1)
# fig.suptitle(f'Fiber immersion = {immersion_choice}mm')
plt.show()


# In[2]
### GRAPHS - Find optimal anode distance FOR NEXT SETUP ###  

# Parameter choices to slice to - DO NOT CHANGE FOR CURRENT DATASET!
fiber_immersion_choice = [3,6]
pitch_choice = [15.6]
el_gap_choice = [10]
anode_distance_choice = [2.5,5,7.5,10,12.5,15,17.5,20]
holder_thickness_choice = [10]
dists = np.arange(16,31,0.5)


for dist in tqdm(dists):
    
    anode_distance_immersion_3, anode_distance_immersion_6 = [], []
    P2Vs_immersion_3, P2Vs_immersion_6 = [], [] 
    fig, ax = plt.subplots(figsize=(10,7), dpi = 600)
    fig.patch.set_facecolor('white')
    
    for i, geo_dir in tqdm(enumerate(geometry_dirs)):
        
        working_dir = geo_dir + r'/P2V'
        geo_params = geo_dir.split('/SquareFiberDatabase/')[-1]
        
        el_gap = float(re.search(r"ELGap=(-?\d+\.?\d*)mm",
                                 geo_params).group(1))
        pitch = float(re.search(r"pitch=(-?\d+\.?\d*)mm",
                                geo_params).group(1))
        anode_distance = float(re.search(r"distanceAnodeHolder=(-?\d+\.?\d*)mm",
                                         geo_params).group(1))
        holder_thickness = float(re.search(r"holderThickness=(-?\d+\.?\d*)mm",
                                           geo_params).group(1))
        fiber_immersion = float(re.search(r"distanceFiberHolder=(-?\d+\.?\d*)mm",
                                          geo_params).group(1))
        fiber_immersion = 5 - fiber_immersion
        
        
        if (el_gap not in el_gap_choice or
            pitch not in pitch_choice or
            anode_distance not in anode_distance_choice or
            fiber_immersion not in fiber_immersion_choice or
            holder_thickness not in holder_thickness_choice):
            continue
        
        dist = 24
        user_chosen_dir = find_subdirectory_by_distance(working_dir, dist)
        dir_data = glob.glob(user_chosen_dir + '/*.txt')[-1]
        _, P2V = np.loadtxt(dir_data)
    
            
        if fiber_immersion==fiber_immersion_choice[0]:
            anode_distance_immersion_3.append(anode_distance)
            P2Vs_immersion_3.append(P2V)
    
            
        if fiber_immersion==fiber_immersion_choice[1]:
            anode_distance_immersion_6.append(anode_distance)
            P2Vs_immersion_6.append(P2V)
    
                
    # zip and sort in ascending order
    sorted_dists_3, sorted_P2Vs_3 = sort_ascending_dists_P2Vs(anode_distance_immersion_3,
                                                          P2Vs_immersion_3)
    sorted_dists_6, sorted_P2Vs_6 = sort_ascending_dists_P2Vs(anode_distance_immersion_6,
                                                          P2Vs_immersion_6)
    
    
    ax.plot(sorted_dists_3, sorted_P2Vs_3, color='green', ls='-',linewidth=3, 
                marker='^', markersize=10, alpha=0.8,
                label=f"immersion={fiber_immersion_choice[0]}mm")           
    ax.plot(sorted_dists_6, sorted_P2Vs_6, color='red', ls='-', linewidth=3, 
                marker='*',markersize=10, alpha=0.8,
                label=f"immersion={fiber_immersion_choice[1]}mm") 
    
    plt.xlabel('Anode distance [mm]', fontweight='bold')
    plt.ylabel('P2V', fontweight='bold')
    plt.grid()
    plt.legend(loc='upper left',fontsize=10)
    
    text = ( f'Immersion={fiber_immersion_choice[0]} mm' +
             f'\nEL gap={el_gap_choice[-1]} mm' +
             f'\nPitch={pitch_choice[-1]} mm')
    
    plt.text(0.04, 0.95, text, ha='left', va='top',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.5,boxstyle='round,pad=0.3'),
             fontsize=13, weight='bold', linespacing=1.5)  # Adjust this value to increase spacing between lines, if necessary
    fig.suptitle(f'{delta_r} = {dist} [mm]',fontsize=15, fontweight='bold')
    fig.tight_layout()
    plt.show()

# In[3]
### GRAPHS - Find optimal pitch ###  

def custom_polyfit(sorted_dists, sorted_P2Vs, power=2):
    
    sorted_dists, sorted_P2Vs = np.array(sorted_dists), np.array(sorted_P2Vs)
    
    for i, p2v in enumerate(sorted_P2Vs):
        if p2v > 1:
            last_index_with_P2V_1 = i - 1
            break
    else: # who knew you can have an else statement for a "for" loop ?!
        last_index_with_P2V_1 = len(sorted_P2Vs) - 1

    # last_index_with_P2V_1 = np.max(np.where(sorted_P2Vs == 1)[0])
    
    # Use data from this index onwards for fitting
    relevant_dists = sorted_dists[last_index_with_P2V_1+1:]
    relevant_P2Vs = sorted_P2Vs[last_index_with_P2V_1+1:]
    
    # Fit a polynomial of degree 2 (a parabola) to the relevant part of the data
    coefficients = np.polyfit(relevant_dists, relevant_P2Vs, power)
    
    # Ensure the leading coefficient is positive for a "smiling" parabola
    if coefficients[0] < 0:
        coefficients[0] = - coefficients[0]
    
    # Generate a polynomial function from the coefficients
    polynomial = np.poly1d(coefficients)
    
    # Generate x values for the polynomial curve (e.g., for plotting)
    sorted_dists_fit = np.linspace(relevant_dists[0], relevant_dists[-1], 100)
    
    # Generate y values for the polynomial curve
    sorted_P2V_fit = polynomial(sorted_dists_fit)
    
    return sorted_dists_fit, sorted_P2V_fit

# Parameter choices to slice to - DO NOT CHANGE FOR CURRENT DATASET!
fiber_immersion_choice = [3]
pitch_choice = [5,10,15.6]
el_gap_choice = [1]
anode_distance_choice = [2.5]
holder_thickness_choice = [10]
color = iter(['red', 'green', 'blue'])
marker = iter(['^', '*', 's'])

SHOW_ORIGINAL_SPACING = False
if SHOW_ORIGINAL_SPACING is False:
    custom_spacing = 2
POLYNOMIAL_FIT = False

fig, ax = plt.subplots(figsize=(10,7), dpi = 600)
fig.patch.set_facecolor('white')

for i, geo_dir in tqdm(enumerate(geometry_dirs)):
    
    working_dir = geo_dir + r'/P2V'
    geo_params = geo_dir.split('/SquareFiberDatabase/')[-1]
    
    
    el_gap = float(re.search(r"ELGap=(-?\d+\.?\d*)mm",
                             geo_params).group(1))
    pitch = float(re.search(r"pitch=(-?\d+\.?\d*)mm",
                            geo_params).group(1))
    anode_distance = float(re.search(r"distanceAnodeHolder=(-?\d+\.?\d*)mm",
                                     geo_params).group(1))
    holder_thickness = float(re.search(r"holderThickness=(-?\d+\.?\d*)mm",
                                       geo_params).group(1))
    fiber_immersion = float(re.search(r"distanceFiberHolder=(-?\d+\.?\d*)mm",
                                      geo_params).group(1))
    fiber_immersion = 5 - fiber_immersion
    
    # continue if geometry is not in search space
    if (el_gap not in el_gap_choice or
        pitch not in pitch_choice or
        anode_distance not in anode_distance_choice or
        fiber_immersion not in fiber_immersion_choice or
        holder_thickness not in holder_thickness_choice):
        # print(i)
        continue
    
    
    dir_data = glob.glob(working_dir + '/*/*.txt')
    dists = []
    P2Vs = []
        
    if SHOW_ORIGINAL_SPACING:
        for data in dir_data:
            dist, P2V = np.loadtxt(data)
            if P2V > 100 or P2V == float('inf'):
                P2V = 100
            dists.append(dist)
            P2Vs.append(P2V)
            
        # zip and sort in ascending order
        sorted_dists, sorted_P2Vs = sort_ascending_dists_P2Vs(dists,P2Vs)
       
    else:
        averaged_dists = []
        averaged_P2Vs = []
        for j in range(0, len(dir_data), custom_spacing):  # Step through dir_data two at a time
            data_pairs = dir_data[j:j+custom_spacing]  # Get the current pair of data files
            dists = []
            P2Vs = []
            for data in data_pairs:
                dist, P2V = np.loadtxt(data)
                if P2V > 100 or P2V == float('inf'):
                    P2V = 100
                dists.append(dist)
                P2Vs.append(P2V)
                # print(f'data_pair dist={dist}, P2V={P2V}')
            
            # this part is to avoid index out of range errors
            if len(dists) == custom_spacing and len(P2Vs) == custom_spacing:
                averaged_dist = sum(dists) / custom_spacing
                averaged_P2V = sum(P2Vs) / custom_spacing
                # print(f'averaged_dist={averaged_dist}, averaged_P2V={averaged_P2V}',end='\n\n')
                
                '''
                the 0.25 acts to fix the fact the dist dirs are sorted
                to 0.5mm intervals of source spaceing. So when you just average
                Them out, you get a missing 0.25 missing.
                Example:
                    dist dirs are:
                        3+3.5+4+4.5  -> avg is (3+3.5+4+4.5)/4=3.75, however 
                        these dist dirs show an actual range of 3-5 mm, so the 
                        real average should be 4, hence the addition of 0.25.
                        Same logic works for 2 or 3 dist dirs.
                '''
                averaged_dists.append(averaged_dist+0.25)
                averaged_P2Vs.append(averaged_P2V)
        
        # zip and sort in ascending order
        sorted_dists, sorted_P2Vs = sort_ascending_dists_P2Vs(averaged_dists,
                                                              averaged_P2Vs)
        
    if POLYNOMIAL_FIT:
        # custom fit a polynomial of 2nd degree to data
        sorted_dists_fit, sorted_P2V_fit = custom_polyfit(sorted_dists,
                                                          sorted_P2Vs)
        
        
        this_color = next(color)
        this_marker = next(marker)
        # Plot original data
        ax.plot(sorted_dists, sorted_P2Vs, color=this_color, ls='-', 
                    alpha=0.8, label=f'pitch={pitch} mm',marker=this_marker,
                    linewidth=3, markersize=10)
        # Plot fitted data
        ax.plot(sorted_dists_fit, sorted_P2V_fit, color=this_color, ls='--',
                 alpha=0.5, label=f'{pitch} mm fitted',linewidth=3)
        
    else:
        ax.plot(sorted_dists, sorted_P2Vs, color=next(color), ls='-', 
                    marker=next(marker),alpha=0.8, label=f'pitch={pitch} mm',
                    linewidth=3, markersize=10)
        
    
plt.xlabel(f'{delta_r} [mm]', fontsize=15)
plt.ylabel('P2V', fontsize=15)
plt.grid()
plt.xlim([None,25])
# plt.ylim([0.99,2.5])
plt.legend(loc='upper left',fontsize=14)

text = (f'EL gap={el_gap_choice[-1]} mm' +
         f'\nImmersion={fiber_immersion_choice[-1]} mm' +
         f'\nAnode dist={anode_distance_choice[-1]} mm')
# fine tune box positionand parameters
if POLYNOMIAL_FIT:
    xloc, yloc = 0.14, 0.65
    
else:
    xloc, yloc = 0.02, 0.79
    
plt.text(xloc, yloc,  text, ha='left', va='top',
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.5,boxstyle='round,pad=0.3'),
         fontsize=14, weight='bold', linespacing=1.5)  # Adjust this value to increase spacing between lines, if necessary

# fig.suptitle(title,size=12)
# fig.tight_layout()
plt.show()



# In[4]
### GRAPHS - Find optimal EL gap 3x3 ###  

def dists_and_P2Vs_custom_spacing(dir_data,spacing):
    averaged_dists = []
    averaged_P2Vs = []
    for j in range(0, len(dir_data), custom_spacing):  # Step through dir_data two at a time
        data_pairs = dir_data[j:j+custom_spacing]  # Get the current pair of data files
        dists = []
        P2Vs = []
        for data in data_pairs:
            dist, P2V = np.loadtxt(data)
            if P2V > 100 or P2V == float('inf'):
                P2V = 100
            dists.append(dist)
            P2Vs.append(P2V)
            # print(f'data_pair dist={dist}, P2V={P2V}')
        
        # Only proceed if we have a pair, to avoid index out of range errors
        if len(dists) == custom_spacing and len(P2Vs) == custom_spacing:
            averaged_dist = sum(dists) / custom_spacing
            averaged_P2V = sum(P2Vs) / custom_spacing
            # print(f'averaged_dist={averaged_dist}, averaged_P2V={averaged_P2V}',end='\n\n')
            
            '''
            the 0.25 acts to fix the fact the dist dirs are sorted
            to 0.5mm intervals of source spaceing. So when you just average
            Them out, you get a missing 0.25 missing.
            Example:
                dist dirs are:
                    3+3.5+4+4.5  -> avg is (3+3.5+4+4.5)/4=3.75, however 
                    these dist dirs show an actual range of 3-5 mm, so the 
                    real average should be 4, hence the addition of 0.25.
                    Same logic works for 2 or 3 dist dirs.
            '''
            averaged_dists.append(averaged_dist+0.25)
            averaged_P2Vs.append(averaged_P2V)
    
    # zip and sort in ascending order
    sorted_dists, sorted_P2Vs = sort_ascending_dists_P2Vs(averaged_dists,
                                                          averaged_P2Vs)
    
    return sorted_dists, sorted_P2Vs

        
# Parameter choices to slice to - DO NOT CHANGE FOR CURRENT DATASET!
fiber_immersion_choice = [0,3,6]
pitch_choice = [15.6]
el_gap_choice = [1,10]
anode_distance_choice = [2.5,5,10]
holder_thickness_choice = [10]
x_left_lim = pitch_choice[0]-1 # manual option remove plot area where there's P2V of 1 

SHOW_ORIGINAL_SPACING = True
if SHOW_ORIGINAL_SPACING is False:
    custom_spacing = 2 # 2 or 4
POLYNOMIAL_FIT = False

min_sep_P2V_list = []
min_sep_dist_list = []

fig, ax = plt.subplots(3,3,figsize=(10,7), sharex=True,sharey=True,dpi=600)
fig.patch.set_facecolor('white')
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)
fig.supxlabel(f'{delta_r} [mm]',fontsize=12,fontweight='bold')
fig.supylabel('P2V',fontweight='bold')

for m,anode_dist in enumerate(anode_distance_choice):
    for n,immersion in enumerate(fiber_immersion_choice):
        
        color = iter(['red', 'green'])
        marker = iter(['^', '*'])
        ax[m,n].grid()
        ax[m,n].set_ylim([0.99,2.5])
        
        if m == 0:
            ax[m,n].set_xlabel(f'Immersion = {immersion} mm',fontsize=10,fontweight='bold')
            ax[m,n].xaxis.set_label_position('top')
        for i, geo_dir in tqdm(enumerate(geometry_dirs)):

            working_dir = geo_dir + r'/P2V'
            geo_params = geo_dir.split('/SquareFiberDatabase/')[-1]
            
            
            el_gap = float(re.search(r"ELGap=(-?\d+\.?\d*)mm",
                                     geo_params).group(1))
            pitch = float(re.search(r"pitch=(-?\d+\.?\d*)mm",
                                    geo_params).group(1))
            anode_distance = float(re.search(r"distanceAnodeHolder=(-?\d+\.?\d*)mm",
                                             geo_params).group(1))
            holder_thickness = float(re.search(r"holderThickness=(-?\d+\.?\d*)mm",
                                               geo_params).group(1))
            fiber_immersion = float(re.search(r"distanceFiberHolder=(-?\d+\.?\d*)mm",
                                              geo_params).group(1))
            fiber_immersion = 5 - fiber_immersion
            
            
            if (el_gap not in el_gap_choice or
                pitch not in pitch_choice or
                anode_distance != anode_dist or
                fiber_immersion != immersion or
                holder_thickness not in holder_thickness_choice):
                # print(i)
                continue
            
            
            dir_data = glob.glob(working_dir + '/*/*.txt')
            dists = []
            P2Vs = []
            
                
            if SHOW_ORIGINAL_SPACING:
                for data in dir_data:
                    dist, P2V = np.loadtxt(data)
                    if P2V > 100 or P2V == float('inf'):
                        P2V = 100
                    dists.append(dist)
                    P2Vs.append(P2V)
                    
                # zip and sort in ascending order
                sorted_dists, sorted_P2Vs = sort_ascending_dists_P2Vs(dists,P2Vs)
               
            else:           
                sorted_dists, sorted_P2Vs = dists_and_P2Vs_custom_spacing(dir_data,
                                            custom_spacing)
                
            if 'x_left_lim' in locals():
                filtered_indices = [i for i, dist in enumerate(sorted_dists) if dist > x_left_lim]
                sorted_dists = [sorted_dists[i] for i in filtered_indices]
                sorted_P2Vs = [sorted_P2Vs[i] for i in filtered_indices]
            
            if POLYNOMIAL_FIT:
                # custom fit a polynomial of 2nd degree to data        
                sorted_dists_fit, sorted_P2V_fit = custom_polyfit(sorted_dists,
                                                                  sorted_P2Vs,
                                                                  power=2)
                
                this_color = next(color)
                # Plot original data
                ax[m,n].plot(sorted_dists, sorted_P2Vs, color=this_color,
                             ls='-', linewidth=2, alpha=0.85)
                # Plot fitted data
                ax[m,n].plot(sorted_dists_fit, sorted_P2V_fit, color=this_color,
                             ls='--')
                
                
            else:
                this_color = next(color)
                # print(f'color={this_color},EL={el_gap}')
                ax[m,n].plot(sorted_dists, sorted_P2Vs, color=this_color,
                             ls='-', linewidth=2, alpha=0.85)
                
                
                # for 3x3 cell plot for P2V at each pitch
                min_sep_P2V_idx = next((i for i, P2V in enumerate(sorted_P2Vs) if P2V >= 1.1), None)
                
                if min_sep_P2V_idx != None:
                    min_sep_dist_val = sorted_dists[min_sep_P2V_idx]
                    # min_sep_P2V_val = sorted_P2Vs[min_sep_P2V_idx]
                else:
                    min_sep_dist_val = 0
                min_sep_dist_list.append(min_sep_dist_val)
                # min_sep_P2V_list.append(min_sep_P2V_val)
                
             
            
        text=f'Anode dist={anode_dist} mm'
        ax[m, n].text(0.05, 0.93, text, ha='left', va='top',
                      transform=ax[m, n].transAxes,
                      bbox=dict(facecolor='white', boxstyle='square,pad=0.3'),
                      fontsize=10, fontweight='bold')
        
# Manually creating dummy lines for the legend
line1, = plt.plot([], [], color='green', linestyle='-',
                  linewidth=3, label='EL gap=1 mm')
line2, = plt.plot([], [], color='red', linestyle='-',
                  linewidth=3, label='EL gap=10 mm')

# Creating the legend manually with the dummy lines
fig.legend(handles=[line1, line2], labels=['EL gap=1 mm', 'EL gap=10 mm'],
           loc='upper left',ncol=2, fontsize=9)
fig.suptitle(f'Pitch = {pitch_choice[0]} mm',fontsize=18,fontweight='bold')
fig.tight_layout()

plt.show()


min_sep_dist_list = np.reshape(min_sep_dist_list,[3,3])

# # Plot delta r for initial separation 3x3 matrix
# fig, ax = plt.subplots()
# fig.patch.set_facecolor('white')
# # Display the array with no interpolation
# im = ax.imshow(min_sep_dist_list, interpolation='none',cmap='viridis')

# fig.suptitle(f"{delta_r} of earliest spatial separation\n" +\
#              f"Pitch={pitch_choice[0]} mm, EL gap={el_gap_choice[0]} mm",
#              fontsize=13, fontweight='bold')
# ax.set_xlabel("Fiber immersion [mm]",fontweight='bold')
# ax.set_ylabel("Anode distance [mm]",fontweight='bold')
# ax.set_xticks([0, 1, 2])
# ax.set_yticks([0, 1, 2])
# ax.set_xticklabels(['0', '3', '6'])
# ax.set_yticklabels(['2.5', '5', '10'])

# # Annotate each cell with the numeric value
# for i in range(min_sep_dist_list.shape[0]):
#     for j in range(min_sep_dist_list.shape[1]):
#         if min_sep_dist_list[i, j] == 0:
#             ax.text(j, i, 'No separation', ha='center', va='center', color='r',
#                     fontsize=12,fontweight='bold')
#         else:
#             ax.text(j, i, f'{min_sep_dist_list[i, j]:.1f}', ha='center', va='center',
#                     color='k',fontsize=15,fontweight='bold')

# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im, cax=cax)

# # Show the plot
# plt.show()


# In[4.1]
### GRAPHS - Find optimal EL gap 1x1 ###  

# Parameter choices to slice to - DO NOT CHANGE FOR CURRENT DATASET!
fiber_immersion_choice = [0]
pitch_choice = [15.6]
el_gap_choice = [1,10]
anode_distance_choice = [10]
holder_thickness_choice = [10]

color = iter(['red', 'green'])
marker = iter(['^', '*'])
SHOW_ORIGINAL_SPACING = False
if SHOW_ORIGINAL_SPACING is False:
    custom_spacing = 2
POLYNOMIAL_FIT = False

fig, ax = plt.subplots(figsize=(10,7), dpi = 600)
fig.patch.set_facecolor('white')


for i, geo_dir in tqdm(enumerate(geometry_dirs)):
    
    working_dir = geo_dir + r'/P2V'
    geo_params = geo_dir.split('/SquareFiberDatabase/')[-1]
    
    
    el_gap = float(re.search(r"ELGap=(-?\d+\.?\d*)mm",
                             geo_params).group(1))
    pitch = float(re.search(r"pitch=(-?\d+\.?\d*)mm",
                            geo_params).group(1))
    anode_distance = float(re.search(r"distanceAnodeHolder=(-?\d+\.?\d*)mm",
                                     geo_params).group(1))
    holder_thickness = float(re.search(r"holderThickness=(-?\d+\.?\d*)mm",
                                       geo_params).group(1))
    fiber_immersion = float(re.search(r"distanceFiberHolder=(-?\d+\.?\d*)mm",
                                      geo_params).group(1))
    fiber_immersion = 5 - fiber_immersion
    
    
    if (el_gap not in el_gap_choice or
        pitch not in pitch_choice or
        anode_distance not in anode_distance_choice or
        fiber_immersion not in fiber_immersion_choice or
        holder_thickness not in holder_thickness_choice):
        # print(i)
        continue
    
    
    dir_data = glob.glob(working_dir + '/*/*.txt')
    dists = []
    P2Vs = []
        
    if SHOW_ORIGINAL_SPACING:
        for data in dir_data:
            dist, P2V = np.loadtxt(data)
            if P2V > 100 or P2V == float('inf'):
                P2V = 100
            dists.append(dist)
            P2Vs.append(P2V)
            
        # zip and sort in ascending order
        sorted_dists, sorted_P2Vs = sort_ascending_dists_P2Vs(dists,P2Vs)
       
    else:
        averaged_dists = []
        averaged_P2Vs = []
        for j in range(0, len(dir_data), custom_spacing):  # Step through dir_data two at a time
            data_pairs = dir_data[j:j+custom_spacing]  # Get the current pair of data files
            dists = []
            P2Vs = []
            for data in data_pairs:
                dist, P2V = np.loadtxt(data)
                if P2V > 100 or P2V == float('inf'):
                    P2V = 100
                dists.append(dist)
                P2Vs.append(P2V)
                # print(f'data_pair dist={dist}, P2V={P2V}')
            
            # Only proceed if we have a pair, to avoid index out of range errors
            if len(dists) == custom_spacing and len(P2Vs) == custom_spacing:
                averaged_dist = sum(dists) / custom_spacing
                averaged_P2V = sum(P2Vs) / custom_spacing
                # print(f'averaged_dist={averaged_dist}, averaged_P2V={averaged_P2V}',end='\n\n')
                averaged_dists.append(averaged_dist)
                averaged_P2Vs.append(averaged_P2V)
        
        # zip and sort in ascending order
        sorted_dists, sorted_P2Vs = sort_ascending_dists_P2Vs(averaged_dists,
                                                              averaged_P2Vs)
        
    if POLYNOMIAL_FIT:
        # custom fit a polynomial of 2nd degree to data        
        sorted_dists_fit, sorted_P2V_fit = custom_polyfit(sorted_dists,
                                                          sorted_P2Vs)
        
        
        this_color = next(color)
        this_marker = next(marker)
        # Plot original data
        ax.plot(sorted_dists, sorted_P2Vs, color=this_color, ls='-', 
                    alpha=0.8, label=f'EL={el_gap}mm',marker=this_marker,
                    linewidth=2, markersize=10)
        # Plot fitted data
        ax.plot(sorted_dists_fit, sorted_P2V_fit, color=this_color, ls='--',
                 alpha=0.5, label=f'{el_gap}mm fitted',linewidth=3)
        
    else:
        print(f'EL={el_gap}')
        ax.plot(sorted_dists, sorted_P2Vs, color=next(color), ls='-', 
                    marker=next(marker),alpha=0.85, label=f'EL gap={el_gap}mm',
                    linewidth=2, markersize=10)
        
    

plt.xlabel(f'{delta_r} [mm]')
plt.ylabel('P2V')
plt.grid()
plt.legend(loc='upper left',fontsize=13)
text = (f'Pitch={pitch_choice[-1]} mm' +
         f'\nFiber immersion={fiber_immersion_choice[-1]} mm' +
         f'\nAnode distance={anode_distance_choice[-1]} mm')
if POLYNOMIAL_FIT:
    xloc, yloc = 0.0, 0.75
else:
    xloc, yloc = 0.03, 0.85
    
plt.text(xloc, yloc, text, ha='left', va='top',
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.5,boxstyle='round,pad=0.5'),
         fontsize=13, weight='bold', linespacing=1.5)  # Adjust this value to increase spacing between lines, if necessary


# fig.suptitle(title,size=15)
fig.tight_layout()
plt.show()


# In[5]
### GRAPHS - Find optimal immersion 2x3 ###  
  
# Parameter choices to slice to - DO NOT CHANGE FOR CURRENT DATASET!
fiber_immersion_choice = [0,3,6]
pitch_choice = [15.6]
el_gap_choice = [1,10]
anode_distance_choice = [2.5,5,10]
holder_thickness_choice = [10]
x_left_lim = pitch_choice[0]-1 # manual option remove plot area where there's P2V of 1 

SHOW_ORIGINAL_SPACING = False
if SHOW_ORIGINAL_SPACING is False:
    custom_spacing = 2
POLYNOMIAL_FIT = False

fig, ax = plt.subplots(3,2,figsize=(10,7), sharex=True,sharey=True,dpi=600)
fig.patch.set_facecolor('white')
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.2)
fig.supxlabel(f'{delta_r} [mm]',fontsize=12,fontweight='bold')
fig.supylabel('P2V',fontweight='bold')

for m,anode_dist in enumerate(anode_distance_choice):
    for n,el in enumerate(el_gap_choice):
        
        color = iter(['red', 'green', 'blue'])
        marker = iter(['^', '*',' s'])
        ax[m,n].grid()
        ax[m,n].set_ylim([0.99,2.5])
        
        if m == 0:
            ax[m,n].set_xlabel(f'EL gap = {el} mm',fontsize=10,fontweight='bold')
            ax[m,n].xaxis.set_label_position('top')
        for i, geo_dir in tqdm(enumerate(geometry_dirs)):

            working_dir = geo_dir + r'/P2V'
            geo_params = geo_dir.split('/SquareFiberDatabase/')[-1]
            
            
            el_gap = float(re.search(r"ELGap=(-?\d+\.?\d*)mm",
                                     geo_params).group(1))
            pitch = float(re.search(r"pitch=(-?\d+\.?\d*)mm",
                                    geo_params).group(1))
            anode_distance = float(re.search(r"distanceAnodeHolder=(-?\d+\.?\d*)mm",
                                             geo_params).group(1))
            holder_thickness = float(re.search(r"holderThickness=(-?\d+\.?\d*)mm",
                                               geo_params).group(1))
            fiber_immersion = float(re.search(r"distanceFiberHolder=(-?\d+\.?\d*)mm",
                                              geo_params).group(1))
            fiber_immersion = 5 - fiber_immersion
            
            
            if (el_gap != el or
                pitch not in pitch_choice or
                anode_distance != anode_dist or
                fiber_immersion not in fiber_immersion_choice or
                holder_thickness not in holder_thickness_choice):
                # print(i)
                continue
            
            
            dir_data = glob.glob(working_dir + '/*/*.txt')
            dists = []
            P2Vs = []
            
                
            if SHOW_ORIGINAL_SPACING:
                for data in dir_data:
                    dist, P2V = np.loadtxt(data)
                    if P2V > 100 or P2V == float('inf'):
                        P2V = 100
                    dists.append(dist)
                    P2Vs.append(P2V)
                    
                # zip and sort in ascending order
                sorted_dists, sorted_P2Vs = sort_ascending_dists_P2Vs(dists,P2Vs)
               
            else:           
                sorted_dists, sorted_P2Vs = dists_and_P2Vs_custom_spacing(dir_data,
                                            custom_spacing)
                
            if 'x_left_lim' in locals():
                filtered_indices = [i for i, dist in enumerate(sorted_dists) if dist > x_left_lim]
                sorted_dists = [sorted_dists[i] for i in filtered_indices]
                sorted_P2Vs = [sorted_P2Vs[i] for i in filtered_indices]
            
            if POLYNOMIAL_FIT:
                # custom fit a polynomial of 2nd degree to data        
                sorted_dists_fit, sorted_P2V_fit = custom_polyfit(sorted_dists,
                                                                  sorted_P2Vs,
                                                                  power=2)
                
                this_color = next(color)
                # Plot original data
                ax[m,n].plot(sorted_dists, sorted_P2Vs, color=this_color,
                             ls='-', linewidth=2, alpha=0.85)
                # Plot fitted data
                ax[m,n].plot(sorted_dists_fit, sorted_P2V_fit, color=this_color,
                             ls='--')
                
                
            else:
                this_color = next(color)
                print(f'color={this_color},immersion={fiber_immersion}')
                ax[m,n].plot(sorted_dists, sorted_P2Vs, color=this_color,
                             ls='-', linewidth=2, alpha=0.85)
             
            
        text=f'Anode dist={anode_dist} mm'
        ax[m, n].text(0.05, 0.93, text, ha='left', va='top',
                      transform=ax[m, n].transAxes,
                      bbox=dict(facecolor='white', boxstyle='square,pad=0.3'),
                      fontsize=10, fontweight='bold')
        
# Manually creating dummy lines for the legend
line1, = plt.plot([], [], color='blue', linestyle='-',
                  linewidth=3, label='Immersion=0 mm')

line2, = plt.plot([], [], color='green', linestyle='-',
                  linewidth=3, label='Immersion=3 mm')

line3, = plt.plot([], [], color='red', linestyle='-',
                  linewidth=3, label='Immersion=6 mm')

# Creating the legend manually with the dummy lines
fig.legend(handles=[line1, line2, line3], labels=['Immersion=0 mm', 
                                                  'Immersion=3 mm',
                                                  'Immersion=6 mm'],
            loc='upper left',ncol=2, fontsize=9)

fig.suptitle(f'Pitch = {pitch_choice[0]} mm',fontsize=18,fontweight='bold')
fig.tight_layout()
plt.show()

# In[5.1]
### GRAPHS - Find optimal immersion 1x1 ###  

# Parameter choices to slice to - DO NOT CHANGE FOR CURRENT DATASET!
fiber_immersion_choice = [0,3,6]
pitch_choice = [5]
el_gap_choice = [1]
anode_distance_choice = [2.5]
holder_thickness_choice = [10]

color = iter(['red', 'green', 'blue'])
marker = iter(['^', '*', 's'])
SHOW_ORIGINAL_SPACING = False
if SHOW_ORIGINAL_SPACING is False:
    custom_spacing = 2
POLYNOMIAL_FIT = False

fig, ax = plt.subplots(figsize=(10,7), dpi = 600)
fig.patch.set_facecolor('white')
for i, geo_dir in tqdm(enumerate(geometry_dirs)):
    
    working_dir = geo_dir + r'/P2V'
    geo_params = geo_dir.split('/SquareFiberDatabase/')[-1]
    
    
    el_gap = float(re.search(r"ELGap=(-?\d+\.?\d*)mm",
                             geo_params).group(1))
    pitch = float(re.search(r"pitch=(-?\d+\.?\d*)mm",
                            geo_params).group(1))
    anode_distance = float(re.search(r"distanceAnodeHolder=(-?\d+\.?\d*)mm",
                                     geo_params).group(1))
    holder_thickness = float(re.search(r"holderThickness=(-?\d+\.?\d*)mm",
                                       geo_params).group(1))
    fiber_immersion = float(re.search(r"distanceFiberHolder=(-?\d+\.?\d*)mm",
                                      geo_params).group(1))
    fiber_immersion = 5 - fiber_immersion
    
    
    if (el_gap not in el_gap_choice or
        pitch not in pitch_choice or
        anode_distance not in anode_distance_choice or
        fiber_immersion not in fiber_immersion_choice or
        holder_thickness not in holder_thickness_choice):
        # print(i)
        continue
    
    
    dir_data = glob.glob(working_dir + '/*/*.txt')
    dists = []
    P2Vs = []
        
    if SHOW_ORIGINAL_SPACING:
        for data in dir_data:
            dist, P2V = np.loadtxt(data)
            if P2V > 100 or P2V == float('inf'):
                P2V = 100
            dists.append(dist)
            P2Vs.append(P2V)
            
        # zip and sort in ascending order
        sorted_dists, sorted_P2Vs = sort_ascending_dists_P2Vs(dists,P2Vs)
       
    else:
        averaged_dists = []
        averaged_P2Vs = []
        for j in range(0, len(dir_data), custom_spacing):  # Step through dir_data two at a time
            data_pairs = dir_data[j:j+custom_spacing]  # Get the current pair of data files
            dists = []
            P2Vs = []
            for data in data_pairs:
                dist, P2V = np.loadtxt(data)
                if P2V > 100 or P2V == float('inf'):
                    P2V = 100
                dists.append(dist)
                P2Vs.append(P2V)
                # print(f'data_pair dist={dist}, P2V={P2V}')
            
            # Only proceed if we have a pair, to avoid index out of range errors
            if len(dists) == custom_spacing and len(P2Vs) == custom_spacing:
                averaged_dist = sum(dists) / custom_spacing
                averaged_P2V = sum(P2Vs) / custom_spacing
                # print(f'averaged_dist={averaged_dist}, averaged_P2V={averaged_P2V}',end='\n\n')
                averaged_dists.append(averaged_dist)
                averaged_P2Vs.append(averaged_P2V)
        
        # zip and sort in ascending order
        sorted_dists, sorted_P2Vs = sort_ascending_dists_P2Vs(averaged_dists,
                                                              averaged_P2Vs)
        
    if POLYNOMIAL_FIT:
        
        # custom fit a polynomial of 2nd degree to data
        sorted_dists_fit, sorted_P2V_fit = custom_polyfit(sorted_dists,
                                                          sorted_P2Vs)
        
        this_color = next(color)
        this_marker = next(marker)
        # Plot original data
        ax.plot(sorted_dists, sorted_P2Vs, color=this_color, ls='-', 
                    alpha=0.85, label=f'Pitch={pitch} mm',marker=this_marker,
                    linewidth=2, markersize=10)
        # Plot fitted data
        ax.plot(sorted_dists_fit, sorted_P2V_fit, color=this_color, ls='--',
                 label=f'{pitch} mm fitted')
        
    else:
        ax.plot(sorted_dists, sorted_P2Vs, color=next(color), ls='-', 
                    marker=next(marker),alpha=0.85, label=f'immersion={fiber_immersion} mm',
                    linewidth=2, markersize=10)
        
    
plt.xlabel(f'{delta_r} [mm]')
plt.ylabel('P2V')
plt.grid()
plt.legend(loc='upper left',fontsize=13)
text = (f'EL gap={el_gap_choice[-1]} mm' +
         f'\nPitch={pitch_choice[-1]} mm' +
         f'\nAnode distance={anode_distance_choice[-1]} mm'+ 
         f'\nHolder thickness={holder_thickness_choice[-1]} mm')
if POLYNOMIAL_FIT:
    xloc, yloc = 0.03, 0.68
else:
    xloc, yloc = 0.03, 0.8
    
plt.text(xloc, yloc, text, ha='left', va='top',
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.5,boxstyle='round,pad=0.5'),
         fontsize=13, weight='bold', linespacing=1.5)  # Adjust this value to increase spacing between lines, if necessary

# fig.suptitle(title,size=12)
fig.tight_layout()
plt.show()

# In[6]
### GRAPHS - Find optimal anode distance 2x3 ###  
  
# Parameter choices to slice to - DO NOT CHANGE FOR CURRENT DATASET!
fiber_immersion_choice = [0,3,6]
pitch_choice = [15.6]
el_gap_choice = [1,10]
anode_distance_choice = [2.5,5,10]
holder_thickness_choice = [10]
x_left_lim = pitch_choice[0]-1 # manual option remove plot area where there's P2V of 1 

SHOW_ORIGINAL_SPACING = False
if SHOW_ORIGINAL_SPACING is False:
    custom_spacing = 2
POLYNOMIAL_FIT = False

fig, ax = plt.subplots(3,2,figsize=(10,7), sharex=True,sharey=True,dpi=600)
fig.patch.set_facecolor('white')
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.2)
fig.supxlabel(f'{delta_r} [mm]',fontsize=12,fontweight='bold')
fig.supylabel('P2V',fontweight='bold')

for m,immersion in enumerate(fiber_immersion_choice):
    for n,el in enumerate(el_gap_choice):
        
        color = iter(['red', 'green', 'blue'])
        marker = iter(['^', '*',' s'])
        ax[m,n].grid()
        ax[m,n].set_ylim([0.99,2.5])
        
        if m == 0:
            ax[m,n].set_xlabel(f'EL gap = {el} mm',fontsize=10,fontweight='bold')
            ax[m,n].xaxis.set_label_position('top')
        for i, geo_dir in tqdm(enumerate(geometry_dirs)):

            working_dir = geo_dir + r'/P2V'
            geo_params = geo_dir.split('/SquareFiberDatabase/')[-1]
            
            
            el_gap = float(re.search(r"ELGap=(-?\d+\.?\d*)mm",
                                     geo_params).group(1))
            pitch = float(re.search(r"pitch=(-?\d+\.?\d*)mm",
                                    geo_params).group(1))
            anode_distance = float(re.search(r"distanceAnodeHolder=(-?\d+\.?\d*)mm",
                                             geo_params).group(1))
            holder_thickness = float(re.search(r"holderThickness=(-?\d+\.?\d*)mm",
                                               geo_params).group(1))
            fiber_immersion = float(re.search(r"distanceFiberHolder=(-?\d+\.?\d*)mm",
                                              geo_params).group(1))
            fiber_immersion = 5 - fiber_immersion
            
            
            if (el_gap != el or
                pitch not in pitch_choice or
                anode_distance not in  anode_distance_choice or
                fiber_immersion != immersion or
                holder_thickness not in holder_thickness_choice):
                # print(i)
                continue
            
            
            dir_data = glob.glob(working_dir + '/*/*.txt')
            dists = []
            P2Vs = []
            
                
            if SHOW_ORIGINAL_SPACING:
                for data in dir_data:
                    dist, P2V = np.loadtxt(data)
                    if P2V > 100 or P2V == float('inf'):
                        P2V = 100
                    dists.append(dist)
                    P2Vs.append(P2V)
                    
                # zip and sort in ascending order
                sorted_dists, sorted_P2Vs = sort_ascending_dists_P2Vs(dists,P2Vs)
               
            else:           
                sorted_dists, sorted_P2Vs = dists_and_P2Vs_custom_spacing(dir_data,
                                            custom_spacing)
                
            if 'x_left_lim' in locals():
                filtered_indices = [i for i, dist in enumerate(sorted_dists) if dist > x_left_lim]
                sorted_dists = [sorted_dists[i] for i in filtered_indices]
                sorted_P2Vs = [sorted_P2Vs[i] for i in filtered_indices]
            
            if POLYNOMIAL_FIT:
                # custom fit a polynomial of 2nd degree to data        
                sorted_dists_fit, sorted_P2V_fit = custom_polyfit(sorted_dists,
                                                                  sorted_P2Vs,
                                                                  power=2)
                
                this_color = next(color)
                # Plot original data
                ax[m,n].plot(sorted_dists, sorted_P2Vs, color=this_color,
                             ls='-', linewidth=2, alpha=0.85)
                # Plot fitted data
                ax[m,n].plot(sorted_dists_fit, sorted_P2V_fit, color=this_color,
                             ls='--')
                
                
            else:
                this_color = next(color)
                print(f'color={this_color},anode dist={anode_distance}')
                ax[m,n].plot(sorted_dists, sorted_P2Vs, color=this_color,
                             ls='-', linewidth=2, alpha=0.85)
             
            
        text=f'Immersion={immersion} mm'
        ax[m, n].text(0.05, 0.93, text, ha='left', va='top',
                      transform=ax[m, n].transAxes,
                      bbox=dict(facecolor='white', boxstyle='square,pad=0.3'),
                      fontsize=10, fontweight='bold')
        
# Manually creating dummy lines for the legend
line1, = plt.plot([], [], color='green', linestyle='-',
                  linewidth=3, label='Anode dist=2.5 mm')

line2, = plt.plot([], [], color='blue', linestyle='-',
                  linewidth=3, label='Anode dist=5 mm')

line3, = plt.plot([], [], color='red', linestyle='-',
                  linewidth=3, label='Anode dist=10 mm')

# Creating the legend manually with the dummy lines
fig.legend(handles=[line1, line2, line3], labels=['Anode dist=2.5 mm', 
                                                  'Anode dist=5 mm',
                                                  'Anode dist=10 mm'],
            loc='upper left',ncol=2, fontsize=9)

fig.suptitle(f'Pitch = {pitch_choice[0]} mm',fontsize=18,fontweight='bold')
fig.tight_layout()
plt.show()


# In[7]
# Best geometry for each pitch 
# WARNING: These parameters were not easy to come by, do not delete #
best_geo_pitch_5 = {'EL':[1], 'pitch':[5], 'immersion':[3], 'anode_dist':[2.5]}
best_geo_pitch_10 = {'EL':[1], 'pitch':[10], 'immersion':[3], 'anode_dist':[5]}
best_geo_pitch_15 = {'EL':[10], 'pitch':[15.6], 'immersion':[3], 'anode_dist':[10]}
best_geos = [best_geo_pitch_5, best_geo_pitch_10, best_geo_pitch_15]
holder_thickness_choice = [10]



color = iter(['blue', 'red', 'green'])
marker = iter(['s', '^', '*'])
SHOW_ORIGINAL_SPACING = False
if SHOW_ORIGINAL_SPACING is False:
    custom_spacing = 2
POLYNOMIAL_FIT = False

fig, ax = plt.subplots(figsize=(10,7), dpi=600)
fig.patch.set_facecolor('white')
for geo in best_geos:
    for i, geo_dir in tqdm(enumerate(geometry_dirs)):
        
        working_dir = geo_dir + r'/P2V'
        geo_params = geo_dir.split('/SquareFiberDatabase/')[-1]
        
        
        el_gap = float(re.search(r"ELGap=(-?\d+\.?\d*)mm",
                                 geo_params).group(1))
        pitch = float(re.search(r"pitch=(-?\d+\.?\d*)mm",
                                geo_params).group(1))
        anode_distance = float(re.search(r"distanceAnodeHolder=(-?\d+\.?\d*)mm",
                                         geo_params).group(1))
        holder_thickness = float(re.search(r"holderThickness=(-?\d+\.?\d*)mm",
                                           geo_params).group(1))
        fiber_immersion = float(re.search(r"distanceFiberHolder=(-?\d+\.?\d*)mm",
                                          geo_params).group(1))
        fiber_immersion = 5 - fiber_immersion
        
        
        if (el_gap not in geo['EL'] or
            pitch not in geo['pitch'] or
            anode_distance not in geo['anode_dist'] or
            fiber_immersion not in geo['immersion'] or
            holder_thickness not in holder_thickness_choice):
            # print(i)
            continue
        
        
        dir_data = glob.glob(working_dir + '/*/*.txt')
        dists = []
        P2Vs = []
            
        if SHOW_ORIGINAL_SPACING:
            for data in dir_data:
                dist, P2V = np.loadtxt(data)
                if P2V > 100 or P2V == float('inf'):
                    P2V = 100
                dists.append(dist)
                P2Vs.append(P2V)
                
            # zip and sort in ascending order
            sorted_dists, sorted_P2Vs = sort_ascending_dists_P2Vs(dists,P2Vs)
           
        else:
            averaged_dists = []
            averaged_P2Vs = []
            for j in range(0, len(dir_data), custom_spacing):  # Step through dir_data two at a time
                data_pairs = dir_data[j:j+custom_spacing]  # Get the current pair of data files
                dists = []
                P2Vs = []
                for data in data_pairs:
                    dist, P2V = np.loadtxt(data)
                    if P2V > 100 or P2V == float('inf'):
                        P2V = 100
                    dists.append(dist)
                    P2Vs.append(P2V)
                    # print(f'data_pair dist={dist}, P2V={P2V}')
                
                # Only proceed if we have a pair, to avoid index out of range errors
                if len(dists) == custom_spacing and len(P2Vs) == custom_spacing:
                    averaged_dist = sum(dists) / custom_spacing
                    averaged_P2V = sum(P2Vs) / custom_spacing
                    # print(f'averaged_dist={averaged_dist}, averaged_P2V={averaged_P2V}',end='\n\n')
                    averaged_dists.append(averaged_dist)
                    averaged_P2Vs.append(averaged_P2V)
            
            # zip and sort in ascending order
            sorted_dists, sorted_P2Vs = sort_ascending_dists_P2Vs(averaged_dists,
                                                                  averaged_P2Vs)
            
        if POLYNOMIAL_FIT:
            
            # custom fit a polynomial of 2nd degree to data
            sorted_dists_fit, sorted_P2V_fit = custom_polyfit(sorted_dists,
                                                              sorted_P2Vs)
            
            this_color = next(color)
            this_marker = next(marker)
            # Plot original data
            ax.plot(sorted_dists, sorted_P2Vs, color=this_color, ls='-', 
                        alpha=0.85, label=f'Pitch={pitch} mm',marker=this_marker,
                        linewidth=2, markersize=10)
            # Plot fitted data
            ax.plot(sorted_dists_fit, sorted_P2V_fit, color=this_color, ls='--',
                     label=f'{pitch} mm fitted')
            
        else:
            ax.plot(sorted_dists, sorted_P2Vs, color=next(color), ls='-', 
                        marker=next(marker),alpha=0.7, label=f'pitch={pitch} mm',
                        linewidth=3, markersize=10)
        
    
plt.xlabel(f'{delta_r} [mm]',fontsize=15)
plt.ylabel('P2V',fontsize=15)
plt.grid()
plt.legend(loc='upper left',fontsize=13)
plt.ylim([0.99,None])
pitch_5_text = (f"EL gap={best_geo_pitch_5['EL'][-1]} mm" +
         f"\nImmersion={best_geo_pitch_5['immersion'][-1]} mm" +
         f"\nAnode distance={best_geo_pitch_5['anode_dist'][-1]} mm")
plt.text(9.5, 2.5, pitch_5_text, ha='left', va='top',
         bbox=dict(facecolor='white', alpha=0.5,boxstyle='round,pad=0.5'),
         fontsize=10, weight='bold', color='blue', linespacing=1.5)

pitch_10_text = (f"EL gap={best_geo_pitch_10['EL'][-1]} mm" +
         f"\nImmersion={best_geo_pitch_10['immersion'][-1]} mm" +
         f"\nAnode distance={best_geo_pitch_10['anode_dist'][-1]} mm")
plt.text(18, 3, pitch_10_text, ha='left', va='top',
         bbox=dict(facecolor='white', alpha=0.5,boxstyle='round,pad=0.5'),
         fontsize=10, weight='bold', color='red', linespacing=1.5)  # Adjust this value to increase spacing between lines, if necessary

pitch_15_text = (f"EL gap={best_geo_pitch_15['EL'][-1]} mm" +
         f"\nImmersion={best_geo_pitch_15['immersion'][-1]} mm" +
         f"\nAnode distance={best_geo_pitch_15['anode_dist'][-1]} mm")
plt.text(25, 1.6, pitch_15_text, ha='left', va='top',
         bbox=dict(facecolor='white', alpha=0.5,boxstyle='round,pad=0.5'),
         fontsize=10, weight='bold', color='green', linespacing=1.5)  # Adjust this value to increase spacing between lines, if necessary


# fig.suptitle(title,size=12)
plt.show()