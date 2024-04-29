#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:22:36 2023

@author: amir
"""

import os
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.style.use('classic')
from tqdm import tqdm
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
import glob
import re
from scipy import ndimage
# from invisible_cities.reco.deconv_functions     import richardson_lucy
from scipy.interpolate import griddata
from scipy.signal import find_peaks, peak_widths
from scipy.signal import butter, filtfilt, welch
from scipy.ndimage import rotate
from scipy.optimize import curve_fit
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

# In[0]
# Generate all PSFs (of geant4 TPB hits) from SquareFiberDataset

TO_GENERATE = False
TO_PLOT = False
TO_SAVE = False

if TO_GENERATE:
    for directory in tqdm(geometry_dirs[0]):
        
        directory = ('/media/amir/Extreme Pro/SquareFiberDatabase/' +
                    'ELGap=1mm_pitch=5mm_distanceFiberHolder=2mm_' +
                    'distanceAnodeHolder=5mm_holderThickness=10mm')
        
        PSF_TPB = psf_creator(directory,create_from="TPB",
                              to_plot=TO_PLOT,to_smooth=False)
        os.chdir(directory)
        
        if TO_SAVE:
            save_PSF = directory + '/PSF.npy'
            np.save(save_PSF,PSF_TPB)
            
# In[1]
# plot and save all TPB PSFs from SquareFiberDataset in their respective folders
TO_GENERATE = True
TO_SAVE = False
print('Generating PSF plots')
if TO_GENERATE:
    for directory in tqdm(geometry_dirs[0]):
        directory = ('/media/amir/Extreme Pro/SquareFiberDatabase/' +
                    'ELGap=1mm_pitch=5mm_distanceFiberHolder=2mm_' +
                    'distanceAnodeHolder=5mm_holderThickness=10mm')
        os.chdir(directory)
        print(r'Working on directory:'+f'\n{os.getcwd()}')
        PSF = np.load(r'PSF.npy')
        fig = plot_PSF(PSF=PSF)
        
        if TO_SAVE:
            save_path = directory + r'/PSF_plot.jpg'
            fig.savefig(save_path)  
            plt.close(fig)
# In[2]
'''
Generate and save twin events dataset, after shifting, centering and rotation
'''


TO_GENERATE = False
TO_PLOT = False
TO_SAVE = False
FORCE_DISTANCE_RANGE = False
random_shift = False
if random_shift is True:
    m_min = 0
    m_max = 2 # exclusive - up to, not including
    n_min = 0
    n_max = 2 # exclusive - up to, not including
if random_shift is False:
    m,n = 3,3
samples_per_geometry = 10000

if TO_GENERATE:
    x_match_str = r"_x=(-?\d+(?:\.\d+)?(?:e-?\d+)?)mm"
    y_match_str = r"_y=(-?\d+(?:\.\d+)?(?:e-?\d+)?)mm"

    for geo_dir in tqdm(geometry_dirs[0]):
        
        geo_dir = ('/media/amir/Extreme Pro/SquareFiberDatabase/' +
                    'ELGap=1mm_pitch=5mm_distanceFiberHolder=2mm_' +
                    'distanceAnodeHolder=5mm_holderThickness=10mm')
             
        
        # find min distance that will be of interest
        match = re.search(r"_pitch=(\d+(?:\.\d+)?)mm", geo_dir)
        pitch = float(match.group(1))
        if FORCE_DISTANCE_RANGE:
            dist_min_threshold = int(0.5*pitch) # mm
            dist_max_threshold = int(2*pitch) # mm

        # assign input and output directories
        print(geo_dir)
        working_dir = geo_dir + r'/Geant4_Kr_events'
        save_dir = geo_dir + r'/combined_event_SR' 
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        os.chdir(working_dir)

        event_pattern = "SiPM_hits"
        event_list = [entry.name for entry in os.scandir() if entry.is_file() 
                      and entry.name.startswith(event_pattern)]
        
        # # in case we add more samples, find highest existing sample number
        # highest_existing_data_number = find_highest_number(save_dir)
        # if highest_existing_data_number == -1:
        #     highest_existing_data_number = 0


        j = 0
        dists = []
        while j < samples_per_geometry:
            # print(j)
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
            if random_shift:
                m, n = np.random.randint(m_min,m_max), np.random.randint(n_min,n_max)
            shifted_event_SR = event_to_shift_SR + [m*pitch, n*pitch]
    
            # Combine the two events
            combined_event_SR = np.concatenate((event_to_stay_SR, shifted_event_SR))
            shifted_event_coord = np.array([x1, y1]) + [m*pitch, n*pitch]
    
            # get distance between stay and shifted
            dist = (np.sqrt((x0-shifted_event_coord[0])**2+(y0-shifted_event_coord[1])**2))

            # take specific distances in ROI
            
            if FORCE_DISTANCE_RANGE and dist_min_threshold < dist < dist_max_threshold:
                continue
            
            dists.append(dist)
            # get midpoint of stay and shifted
            midpoint = [(x0+shifted_event_coord[0])/2,(y0+shifted_event_coord[1])/2]
            
            theta = np.arctan2(y0-shifted_event_coord[1],x0-shifted_event_coord[0])
            rot_matrix = np.array([[np.cos(theta),-np.sin(theta)],
                                    [np.sin(theta),np.cos(theta)]])
    
            # center combined event using midpoint
            centered_combined_event_SR = combined_event_SR - midpoint
    
            # # Save combined centered event to suitlable folder according to sources distance
            if TO_SAVE:
                # save to dirs with spacing 0.5 mm
                spacing = 0.5
                if int(dist) <= dist < int(dist) + spacing:
                    save_path = save_dir + f'/{int(dist)}_mm'
                if int(dist) + spacing <= dist < np.ceil(dist):
                    save_path = save_dir + f'/{int(dist) + spacing}_mm'
                                       
                if not os.path.isdir(save_path):
                    os.mkdir(save_path)
                save_path = (save_path + f'/{j}_' +
                f'rotation_angle_rad={np.around(theta,5)}.npy')
                np.save(save_path,centered_combined_event_SR)
                
                #also, useful to save dists vector
                np.save(geo_dir+r'/dists.npy', dists) 
    
            j += 1
            
            if TO_PLOT:
                # plot sensor responses
                title = "Staying event"
                plot_sensor_response(event_to_stay_SR, bins, size, title=title)
                title = "Shifted event"
                plot_sensor_response(event_to_shift_SR, bins, size, title=title)
                title = "Combined event"
                plot_sensor_response(combined_event_SR, bins, size, title=title)
                title = "Centered combined event"
                plot_sensor_response(centered_combined_event_SR, bins, size, title=title)
            
        print(f'Created {count_npy_files(save_dir)} events')
            
# In[3]
'''
Load twin events after shifted and centered.
interpolate, deconv, rotate and save.
'''
TO_GENERATE = True
TO_SAVE = False
TO_PLOT_EACH_STEP = True
TO_PLOT_DECONVE_STACK = True
TO_SMOOTH_PSF = False

if TO_GENERATE:
    for geo_dir in tqdm(geometry_dirs[0]):
        
        geo_dir = ('/media/amir/Extreme Pro/SquareFiberDatabase/' +
                    'ELGap=1mm_pitch=5mm_distanceFiberHolder=2mm_' +
                    'distanceAnodeHolder=5mm_holderThickness=10mm')

            
        # grab geometry parameters for plot
        geo_params = geo_dir.split('/SquareFiberDatabase/')[-1]
        el_gap = float(re.search(r"ELGap=(-?\d+\.?\d*)mm",
                                 geo_params).group(1))
        pitch = float(re.search(r"pitch=(-?\d+\.?\d*)mm",
                                geo_params).group(1))
        
        fiber_immersion = float(re.search(r"distanceFiberHolder=(-?\d+\.?\d*)mm",
                                          geo_params).group(1))
        anode_distance = float(re.search(r"distanceAnodeHolder=(-?\d+\.?\d*)mm",
                                         geo_params).group(1))
        holder_thickness = float(re.search(r"holderThickness=(-?\d+\.?\d*)mm",
                                           geo_params).group(1))
        
        fiber_immersion = 5 - fiber_immersion # convert from a Geant4 parameter to a simpler one
    
        # assign directories
        # print(geo_dir)
        working_dir = geo_dir + r'/combined_event_SR'
        save_dir = geo_dir + r'/deconv'
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        
        dist_dirs = glob.glob(working_dir + '/*')
        dist_dirs = sorted(dist_dirs, key=extract_dir_number)
        
        PSF = np.load(geo_dir + '/PSF.npy')
        if TO_SMOOTH_PSF:
            PSF = smooth_PSF(PSF)

        # # option to choose a single distance
        dist = 9
        user_chosen_dir = find_subdirectory_by_distance(working_dir, dist)
        
        for dist_dir in tqdm(dist_dirs):
            # print(dist_dir)
            
            # print(f'Working on:\n{dist_dir}')
            # match = re.search(r'/(\d+)_mm', dist_dir)
            match = re.search(r'/(\d+(?:\.\d+)?)_mm$', dist_dir)
            if match:
                dist = float(match.group(1))
            dist_dir = user_chosen_dir
            print(f'\nWorking on:\n{dist_dir}')
                        
                
            deconv_stack = np.zeros((size,size))
            cutoff_iter_list = []
            rel_diff_checkout_list = []
            event_files = glob.glob(dist_dir + '/*.npy')
            
            
            for event_file in tqdm(event_files):
                event = np.load(event_file)
                
                ##### interpolation #####

                # Create a 2D histogram
                size=250
                bins=size
                # bins = np.arange(-size/2,size/2,3.25)
                hist, x_edges, y_edges = np.histogram2d(event[:,0], event[:,1],
                                                        range=[[-size/2, size/2],
                                                                [-size/2, size/2]],
                                                        bins=bins)


                # Compute the centers of the bins
                x_centers = (x_edges[:-1] + x_edges[1:]) / 2
                y_centers = (y_edges[:-1] + y_edges[1:]) / 2

                hist_hits_x_idx, hist_hits_y_idx = np.where(hist>0)
                hist_hits_x = x_centers[hist_hits_x_idx]
                hist_hits_y = y_centers[hist_hits_y_idx]
                hist_hits_vals = hist[hist>0]


                # Define the interpolation grid
                x_range = np.linspace(-size/2, size/2, num=bins)
                y_range = np.linspace(-size/2, size/2, num=bins)
                x_grid, y_grid = np.meshgrid(x_range, y_range)

                # Perform the interpolation
                interp_img = griddata((hist_hits_x, hist_hits_y), hist_hits_vals,
                                      (x_grid, y_grid), method='cubic', fill_value=0)


                # optional, cut interp image values below 0
                interp_img[interp_img<0] = 0


                ##### RL deconvolution #####
                rel_diff_checkout, cutoff_iter, deconv = richardson_lucy(interp_img, PSF,
                                                                  iterations=75, iter_thr=0.01)
                
                
                cutoff_iter_list.append(cutoff_iter)
                rel_diff_checkout_list.append(rel_diff_checkout)
                

                ##### ROTATE #####
                theta = extract_theta_from_path(event_file)
                rot_matrix = np.array([[np.cos(theta),-np.sin(theta)],
                                        [np.sin(theta),np.cos(theta)]])

                # rotate combined event AFTER deconv
                rotated_deconv = rotate(deconv, np.degrees(theta), reshape=False, mode='nearest')
                deconv_stack += rotated_deconv
                

                
                
                if TO_PLOT_EACH_STEP:
                    
                    size = 50
                    bins = size
                    
                    ## Plot stages of deconv aggregation ##
                    
                    fig, ax = plt.subplots(2,2,figsize=(12,12),dpi=600)
                    fig.patch.set_facecolor('white')
                    plt.subplots_adjust(left=0.1, right=0.9, top=0.9,
                                        bottom=0.1, wspace=0.1, hspace=0.1)
                    # Format colorbar tick labels in scientific notation
                    formatter = ticker.ScalarFormatter(useMathText=True)
                    formatter.set_scientific(True)
                    formatter.set_powerlimits((-1, 1))  # You can adjust limits as needed
                    
                    # SR combined event
                    
                    im = ax[0,0].imshow(hist.T[100:150,100:150],interpolation='nearest',
                                        extent=[-size/2, size/2, -size/2, size/2],
                                        vmin=0, origin='lower')
                    
                    # option to set zero values as different color
                    # bla = hist.T[100:150,100:150]
                    # bla[bla==0] = np.nan
                    # im = ax[0,0].imshow(bla,interpolation='nearest',
                    #                     extent=[-size/2, size/2, -size/2, size/2],
                    #                     vmin=0, origin='lower')
                    
                                        
                    divider = make_axes_locatable(ax[0,0])
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cbar = plt.colorbar(im, cax=cax, label='Photon hits')
                    cbar.ax.yaxis.set_major_formatter(formatter)
                    ax[0,0].set_title("Tracking plane response",fontsize=13,fontweight='bold')
                    ax[0,0].set_xlabel('x [mm]',fontsize=15)
                    ax[0,0].set_ylabel('y [mm]',fontsize=15)
                    
                    # Interpolated combined event
                    im = ax[0,1].imshow(interp_img[100:150,100:150],
                                        extent=[-size/2, size/2, -size/2, size/2],
                                        vmin=0, origin='lower')
                    divider = make_axes_locatable(ax[0,1])
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cbar = plt.colorbar(im, cax=cax, label='Photon hits')
                    cbar.ax.yaxis.set_major_formatter(formatter)
                    ax[0,1].set_title('Cubic Interpolation of response',
                                      fontsize=13,fontweight='bold')
                    ax[0,1].set_xlabel('x [mm]',fontsize=15)
                    ax[0,1].set_ylabel('y [mm]',fontsize=15)
                    
                    # RL deconvolution
                    im = ax[1,0].imshow(deconv[100:150,100:150],
                                        extent=[-size/2, size/2, -size/2, size/2],
                                        vmin=0, origin='lower')
                    divider = make_axes_locatable(ax[1,0])
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cbar = plt.colorbar(im, cax=cax, label='Photon hits')
                    cbar.ax.yaxis.set_major_formatter(formatter)
                    ax[1,0].set_title('RL deconvolution',fontsize=13,fontweight='bold')
                    ax[1,0].set_xlabel('x [mm]',fontsize=15)
                    ax[1,0].set_ylabel('y [mm]',fontsize=15)

                    # ROTATED RL deconvolution
                    im = ax[1,1].imshow(rotated_deconv[100:150,100:150],
                                        extent=[-size/2, size/2, -size/2, size/2],
                                        vmin=0, origin='lower')
                    divider = make_axes_locatable(ax[1,1])
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cbar = plt.colorbar(im, cax=cax, label='Photon hits')
                    cbar.ax.yaxis.set_major_formatter(formatter)
                    ax[1,1].set_title('Rotated RL deconvolution',
                                      fontsize=13,fontweight='bold')
                    ax[1,1].set_xlabel('x [mm]',fontsize=15)
                    ax[1,1].set_ylabel('y [mm]',fontsize=15)
                    fig.tight_layout()
                    plt.show()
                    
                    
                    # # plot deconvolution stacking
                    # plt.imshow(deconv_stack, extent=[-size/2, size/2, -size/2, size/2],
                    #             vmin=0, origin='lower')
                    # plt.colorbar(label='Photon hits')
                    # plt.title('Accomulated RL deconvolution')
                    # plt.xlabel('x [mm]')
                    # plt.ylabel('y [mm]')
                    # plt.show()
                    
                    size = 250
                    bins = size
                    
            # # Optional: show the averaged deconv, added after talk to Gonzalo
            # deconv_stack = deconv_stack/len(event_files)
            
            avg_cutoff_iter = np.mean(cutoff_iter_list)
            avg_rel_diff_checkout = np.mean(rel_diff_checkout_list)
            
            ## save deconv_stack+avg_cutoff_iter+avg_rel_diff_checkout ###
            if TO_SAVE:
                ## plot deconv##
                # size = 50
                # bins = size
                fig, ax = plt.subplots(figsize=(8,7.5),dpi=600)
                fig.patch.set_facecolor('white')
                formatter = ticker.ScalarFormatter(useMathText=True)
                formatter.set_scientific(True)
                formatter.set_powerlimits((-1, 1))  # You can adjust limits as needed
                    
                # deconv
                # im = ax.imshow(deconv_stack[100:150,100:150],
                #                extent=[-size/2, size/2, -size/2, size/2])
                im = ax.imshow(deconv_stack,
                               extent=[-size/2, size/2, -size/2, size/2])
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(im, cax=cax, label='Photon hits')
                cbar.ax.yaxis.set_major_formatter(formatter)
                ax.set_xlabel('x [mm]')
                ax.set_ylabel('y [mm]')
                ax.set_title('Stacked RL deconvolutions',fontsize=15,fontweight='bold')
                    
                title = (f'EL gap={el_gap}mm, pitch={pitch} mm,' + 
                          f' fiber immersion={fiber_immersion} mm,\nanode distance={anode_distance} mm' + 
                          f'\n\nEvent spacing={dist} mm,' + 
                        f' Avg RL iterations={int(avg_cutoff_iter)},'  +
                        f' Avg RL relative diff={np.around(avg_rel_diff_checkout,4)}')
            
                fig.suptitle(title,fontsize=15,fontweight='bold')
                ax.grid()
                fig.tight_layout()
                # size = 250
                # bins = size
                
                # save files and deconv image plot in correct folder #
                spacing = 0.5
                if int(dist) <= dist < int(dist) + spacing:
                    save_path = save_dir + f'/{int(dist)}_mm'
                if int(dist) + spacing <= dist < np.ceil(dist):
                    save_path = save_dir + f'/{int(dist) + spacing}_mm'
                
                if not os.path.isdir(save_path):
                    os.mkdir(save_path)
                np.save(save_path+'/deconv.npy',deconv_stack)
                np.savetxt(save_path+'/avg_cutoff_iter.txt',
                           np.array([avg_cutoff_iter]))
                np.savetxt(save_path+'/avg_rel_diff_checkout.txt',
                           np.array([avg_rel_diff_checkout]))
                fig.savefig(save_path+'/deconv_plot', format='svg')
            if TO_PLOT_DECONVE_STACK: 
                plt.show()
                continue
            plt.close(fig)
          
# In[4]
'''
Load deconv stack matrix for each dist dir and calculate P2V
(w/wo signal smoothing)
'''

# Design a Butterworth low-pass filter
def butter_lowpass_filter(signal, fs, cutoff, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, signal)
    return filtered_data


def frequency_analyzer(signal, spatial_sampling_rate):
    f, Pxx = welch(signal, fs=spatial_sampling_rate, nperseg=250)
    cumulative_power = np.cumsum(Pxx) / np.sum(Pxx)
    cutoff_index = np.where(cumulative_power >= 0.8)[0][0]
    dynamic_spatial_cutoff_frequency = f[cutoff_index]  # Correctly identified cutoff frequency
    return dynamic_spatial_cutoff_frequency


# Define the model function for two Gaussians
def double_gaussian(x, A1, A2, mu1, mu2, sigma):
    f = A1 * np.exp(-((x - mu1)**2) / (2 * sigma**2)) + \
    A2 * np.exp(-((x - mu2)**2) / (2 * sigma**2))
    return f


TO_GENERATE = True
TO_SAVE = False
TO_PLOT_P2V = True
TO_SMOOTH_SIGNAL = False
if TO_SMOOTH_SIGNAL: # If True, only one option must be True - the other, False.
    DOUBLE_GAUSSIAN_FIT = True # currently, the most promising approach !
    PSD_AND_BUTTERWORTH = not DOUBLE_GAUSSIAN_FIT



if TO_GENERATE:
    for geo_dir in tqdm(geometry_dirs[0]):
        
        # geo_dir = ('/media/amir/Extreme Pro/SquareFiberDatabase/' +
        #             'ELGap=1mm_pitch=15.6mm_distanceFiberHolder=-1mm_' +
        #             'distanceAnodeHolder=2.5mm_holderThickness=10mm')
        
        geo_dir = ('/media/amir/Extreme Pro/SquareFiberDatabase/' +
                    'ELGap=1mm_pitch=5mm_distanceFiberHolder=2mm_' +
                    'distanceAnodeHolder=5mm_holderThickness=10mm')

        # grab geometry parameters for plot
        geo_params = geo_dir.split('/SquareFiberDatabase/')[-1]
        el_gap = float(re.search(r"ELGap=(-?\d+\.?\d*)mm",
                                 geo_params).group(1))
        pitch = float(re.search(r"pitch=(-?\d+\.?\d*)mm",
                                geo_params).group(1))
        
        fiber_immersion = float(re.search(r"distanceFiberHolder=(-?\d+\.?\d*)mm",
                                          geo_params).group(1))
        anode_distance = float(re.search(r"distanceAnodeHolder=(-?\d+\.?\d*)mm",
                                         geo_params).group(1))
        holder_thickness = float(re.search(r"holderThickness=(-?\d+\.?\d*)mm",
                                           geo_params).group(1))
        
        fiber_immersion = 5 - fiber_immersion # convert from a Geant4 parameter to a simpler one
    
        # assign directories
        # print(geo_dir)
        working_dir = geo_dir + r'/deconv'
        save_dir = geo_dir + r'/P2V'
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        
        dist_dirs = glob.glob(working_dir + '/*')
        dist_dirs = sorted(dist_dirs, key=extract_dir_number)

        # # option to choose a single distance and see P2V
        dist = 9
        user_chosen_dir = find_subdirectory_by_distance(working_dir, dist)
        
        for dist_dir in dist_dirs:
            # print(dist_dir)
            
            # print(f'Working on:\n{dist_dir}')
            match = re.search(r'/(\d+(?:\.\d+)?)_mm$', dist_dir)
            if match:
                dist = float(match.group(1))
                
            dist_dir = user_chosen_dir
            match = re.search(r'/(\d+(?:\.\d+)?)_mm$', dist_dir)
            if match:
                dist = float(match.group(1))
            print(f'Working on:\n{dist_dir}')
            
            
            ## load deconv_stack+avg_cutoff_iter+avg_rel_diff_checkout ###
            deconv_stack = np.load(dist_dir + '/deconv.npy')
            avg_cutoff_iter = float(np.genfromtxt(dist_dir + '/avg_cutoff_iter.txt'))
            avg_rel_diff_checkout = float(np.genfromtxt(dist_dir + '/avg_rel_diff_checkout.txt'))

            deconv_stack_1d = deconv_stack.mean(axis=0)
        
            
            if TO_SMOOTH_SIGNAL:
                
                if DOUBLE_GAUSSIAN_FIT:
                    
                    # Method 1 - use custom 2 gaussianinitial_guesses fit function #
                    x = np.arange(-size/2, size/2)
                    y = deconv_stack_1d
                    
                    # # look at zoomed region near peaks
                    # one_side_dist = 5*pitch
                    # x = np.arange(-one_side_dist, one_side_dist)
                    # center_y = int(len(deconv_stack_1d)/2)
                    # y = deconv_stack_1d[center_y-int(np.round(one_side_dist)):
                    #                     center_y+int(np.round(one_side_dist))]
                    
                    # # try to extrach sigma using FWHM
                    # # problem: when peaks are combind gives FWHM of combined peak
                    # peak_idx, _ = find_peaks(y, height=max(y))
                    # fwhm = np.max(peak_widths(y, peak_idx, rel_height=0.5)[0])
                    # sigma = fwhm/2.4# (STD)
                    
                    
                    mu = dist/2
                    sigma = pitch/2
                    # Initial guesses for the parameters: [A1, A2, mu1, mu2, sigma]
                    initial_guesses = [max(y), max(y), -mu, mu, sigma]
        
                    # Fit the model to the data
                    params, covariance = curve_fit(double_gaussian, x, y,
                                                   p0=initial_guesses, maxfev=10000)
        
                    # Extract fitted parameters
                    A1_fitted, A2_fitted, mu1_fitted, mu2_fitted, sigma_fitted = params
        
                    # Generate fitted curve
                    fitted_signal = double_gaussian(x, A1_fitted, A2_fitted, mu1_fitted, mu2_fitted, sigma_fitted)
    
                    # Calculate P2V
                    P2V_deconv_stack_1d = P2V(fitted_signal)
                    
                        
                if PSD_AND_BUTTERWORTH:
                    
                    # Method 2 - use butter filter + Power Spectrum Density (PSD) #
                    # Filter parameters
                    spatial_sampling_rate = 1
                    spatial_cutoff_frequency = (spatial_sampling_rate/2)/5  # 1/5 of nyquist frequency
                    
                    # analyze 1d signal
                    dynamic_spatial_cutoff_frequency = frequency_analyzer(deconv_stack_1d,
                                                                          spatial_sampling_rate)
                    
                    # Apply Low Pass Filter on signal
                    fitted_signal = butter_lowpass_filter(deconv_stack_1d,
                                                        spatial_sampling_rate,
                                                        dynamic_spatial_cutoff_frequency)
                    
                    # Calculate P2V
                    P2V_deconv_stack_1d = P2V(fitted_signal)
            else:
                # No smoothing #
                P2V_deconv_stack_1d = P2V(deconv_stack_1d)
            
    
            
            ## plot deconv + deconv profile (with rotation) ##
            fig, (ax0, ax1) = plt.subplots(1,2,figsize=(15,7),dpi=600)
            fig.patch.set_facecolor('white')
            # deconv
            deconv_stack = np.flip(deconv_stack)
            deconv_stack_1d = np.flip(deconv_stack_1d)
            # im = ax0.imshow(deconv_stack, extent=[-size/2, size/2, -size/2, size/2])
            im = ax0.imshow(deconv_stack[100:150,100:150],
                            extent=[-25, 25, -25, 25])
            divider = make_axes_locatable(ax0)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            
            # Format colorbar tick labels in scientific notation
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-1, 1))  # You can adjust limits as needed
            cbar.ax.yaxis.set_major_formatter(formatter)
            
            ax0.set_xlabel('x [mm]', fontsize=18)
            ax0.set_ylabel('y [mm]', fontsize=18)
            ax0.set_title('Stacked RL deconvolution')
            # deconv profile
            # ax1.plot(np.arange(-size/2,size/2), deconv_stack_1d,
            #           linewidth=3,color='blue')
            ax1.plot(np.arange(-25,25), deconv_stack_1d[100:150],
                      linewidth=3,color='blue')
            ax1.ticklabel_format(style='sci', axis='y',scilimits=(0,0))
            
            if TO_SMOOTH_SIGNAL:
                ax1.plot(np.arange(-size/2,size/2), fitted_signal,
                          label='fitted signal',linewidth=3, color='black')
                
            # ax1.plot([], [], ' ', label=legend)  # ' ' creates an invisible line
            ax1.set_xlabel('x [mm]', fontsize=18)
            ax1.set_ylabel('photon hits', fontsize=18)
            ax1.set_title('Stacked RL deconvolution profile')
            ax1.grid()
            
            text = (f'P2V={np.around(P2V_deconv_stack_1d,2)}'+
                    f'\nPitch={pitch} mm' +
                    f'\nEL gap={el_gap} mm' +
                    f'\nImmersion={fiber_immersion} mm' +
                    f'\nanode dist={anode_distance} mm')
            ax1.text(0.04, 0.95, text, ha='left', va='top',
                     transform=ax1.transAxes,
                     bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.3'),  # Set edgecolor to 'none'
                     fontsize=13, color='blue', fontweight='bold', linespacing=1.5)


                  
            title = (f' Event spacing={dist} mm,' + 
                     f' Avg RL iterations={int(avg_cutoff_iter)},'  +
                     f' Avg RL relative diff={np.around(avg_rel_diff_checkout,4)}')
            



            fig.suptitle(title,fontsize=15,fontweight='bold')
            fig.tight_layout()
            if TO_SAVE:
                # save files and deconv image plot in correct folder #
                spacing = 0.5
                if int(dist) <= dist < int(dist) + spacing:
                    save_path = save_dir + f'/{int(dist)}_mm'
                if int(dist) + spacing <= dist < np.ceil(dist):
                    save_path = save_dir + f'/{int(dist) + spacing}_mm'
                # save_path = save_dir + f'/{int(dist)}_mm'
                
                if not os.path.isdir(save_path):
                    os.mkdir(save_path)
                fig.savefig(save_path+r'/P2V_plot', format='svg')
                P2V_arr = [dist, P2V_deconv_stack_1d]
                np.savetxt(save_path+r'/[distance,P2V].txt', P2V_arr, delimiter=' ', fmt='%s')
            if TO_PLOT_P2V:
                plt.show()
                continue
            plt.close(fig)

            
# In[4.1]
## Test ground for Low Pass Filter to get accurate P2V ##

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch

# Design a Butterworth low-pass filter
def butter_lowpass_filter(signal, fs, cutoff, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, signal)
    return filtered_data



def frequency_analyzer(signal, spatial_sampling_rate, spatial_cutoff_frequency):
    # analyze the frequency content of y and set a cutoff frequency dynamically
    # Power Spectral Density (PSD)
    f, Pxx = welch(signal, fs=spatial_sampling_rate, nperseg=250)
    # Set cutoff frequency at the frequency that captures 95% of the power
    cumulative_power = np.cumsum(Pxx) / np.sum(Pxx)
    cutoff_index = np.where(cumulative_power >= 0.95)[0][0]
    dynamic_spatial_cutoff_frequency = f[cutoff_index]/0.5
    return dynamic_spatial_cutoff_frequency


# Generate sample data
np.random.seed(0)  # For reproducibility
x = np.arange(-size/2, size/2)
y = deconv_stack_1d


# # Filter parameters
spatial_sampling_rate = 1
spatial_cutoff_frequency = (spatial_sampling_rate/2)/5  # 1/5 of nyquist frequency


dynamic_spatial_cutoff_frequency = frequency_analyzer(y, spatial_sampling_rate,
                                                      spatial_cutoff_frequency)

# Apply the dynamically adjusted filter
filtered_signal = butter_lowpass_filter(y, spatial_sampling_rate,                               
                                        dynamic_spatial_cutoff_frequency)


# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Original Signal', color='red', linestyle='dashed')
plt.plot(x, filtered_signal, label='Filtered Signal', color='darkblue')
plt.legend()
plt.title('Low-Pass Filtering of Noisy Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# In[5]
# compare smoothed vs unsmoothed PSF for 2 geometry families:
# one with a round PSF, and one with a square PSF
# This will generate a comparison plot for each family

immersions = np.array([-1, 2, 5]) # geant4 parameters
PSF_shape = 'SQUARE' # or 'SQUARE
if PSF_shape == 'ROUND':
    dists = np.arange(15,34,2)
    # dists = [17,20]
if PSF_shape == 'SQUARE':
    dists = np.arange(5,12)
SAVE_PLOT = False

# Prepare figure for all 3 immersions
fig, ax = plt.subplots(1, 3, figsize=(24,7), dpi=600)

for i,immersion in tqdm(enumerate(immersions)):
    # round PSF geometry (even at most immersion of fibers)
    if PSF_shape == 'ROUND':
        round_PSF_geo = ('/media/amir/Extreme Pro/SquareFiberDatabase/' +
                    f'ELGap=10mm_pitch=15.6mm_distanceFiberHolder={immersion}mm_' +
                    'distanceAnodeHolder=10mm_holderThickness=10mm')
        geo_dir = round_PSF_geo
        
    if PSF_shape == 'SQUARE':
        # square PSF geometry when fibers are immersed
        square_PSF_geo = ('/media/amir/Extreme Pro/SquareFiberDatabase/' +
                    f'ELGap=1mm_pitch=5mm_distanceFiberHolder={immersion}mm_'+
                    'distanceAnodeHolder=2.5mm_holderThickness=10mm')
        geo_dir = square_PSF_geo


    print(geo_dir)
    working_dir = geo_dir + r'/combined_event_SR'
    output_dir = geo_dir + r'/P2V'
    dist_dirs = glob.glob(working_dir + '/*')
    
    # grab geometry parameters for plot
    geo_params = geo_dir.split('/SquareFiberDatabase/')[-1]
    el_gap = float(re.search(r"ELGap=(-?\d+\.?\d*)mm",
                             geo_params).group(1))
    pitch = float(re.search(r"pitch=(-?\d+\.?\d*)mm",
                            geo_params).group(1))
    
    fiber_immersion = float(re.search(r"distanceFiberHolder=(-?\d+\.?\d*)mm",
                                      geo_params).group(1))
    anode_distance = float(re.search(r"distanceAnodeHolder=(-?\d+\.?\d*)mm",
                                     geo_params).group(1))
    holder_thickness = float(re.search(r"holderThickness=(-?\d+\.?\d*)mm",
                                       geo_params).group(1))
    
    # convert from a Geant4 parameter to a more intuitive one
    fiber_immersion = 5 - fiber_immersion 
    
    PSF = np.load(geo_dir + '/PSF.npy')
    points_unsmoothed = []
    points_smoothed = []
    

    for j,dist in tqdm(enumerate(dists)):
        dist_dir = find_subdirectory_by_distance(working_dir, dist)
    
        # print(f'Working on:\n{dist_dir}')
        event_files = glob.glob(dist_dir + '/*.npy')
        deconv_stack_smoothed = np.zeros((size,size))
        deconv_stack_unsmoothed = np.zeros((size,size))
        
        
        for event_file in event_files:
            event = np.load(event_file)
            
            ##### interpolation #####
    
            # Create a 2D histogram
            hist, x_edges, y_edges = np.histogram2d(event[:,0], event[:,1],
                                                    range=[[-size/2, size/2],
                                                           [-size/2, size/2]],
                                                    bins=bins)
    
    
            # Compute the centers of the bins
            x_centers = (x_edges[:-1] + x_edges[1:]) / 2
            y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    
            hist_hits_x_idx, hist_hits_y_idx = np.where(hist>0)
            hist_hits_x = x_centers[hist_hits_x_idx]
            hist_hits_y = y_centers[hist_hits_y_idx]
            hist_hits_vals = hist[hist>0]
    
    
            # Define the interpolation grid
            x_range = np.linspace(-size/2, size/2, num=bins)
            y_range = np.linspace(-size/2, size/2, num=bins)
            x_grid, y_grid = np.meshgrid(x_range, y_range)
    
            # Perform the interpolation
            interp_img = griddata((hist_hits_x, hist_hits_y), hist_hits_vals,
                                  (x_grid, y_grid), method='cubic', fill_value=0)
    
    
            # optional, cut interp image values below 0
            interp_img[interp_img<0] = 0
    
    
            ##### RL deconvolution UNSMOOTHED #####
            _, _, deconv_unsmoothed = richardson_lucy(interp_img, PSF,
                                                      iterations=75, iter_thr=0.01)
            
        
            ##### ROTATE #####
            theta = extract_theta_from_path(event_file)
            rot_matrix = np.array([[np.cos(theta),-np.sin(theta)],
                                    [np.sin(theta),np.cos(theta)]])
    
            # rotate combined event AFTER deconv
            rotated_deconv_unsmoothed = rotate(deconv_unsmoothed, np.degrees(theta), reshape=False, mode='nearest')
            deconv_stack_unsmoothed += rotated_deconv_unsmoothed
            
       
            
            ##### RL deconvolution SMOOTHED #####
            _, _, deconv_smoothed = richardson_lucy(interp_img,smooth_PSF(PSF),
                                                    iterations=75, iter_thr=0.01)
            
            
            ##### ROTATE #####
            theta = extract_theta_from_path(event_file)
            rot_matrix = np.array([[np.cos(theta),-np.sin(theta)],
                                    [np.sin(theta),np.cos(theta)]])
    
            # rotate combined event AFTER deconv
            rotated_deconv_smoothed = rotate(deconv_smoothed, np.degrees(theta), reshape=False, mode='nearest')
            deconv_stack_smoothed += rotated_deconv_smoothed
    
                
    
        # P2V deconv unsmoothed
        x_cm, y_cm = ndimage.measurements.center_of_mass(deconv_stack_unsmoothed)
        x_cm, y_cm = int(x_cm), int(y_cm)
        deconv_stack_1d_unsmoothed = deconv_stack_unsmoothed[y_cm,:]
        P2V_deconv_stack_unsmoothed = P2V(deconv_stack_1d_unsmoothed)
        
        point_unsmoothed = [dist, P2V_deconv_stack_unsmoothed]
        points_unsmoothed.append(point_unsmoothed)
        
        
        # P2V deconv smoothed
        x_cm, y_cm = ndimage.measurements.center_of_mass(deconv_stack_smoothed)
        x_cm, y_cm = int(x_cm), int(y_cm)
        deconv_stack_1d_smoothed = deconv_stack_smoothed[y_cm,:]
        P2V_deconv_stack_smoothed = P2V(deconv_stack_1d_smoothed)
        
        point_smoothed = [dist, P2V_deconv_stack_smoothed]
        points_smoothed.append(point_smoothed)
            
        
    points_unsmoothed = np.vstack(points_unsmoothed)
    points_smoothed = np.vstack(points_smoothed)

    # plot P2V vs dist for differebt fiber immersions
    ax[i].plot(points_unsmoothed[:,0], points_unsmoothed[:,1],
            '--r^', label="unsmoothed PSF")
    ax[i].plot(points_smoothed[:,0], points_smoothed[:,1],
            '--g*', label="smoothed PSF")
    ax[i].set_xlabel("distance [mm]")
    ax[i].set_ylabel("P2V")
    ax[i].grid()
    ax[i].legend(loc='lower right',fontsize=10)
    ax[i].set_title(f'Fiber immersion={fiber_immersion}mm')
 
    
title = (f'EL gap={el_gap}mm, pitch={pitch}mm, anode distance={anode_distance}mm,' +
          f' holder thickness={holder_thickness}mm, PSF shape={PSF_shape}') 
fig.suptitle(title, fontsize=12)
if SAVE_PLOT:
    save_path = (r'/media/amir/Extreme Pro/miscellaneous/smoothed_vs_unsmoothed_PSF' + 
                 f'/{PSF_shape}_PSF')
    fig.savefig(save_path, format='svg')
plt.show()
        


# In[10]

immersions = np.array([0, 3, 6])
dists = [20,25]
# dists = np.arange(15,50,3) # round
# dists = np.arange(5,40,3) # square
# immersions = 5 - immersions

fig, ax = plt.subplots(1, 3, figsize=(24,7), dpi=600)
for i,immersion in tqdm(enumerate(immersions)):
    x = np.linspace(10,80,30)
    y = np.linspace(1,1000,30)
    
    ax[i].plot(x, y,
            '--r^', label="unsmoothed PSF")
    ax[i].plot(points_smoothed[:,0], points_smoothed[:,1],
            '--g*', label="smoothed PSF")
    ax[i].set_xlabel("distance [mm]")
    ax[i].set_ylabel("P2V")
    ax[i].grid()
    ax[i].legend(loc='lower right',fontsize=10)
    
    title = (f'EL gap={el_gap}mm, pitch={pitch}mm, anode distance={anode_distance}mm,' +
              ' holder thickness={holder_thickness}mm') 
    ax[i].set_title(f'Fiber immersion={immersion}mm')

fig.suptitle(title, fontsize=12)    
plt.show()

# In[9]
'''
combine events, interpolate and RL
This shows 1 sample at a time for a chosen m,n shift values for different geometries
for each geometry:
sample 2 events -> shift 1 of them to (randint(0,max_n),randint(0,max_n))*(x2,y2)
-> make sensor response, save distance (example 16-17mm, 17-18mm), rotate,
interpolate, RL and Peak to Valley (P2V)
'''


TO_PLOT = True

# override previous bins/size settings
bins = 250
size = bins
random_shift = False
if random_shift == False:
    m, n = 2, 2
seed = random.randint(0,10**9)
# seed = 428883858
random.seed(seed)
np.random.seed(seed)

x_match_str = r"_x=(-?\d+(?:\.\d+)?(?:e-?\d+)?)mm"
y_match_str = r"_y=(-?\d+(?:\.\d+)?(?:e-?\d+)?)mm"

# for i,geo_dir in tqdm(enumerate(geometry_dirs)):
# geo_dir = geometry_dirs[-1]

geo_dir = ('/media/amir/Extreme Pro/SquareFiberDatabase/' +
            'ELGap=1mm_pitch=5mm_distanceFiberHolder=5mm_distanceAnodeHolder=5mm_holderThickness=10mm')

# geo_dir = str(random.sample(geometry_dirs, k=1)[0])

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
# PSF = smooth_PSF(PSF)

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
if random_shift:
    m, n = np.random.randint(1,4), np.random.randint(1,4) 


shifted_event_SR = event_to_shift_SR + [m*pitch, n*pitch]
# Combine the two events
combined_event_SR = np.concatenate((event_to_stay_SR, shifted_event_SR))
shifted_event_coord = np.array([x1, y1]) + [m*pitch, n*pitch]

# get distance between stay and shifted
dist = (np.sqrt((x0-shifted_event_coord[0])**2+(y0-shifted_event_coord[1])**2))

# get midpoint of stay and shifted
midpoint = [(x0+shifted_event_coord[0])/2,(y0+shifted_event_coord[1])/2]
print(f'distance = {dist}mm')
# print(f'midpoint = {midpoint}mm')

theta = np.arctan2(y0-shifted_event_coord[1],x0-shifted_event_coord[0])
rot_matrix = np.array([[np.cos(theta),-np.sin(theta)],
                        [np.sin(theta),np.cos(theta)]])

# center combined event using midpoint
centered_combined_event_SR = combined_event_SR - midpoint


##### interpolation #####

# Create a 2D histogram
hist, x_edges, y_edges = np.histogram2d(centered_combined_event_SR[:,0],
                                        centered_combined_event_SR[:,1],
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
interp_img = griddata((hist_hits_x, hist_hits_y), hist_hits_vals,
                      (x_grid, y_grid), method='cubic', fill_value=0)

# optional, cut interp image values below 0
interp_img[interp_img<0] = 0



##### RL deconvolution #####
# rel_diff_checkout, cutoff_iter, deconv = richardson_lucy(interp_img, PSF,
#                                                   iterations=75, iter_thr=0.05)
rel_diff_checkout, cutoff_iter, deconv = richardson_lucy(interp_img, PSF,
                                                  iterations=75, iter_thr=0.01)


##### ROTATE #####

# rotate combined event BEFORE deconv
# points = np.column_stack((x_grid.ravel(), y_grid.ravel())) # Flatten the grids

# rotated_points = np.dot(points, rot_matrix.T) # Rotate each point

# # Reshape rotated points back into 2D grid
# x_rotated = rotated_points[:, 0].reshape(x_grid.shape)
# y_rotated = rotated_points[:, 1].reshape(y_grid.shape)

# # Perform the interpolation on the rotated grid
# rotated_interp_img = griddata((hist_hits_x, hist_hits_y), hist_hits_vals,
#                               (x_rotated, y_rotated),
#                               method='cubic', fill_value=0)
# rel_diff_checkout, cutoff_iter, deconv = richardson_lucy(rotated_interp_img, PSF,
#                                                   iterations=75, iter_thr=0.01)


# rotate combined event AFTER deconv
rotated_deconv = rotate(deconv, np.degrees(theta), reshape=False, mode='nearest')


print(f'rel_diff = {rel_diff_checkout}')
print(f'cut off iteration = {cutoff_iter}')


# Not relevant for now
# ##### P2V #####
# # try P2V without deconv
# x_cm, y_cm = ndimage.measurements.center_of_mass(rotated_interp_img)
# x_cm, y_cm = int(x_cm), int(y_cm)
# interp_img_1d = rotated_interp_img[y_cm,:]
# avg_P2V_interp = P2V(interp_img_1d)
# print(f'avg_P2V_interp = {avg_P2V_interp}')


# # try P2V with deconv
# # print(f'min deconv = {np.around(np.min(deconv),3)}')
# x_cm, y_cm = ndimage.measurements.center_of_mass(rotated_deconv)
# x_cm, y_cm = int(x_cm), int(y_cm)
# rotated_deconv_1d = rotated_deconv[y_cm,:]
# avg_P2V_deconv = P2V(rotated_deconv_1d)
# print(f'avg_P2V_deconv = {avg_P2V_deconv}')
# print(f'seed = {seed}')

# # deconvolution diverges
# if rel_diff_checkout >= 1 :
#     chosen_avg_P2V = np.around(avg_P2V_interp,3)
#     print('\n\nDeconvolution process status: FAIL - diverging' +
#           f'\nInterpolation P2V outperforms, avg_P2V={chosen_avg_P2V}')
# # deconvolution converges
# if rel_diff_checkout < 1:
#     if avg_P2V_deconv >= avg_P2V_interp:
#         chosen_avg_P2V = np.around(avg_P2V_deconv,3)
#         print('\n\nDeconvolution process status: SUCCESS - converging' + 
#               f'\nDeconvolution P2V outperforms, avg_P2V={chosen_avg_P2V}')
#     if avg_P2V_deconv < avg_P2V_interp: # deconvolution converges but didn't outperform interp P2V
#         chosen_avg_P2V = np.around(avg_P2V_interp,3)
#         print('\n\nDeconvolution process status: SUCCEED - converging' + 
#               f'\nInterpolation P2V outperforms, avg_P2V={chosen_avg_P2V}')
#     if avg_P2V_deconv == 0 and avg_P2V_interp == 0:
#         chosen_avg_P2V = -1
#         print('Could not find a P2V value. Check sensor response image.')
        



if TO_PLOT:
    # plot sensor responses
    plot_sensor_response(event_to_stay_SR,bins,size)
    plot_sensor_response(event_to_shift_SR,bins,size)
    plot_sensor_response(combined_event_SR, bins, size)
    plot_sensor_response(centered_combined_event_SR, bins, size)
    
    # plot interpolated combined event (no rotation)
    # Transpose the array to correctly align the axes
    plt.imshow(interp_img, extent=[-size/2, size/2, -size/2, size/2],
               vmin=0, origin='lower')
    plt.colorbar(label='Photon hits')
    plt.title('Cubic Interpolation of Combined event')
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    plt.show()
    
    # plot deconvoltion of combined event (no rotation)
    plt.imshow(deconv, extent=[-size/2, size/2, -size/2, size/2],
               vmin=0, origin='lower')
    plt.colorbar(label='Photon hits')
    plt.title('Deconvolution of combined event')
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    plt.show()
    
    # # plot PSF
    # plt.imshow(PSF,vmin=0)
    # plt.colorbar()
    # plt.xlabel('x [mm]')
    # plt.ylabel('y [mm]')
    # plt.title('PSF')
    # plt.show()
    
    plt.imshow(rotated_deconv, extent=[-size/2, size/2, -size/2, size/2],
               vmin=0, origin='lower')
    plt.colorbar(label='Photon hits')
    plt.title('Rotated deconvolution')
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    plt.show()
    
    # ## plot deconv + deconv profile (with rotation) ##
    # fig, ax = plt.subplots(2,2,figsize=(15,13))
    # im = ax[0,0].imshow(rotated_interp_img, extent=[-size/2, size/2, -size/2, size/2])
    # divider = make_axes_locatable(ax[0,0])
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(im, cax=cax)
    # ax[0,0].set_xlabel('x [mm]')
    # ax[0,0].set_ylabel('y [mm]')
    # ax[0,0].set_title('Rotated interpolated image')

    # legend = f'Avg P2V={np.around(avg_P2V_interp,3)}'
    # ax[0,1].plot(np.arange(-size/2, size/2), interp_img_1d,label=legend)
    # ax[0,1].set_xlabel('x [mm]')
    # ax[0,1].set_ylabel('photon hits')
    # ax[0,1].set_title('Rotated interpolated image profile')
    # ax[0,1].grid()
    # ax[0,1].legend(fontsize=10)
    # # ax[0,1].set_ylim([0,None])
    
    # # deconv
    # im = ax[1,0].imshow(rotated_deconv, extent=[-size/2, size/2, -size/2, size/2])
    # divider = make_axes_locatable(ax[1,0])
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(im, cax=cax)
    # ax[1,0].set_xlabel('x [mm]')
    # ax[1,0].set_ylabel('y [mm]')
    # ax[1,0].set_title('Rotated RL deconvolution')
    # # deconv profile
    # legend = f'Avg P2V={np.around(avg_P2V_deconv,3)}'
    # ax[1,1].plot(np.arange(-size/2, size/2), rotated_deconv_1d,label=legend)
    # ax[1,1].set_xlabel('x [mm]')
    # ax[1,1].set_ylabel('photon hits')
    # ax[1,1].set_title('Rotated RL deconvolution profile')
    # ax[1,1].grid()
    # ax[1,1].legend(fontsize=10)
    # # ax[1,1].set_ylim([0,None])
    # geo_params = geo_dir.split('/SquareFiberDatabase/')[-1]
    # title = (f'{geo_params}\nEvent spacing = {np.around(dist,3)}[mm], ' +
    #          f'cutoff_iter={cutoff_iter}, rel_diff={np.around(rel_diff_checkout,4)}')
    # fig.suptitle(title,fontsize=15)
    # fig.tight_layout()
    # plt.show()

