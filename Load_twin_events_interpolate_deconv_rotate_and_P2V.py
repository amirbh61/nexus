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
# from invisible_cities.reco.deconv_functions     import richardson_lucy
from scipy.interpolate import interp2d
from scipy.interpolate import griddata
from scipy.interpolate import RectBivariateSpline
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import rotate
import random
# Global settings #


n_sipms = 25 # DO NOT CHANGE THIS VALUE
n_sipms_per_side = (n_sipms-1)/2
size = 250
bins = size


path_to_dataset = '/media/amir/Extreme Pro/SquareFiberDatabase'

# List full paths of the Geant4_PSF_events folders inside SquareFiberDatabase
geometry_dirs = [os.path.join(path_to_dataset, d) for d in os.listdir(path_to_dataset)
                 if os.path.isdir(os.path.join(path_to_dataset, d))]
                 
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
""

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
    size = 100 # value override, keep the same for all PSF histograms
    bins = 100 # value override, keep the same for all PSF histograms
    
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

    # # Transpose the array to correctly align the axes
    sipm_assigned_event = sipm_assigned_event.T

    plt.imshow(sipm_assigned_event,
                extent=[-size/2, size/2, -size/2, size/2], vmin=0, origin='lower')
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


def peaks(array):
    fail = 0
    hight_threshold = 0.1*max(array) # ignore peaks of 10% max peak
    peak_idx, properties = find_peaks(array, height = hight_threshold)
    if len(peak_idx) == 0 or len(peak_idx) == 1:
        fail = 1 # no peaks
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
        # print(r'Could not find any peaks for event!')
        return 1
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
    
    
def find_subdirectory_by_distance(directory, user_distance):
    """
    Finds the subdirectory that corresponds to the 
    user-specified distance within a given directory.
    
    Parameters
    ----------
    directory : str
        The directory to search in.
    user_distance : int
        The user-specified distance.
    
    Returns
    -------
    str or None
        The subdirectory corresponding to the user-specified distance,
        or None if not found.
    """
    for entry in os.scandir(directory):
        if entry.is_dir():
            match = re.search(r'(\d+)_mm', entry.name)
            if match and int(match.group(1)) == user_distance:
                return entry.path
    return None


def extract_theta_from_path(file_path):
    """
    Extracts theta value from a given file path of the form "8754_rotation_angle_rad=-2.84423.npy"

    Parameters
    ----------
    file_path : str
        The file path string.

    Returns
    -------
    float
        The extracted theta value.
    """
    try:
        # Splitting by '_' and then further splitting by '='
        parts = file_path.split('_')
        theta_part = parts[-1]  # Get the last part which contains theta
        theta_str = theta_part.split('=')[-1]  # Split by '=' and get the last part
        theta_str = theta_str.replace('.npy', '')  # Remove the .npy extension
        return float(theta_str)
    except Exception as e:
        print(f"Error extracting theta from path: {file_path}. Error: {e}")
        return None


# # tests for function assign_hit_to_SiPM
# test_cases = [
#     ((x, y), pitch, n)
#     for x in np.linspace(-10, 10, 20)
#     for y in np.linspace(-10, 10, 20)
#     for pitch in [5,10,15.6]
#     for n in [n_sipms]
# ]

# # Compare the outputs of the two functions
# for hit, pitch, n in test_cases:
#     result_original = assign_hit_to_SiPM_original(hit, pitch, n)
#     result_optimized = assign_hit_to_SiPM(hit, pitch, n)

#     if result_original != result_optimized:
#         print(f"Discrepancy found for hit {hit}, pitch {pitch}, n {n}:")
#         print(f"  Original: {result_original}, Optimized: {result_optimized}")

# # If no output, then the two functions are consistent for the test cases
# print("Test completed.")
           
                 
# In[7]
'''
Load twin events after shifted and centered.
interpolate, deconv, rotate and P2V.
Load_twin_events_interpolate_deconv_rotate_and_P2V.py
'''

TO_GENERATE = True
TO_SAVE = True
TO_PLOT_EACH_STEP = False
TO_PLOT_P2V = False
TO_SMOOTH_PSF = False

if TO_GENERATE:
    for geo_dir in tqdm(geometry_dirs):

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
    
        # assign input and output directories
        print(geo_dir)
        working_dir = geo_dir + r'/combined_event_SR'
        
        # dist_dirs = glob.glob(working_dir + '/*')
        
        ### This is only for completing the dataset, TO DELETE LATER ###
        # Set ranges for different pitch values
        if pitch == 5:
            valid_dirs = [f"{i}_mm" for i in range(1, 5)]
        elif pitch == 10:
            valid_dirs = [f"{i}_mm" for i in range(1, 10)]
        elif pitch == 15.6:
            valid_dirs = [f"{i}_mm" for i in range(1, 15)]
        
        # Build the full paths
        dist_dirs = [os.path.join(working_dir, d) for d in valid_dirs if os.path.isdir(os.path.join(working_dir, d))]
    
        PSF = np.load(geo_dir + '/PSF.npy')
        if TO_SMOOTH_PSF:
            PSF = smooth_PSF(PSF)

        # option to choose a single distance and see P2V
        # dist = 22
        # user_chosen_dir = find_subdirectory_by_distance(working_dir, dist)
        
        for dist_dir in tqdm(dist_dirs):
            print(dist_dir)
            
            # print(f'Working on:\n{dist_dir}')
            match = re.search(r'/(\d+)_mm', dist_dir)
            if match:
                dist = int(match.group(1))
                
            # dist_dir = user_chosen_dir
            # print(f'Working on:\n{dist_dir}')
            deconv_stack = np.zeros((size,size))
            cutoff_iter_list = []
            rel_diff_checkout_list = []
            event_files = glob.glob(dist_dir + '/*.npy')
            
            
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


                ##### RL deconvolution #####
                rel_diff_checkout, cutoff_iter, deconv = richardson_lucy(interp_img, PSF,
                                                                  iterations=75, iter_thr=0.01)
                
                
                cutoff_iter_list.append(cutoff_iter)
                rel_diff_checkout_list.append(rel_diff_checkout)
                

                ##### ROTATE #####
                theta = extract_theta_from_path(event_file)
                rot_matrix = np.array([[np.cos(theta),-np.sin(theta)],
                                        [np.sin(theta),np.cos(theta)]])

                # # rotate combined event BEFORE deconv
                # points = np.column_stack((x_grid.ravel(), y_grid.ravel())) # Flatten the grids

                # rotated_points = np.dot(points, rot_matrix.T) # Rotate each point

                # # Reshape rotated points back into 2D grid
                # x_rotated = rotated_points[:, 0].reshape(x_grid.shape)
                # y_rotated = rotated_points[:, 1].reshape(y_grid.shape)

                # # Perform the interpolation on the rotated grid
                # rotated_interp_img = griddata((hist_hits_x, hist_hits_y), hist_hits_vals,
                #                               (x_rotated, y_rotated),
                #                               method='cubic', fill_value=0)


                # rotate combined event AFTER deconv
                rotated_deconv = rotate(deconv, np.degrees(theta), reshape=False, mode='nearest')
                deconv_stack += rotated_deconv


                
                if TO_PLOT_EACH_STEP:
                    
                    # plot SR combined event
                    plt.imshow(hist, extent=[-size/2, size/2, -size/2, size/2],
                                vmin=0, origin='lower')
                    plt.colorbar(label='Photon hits')
                    plt.title('SR Combined event')
                    plt.xlabel('x [mm]')
                    plt.ylabel('y [mm]')
                    plt.show()
                    
                    # plot interpolated combined event
                    plt.imshow(interp_img, extent=[-size/2, size/2, -size/2, size/2],
                                vmin=0, origin='lower')
                    plt.colorbar(label='Photon hits')
                    plt.title('Cubic Interpolation of Combined event')
                    plt.xlabel('x [mm]')
                    plt.ylabel('y [mm]')
                    plt.show()
                    
                    # plot RL deconvolution
                    plt.imshow(deconv, extent=[-size/2, size/2, -size/2, size/2],
                                vmin=0, origin='lower')
                    plt.colorbar(label='Photon hits')
                    plt.title('RL deconvolution')
                    plt.xlabel('x [mm]')
                    plt.ylabel('y [mm]')
                    plt.show()
                    
                    # plot ROTATED RL deconvolution
                    plt.imshow(rotated_deconv, extent=[-size/2, size/2, -size/2, size/2],
                                vmin=0, origin='lower')
                    plt.colorbar(label='Photon hits')
                    plt.title('Rotated RL deconvolution')
                    plt.xlabel('x [mm]')
                    plt.ylabel('y [mm]')
                    plt.show()
                    
                    # plot deconvolution stacking
                    plt.imshow(deconv_stack, extent=[-size/2, size/2, -size/2, size/2],
                                vmin=0, origin='lower')
                    plt.colorbar(label='Photon hits')
                    plt.title('Accomulated RL deconvolution')
                    plt.xlabel('x [mm]')
                    plt.ylabel('y [mm]')
                    plt.show()
                
            
            avg_cutoff_iter = np.mean(cutoff_iter_list)
            avg_rel_diff_checkout = np.mean(rel_diff_checkout_list)

            
            # P2V deconv
            # print(f'min deconv = {np.around(np.min(deconv),3)}')
            x_cm, y_cm = ndimage.measurements.center_of_mass(deconv_stack)
            x_cm, y_cm = int(x_cm), int(y_cm)
            deconv_stack_1d = deconv_stack[y_cm,:]
            P2V_deconv_stack = P2V(deconv_stack_1d)
            # print(f'\nP2V_deconv_stack = {P2V_deconv_stack}')
            
            # ##### P2V #####
            # # try P2V without deconv
            # x_cm, y_cm = ndimage.measurements.center_of_mass(rotated_interp_img)
            # x_cm, y_cm = int(x_cm), int(y_cm)
            # interp_img_1d = rotated_interp_img[y_cm,:]
            # avg_P2V_interp = P2V(interp_img_1d)
            # # print(f'avg_P2V_interp = {avg_P2V_interp}')
            

            
            ## plot deconv + deconv profile (with rotation) ##
            fig, (ax0, ax1) = plt.subplots(1,2,figsize=(15,7),dpi=600)
            # deconv
            im = ax0.imshow(deconv_stack, extent=[-size/2, size/2, -size/2, size/2])
            divider = make_axes_locatable(ax0)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            ax0.set_xlabel('x [mm]')
            ax0.set_ylabel('y [mm]')
            ax0.set_title('Stacked RL deconvolution')
            # deconv profile
            legend = f'P2V={np.around(P2V_deconv_stack,3)}'
            ax1.plot(np.arange(-size/2, size/2), deconv_stack_1d,label=legend)
            ax1.set_xlabel('x [mm]')
            ax1.set_ylabel('photon hits')
            ax1.set_title('Stacked RL deconvolution profile')
            ax1.grid()
            ax1.legend(fontsize=10)
            # ax[1,1].set_ylim([0,None])
                  
            title = (f'EL gap={el_gap}mm, pitch={pitch}mm,' + 
                      f' fiber immersion={fiber_immersion}mm, anode distance={anode_distance}mm,' + 
                      f' holder thickness={holder_thickness}mm\n\nEvent spacing={dist}mm,' + 
                    f' Avg RL iterations={int(avg_cutoff_iter)},'  +
                    f' Avg RL relative diff={np.around(avg_rel_diff_checkout,4)}')
            
            # title = (f'{geo_params}\n\nEvent spacing={dist}mm,' + 
            #          f' samples={len(event_files)},' +
            # f' avg RL iterations={int(avg_cutoff_iter)},'  +
            # f' avg RL relative diff={np.around(avg_rel_diff_checkout,4)}')
            
            
            fig.suptitle(title,fontsize=15)
            fig.tight_layout()
            if TO_SAVE:
                fig.savefig(dist_dir+r'/P2V_plot', format='svg')
                P2V_arr = [dist, P2V_deconv_stack]
                np.savetxt(dist_dir+r'/[distance,P2V].txt', P2V_arr, delimiter=' ', fmt='%s')
            if TO_PLOT_P2V:
                plt.show()
                continue
            plt.close(fig)
                 
                 

