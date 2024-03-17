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
from scipy.signal import butter, filtfilt, welch
from scipy.ndimage import rotate
from scipy.optimize import curve_fit
import random

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
'''
FUNCTIONS
'''

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





def psf_creator(directory, create_from, to_plot=False,to_smooth=False):
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
    if create_from == 'SiPM':
        sub_dir = '/Geant4_Kr_events'
    if create_from == 'TPB':
        sub_dir = '/Geant4_PSF_events'
        
    os.chdir(directory)
    print(r'Working on directory:'+f'\n{os.getcwd()}')
    
    files = glob.glob(directory + sub_dir + f'/{create_from}*')
    
    PSF_list = []
    size = 100 # value override, keep the same for all PSF histograms
    bins = 100 # value override, keep the same for all PSF histograms
    
    # For DEBUGGING
    plot_event = False
    plot_sipm_assigned_event = False
    plot_shifted_event = False
    plot_accomulated_events = False
    
    
    # Search for the pitch value pattern
    match = re.search(r"_pitch=(\d+(?:\.\d+)?)mm", directory)
    pitch = float(match.group(1))

    ### assign each hit to its corresponding SiPM ###
    for filename in files:

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

        
        # pattern to extract x,y values of each event from file name
        x_pattern = r"x=(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)mm"
        y_pattern = r"y=(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)mm"
        
        # Store x,y values of event
        x_match = re.findall(x_pattern, filename)
        x_event = float(x_match[0])
        y_match = re.findall(y_pattern, filename)
        y_event = float(y_match[0])
        # print(f'x,y=({x_event},{y_event})')
        
        if plot_event:
            single_event, x_hist, y_hist = np.histogram2d(hitmap[:,0], hitmap[:,1],
                                                  range=[[-size/2,size/2],[-size/2,size/2]],
                                                  bins=bins)
            single_event = single_event.T
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
            shifted_event = shifted_event.T
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
            PSF = PSF.T
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


def plot_sensor_response(event, title, bins, size, noise=False):
    sipm_assigned_event, x_hist, y_hist = np.histogram2d(event[:,0], event[:,1],
                                                         range=[[-size/2, size/2], [-size/2, size/2]],
                                                         bins=bins)  
    if noise:
        sipm_assigned_event = np.random.poisson(sipm_assigned_event)

    # # Transpose the array to correctly align the axes
    sipm_assigned_event = sipm_assigned_event.T

    plt.imshow(sipm_assigned_event,
                extent=[-size/2, size/2, -size/2, size/2], vmin=0, origin='lower')
    if title!="":
        plt.title(title)
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
    # Define a height threshold as 10% of the maximum value in the vector
    height_threshold = 0.1 * np.max(vector)
    
    # Use scipy.signal.find_peaks with the height threshold
    peak_idx, properties = find_peaks(vector, height=height_threshold)
    
    # Check if there are less than 2 peaks, indicating failure to find valid peaks
    if len(peak_idx) < 2:
        # print(r'Could not find two valid peaks for event!')
        return 1
    else:
        # Extract peak heights
        heights = properties['peak_heights']
        
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
        
        # Find the minimum value (valley) between the two strongest peaks
        valley_height = np.min(vector[left_idx:right_idx + 1])
        
        if valley_height <= 0 and avg_peak > 0:
            return float('inf')
        
        # Calculate and return the average peak to valley ratio
        avg_P2V = avg_peak / valley_height
        return avg_P2V

    
    
def find_subdirectory_by_distance(directory, user_distance):
    """
    Finds the subdirectory that corresponds to the 
    user-specified distance within a given directory, supporting both
    integer and floating-point distances.
    
    Parameters
    ----------
    directory : str
        The directory to search in.
    user_distance : float
        The user-specified distance.
    
    Returns
    -------
    str or None
        The subdirectory corresponding to the user-specified distance,
        or None if not found.
    """
    # Convert user_distance to a float for comparison
    user_distance = float(user_distance)
    
    for entry in os.scandir(directory):
        if entry.is_dir():
            # Updated regex to capture both integer and floating-point numbers
            match = re.search(r'(\d+(?:\.\d+)?)_mm', entry.name)
            if match and abs(float(match.group(1)) - user_distance) < 1e-6:
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
    
    
def extract_dir_number(dist_dir):
    # Adjusted regex to match the number at the end of the path before "_mm"
    match = re.search(r'/(\d+(?:\.\d+)?)_mm$', dist_dir)
    if match:
        return float(match.group(1))
    return 0  # Default to 0 if no number found


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

# In[2]
'''
 Generate PSF and save
'''

path_to_dataset = r'/media/amir/9C33-6BBD/NEXT_work/Geant4/nexus/' + \
                   r'small_cluster_hitpoints_dataset/SquareFiberMacrosAndOutputsCloseEL10KSources'
                  
# path_to_dataset = r'/media/amir/Extreme Pro/SquareFiberDatabase'
geometry_dirs = os.listdir(path_to_dataset)
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
# plt.colorbar(im, cax=cax)False

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

TO_GENERATE = True
TO_PLOT = True
TO_SAVE = False

if TO_GENERATE:
    for directory in tqdm(geometry_dirs[0]):
        directory = ('/media/amir/Extreme Pro/SquareFiberDatabase/' +
                    'ELGap=10mm_pitch=15.6mm_distanceFiberHolder=2mm_' +
                    'distanceAnodeHolder=10mm_holderThickness=10mm')
        PSF_TPB = psf_creator(directory,create_from="TPB",
                              to_plot=TO_PLOT,to_smooth=False)
        os.chdir(directory)
        # os.chdir('..')
        
        if TO_SAVE:
            save_PSF = directory + '/PSF.npy'
            np.save(save_PSF,PSF_TPB)
            
# In[5]
# plot and save all TPB PSFs from SquareFiberDataset in their respective folders
TO_GENERATE = True
TO_SAVE = False
print('Generating PSF plots')
if TO_GENERATE:
    for directory in tqdm(geometry_dirs):
        os.chdir(directory)
        print(r'Working on directory:'+f'\n{os.getcwd()}')
        PSF = np.load(r'PSF.npy')
        fig = plot_PSF(PSF=PSF)
        
        if TO_SAVE:
            save_path = directory + r'/PSF_plot.jpg'
            fig.savefig(save_path)  
            plt.close(fig)
# In[6]
'''
Generate and save twin events dataset, after shifting, centering and rotation
'''

def find_highest_number(directory,file_format='.npy'):
    '''
    Finds the highest data sample number in a directory where files are
    named in the format:
    "intnumber_rotation_angle_rad=value.npy"
    The idea is that if we are adding new data to existing data, then the 
    numbering of oyr newly generated data should continue the existing numbering.

    Parameters
    ----------
    directory : str
        The directory of data to search in.

    Returns
    -------
    highest_number : int
        The number of the last data sample created.
        If not found, returns -1.
    '''
    highest_number = -1

    for entry in os.scandir(directory):
        if entry.is_dir():
            for file in os.scandir(entry.path):
                if file.is_file() and file.name.endswith(file_format):
                    try:
                        # Extracting the number before the first underscore
                        number = int(file.name.split('_')[0])
                        highest_number = max(highest_number, number)
                    except ValueError:
                        # Handle cases where the conversion to int might fail
                        continue

    return highest_number


def count_npy_files(directory):
    '''
    Counts all .npy files in the specified directory and its subdirectories.

    Parameters
    ----------
    directory : str
        The directory to search in.

    Returns
    -------
    int
        The number of .npy files found.
    '''
    npy_file_count = 0

    # Walk through all directories and files in the specified directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.npy'):
                npy_file_count += 1

    return npy_file_count



TO_GENERATE = True
TO_PLOT = True
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
        
        # bad geometry
        geo_dir = ('/media/amir/Extreme Pro/SquareFiberDatabase/' +
                    'ELGap=1mm_pitch=5mm_distanceFiberHolder=5mm_' +
                    'distanceAnodeHolder=10mm_holderThickness=10mm')
        
        # # worst geometry
        # geo_dir = ('/media/amir/Extreme Pro/SquareFiberDatabase/' +
        #             'ELGap=1mm_pitch=15.6mm_distanceFiberHolder=-1mm_' +
        #             'distanceAnodeHolder=2.5mm_holderThickness=10mm')
        
        
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
                plot_sensor_response(event_to_stay_SR, title, bins, size=100)
                title = "Shifted event"
                plot_sensor_response(event_to_shift_SR, title, bins, size=100)
                title = "Combined event"
                plot_sensor_response(combined_event_SR, title, bins, size=100)
                title = "Centered combined event"
                plot_sensor_response(centered_combined_event_SR, title, bins, size=100)
            
        print(f'Created {count_npy_files(save_dir)} events')
            
# In[7]
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
                    'ELGap=10mm_pitch=15.6mm_distanceFiberHolder=2mm_' +
                    'distanceAnodeHolder=10mm_holderThickness=10mm')

        # # overwrite to a specific geometry
        # geo_dir = ('/media/amir/Extreme Pro/SquareFiberDatabase/' +
        #             'ELGap=1mm_pitch=5mm_distanceFiberHolder=-1mm_'+
        #             'distanceAnodeHolder=2.5mm_holderThickness=10mm')
        
        # geo_dir = ('/media/amir/Extreme Pro/SquareFiberDatabase/' +
        #             'ELGap=1mm_pitch=5mm_distanceFiberHolder=2mm_' +
        #             'distanceAnodeHolder=5mm_holderThickness=10mm')

        # # worst geometry
        # geo_dir = ('/media/amir/Extreme Pro/SquareFiberDatabase/' +
        #             'ELGap=1mm_pitch=15.6mm_distanceFiberHolder=-1mm_' +
        #             'distanceAnodeHolder=2.5mm_holderThickness=10mm')
        
        
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
        dist = 25
        user_chosen_dir = find_subdirectory_by_distance(working_dir, dist)
        
        for dist_dir in tqdm(dist_dirs):
            # print(dist_dir)
            
            # print(f'Working on:\n{dist_dir}')
            # match = re.search(r'/(\d+)_mm', dist_dir)
            match = re.search(r'/(\d+(?:\.\d+)?)_mm$', dist_dir)
            if match:
                dist = float(match.group(1))
            dist_dir = user_chosen_dir
            print(f'Working on:\n{dist_dir}')
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

                # rotate combined event AFTER deconv
                rotated_deconv = rotate(deconv, np.degrees(theta), reshape=False, mode='nearest')
                deconv_stack += rotated_deconv
                


                
                if TO_PLOT_EACH_STEP:
                    
                    # plot SR combined event
                    plt.imshow(hist.T, extent=[-size/2, size/2, -size/2, size/2],
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
            
            ## save deconv_stack+avg_cutoff_iter+avg_rel_diff_checkout ###
            if TO_SAVE:
                ## plot deconv##
                fig = plt.figure(figsize=(8,7.5),dpi=600)
                # deconv
                plt.imshow(deconv_stack, extent=[-size/2, size/2, -size/2, size/2])
                plt.colorbar()
                plt.xlabel('x [mm]')
                plt.ylabel('y [mm]')
                plt.title('Stacked RL deconvolution')
                plt.grid()
                    
                title = (f'EL gap={el_gap}mm, pitch={pitch}mm,' + 
                          f' fiber immersion={fiber_immersion}mm,\nanode distance={anode_distance}mm,' + 
                          f' holder thickness={holder_thickness}mm\n\nEvent spacing={dist}mm,' + 
                        f' Avg RL iterations={int(avg_cutoff_iter)},'  +
                        f' Avg RL relative diff={np.around(avg_rel_diff_checkout,4)}')
            
                fig.suptitle(title,fontsize=15)
                fig.tight_layout()
                
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
          
# In[8]
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
TO_PLOT_P2V = False
TO_SMOOTH_SIGNAL = False
if TO_SMOOTH_SIGNAL: # If True, only one option must be True - the other, False.
    DOUBLE_GAUSSIAN_FIT = True # currently, the most promising approach !
    PSD_AND_BUTTERWORTH = False 



if TO_GENERATE:
    for geo_dir in tqdm(geometry_dirs):
        
        # geo_dir = ('/media/amir/Extreme Pro/SquareFiberDatabase/' +
        #             'ELGap=10mm_pitch=15.6mm_distanceFiberHolder=2mm_' +
        #             'distanceAnodeHolder=2.5mm_holderThickness=10mm')
        
        # geo_dir = ('/media/amir/Extreme Pro/SquareFiberDatabase/' +
        #             'ELGap=1mm_pitch=5mm_distanceFiberHolder=2mm_' +
        #             'distanceAnodeHolder=5mm_holderThickness=10mm')

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
        # dist = 24
        # user_chosen_dir = find_subdirectory_by_distance(working_dir, dist)
        
        for dist_dir in dist_dirs:
            print(dist_dir)
            
            # print(f'Working on:\n{dist_dir}')
            match = re.search(r'/(\d+(?:\.\d+)?)_mm$', dist_dir)
            if match:
                dist = float(match.group(1))
                
            # dist_dir = user_chosen_dir
            # print(f'Working on:\n{dist_dir}')
            
            
            ## load deconv_stack+avg_cutoff_iter+avg_rel_diff_checkout ###
            deconv_stack = np.load(dist_dir + '/deconv.npy')
            avg_cutoff_iter = float(np.genfromtxt(dist_dir + '/avg_cutoff_iter.txt'))
            avg_rel_diff_checkout = float(np.genfromtxt(dist_dir + '/avg_rel_diff_checkout.txt'))

            # P2V deconv
            x_cm, y_cm = ndimage.measurements.center_of_mass(deconv_stack)
            x_cm, y_cm = int(x_cm), int(y_cm)
            deconv_stack_1d = deconv_stack[y_cm,:]
        
            
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
            # deconv
            im = ax0.imshow(deconv_stack, extent=[-size/2, size/2, -size/2, size/2])
            divider = make_axes_locatable(ax0)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            ax0.set_xlabel('x [mm]')
            ax0.set_ylabel('y [mm]')
            ax0.set_title('Stacked RL deconvolution')
            # deconv profile
            legend = f'P2V = {np.around(P2V_deconv_stack_1d,3)}'
            ax1.plot(np.arange(-size/2,size/2), deconv_stack_1d,
                      label='original signal',color='red')
            if TO_SMOOTH_SIGNAL:
                ax1.plot(np.arange(-size/2,size/2), fitted_signal,
                          label='fitted signal', color='blue')
            ax1.plot([], [], ' ', label=legend)  # ' ' creates an invisible line
            ax1.set_xlabel('x [mm]')
            ax1.set_ylabel('photon hits')
            ax1.set_title('Stacked RL deconvolution profile')
            ax1.grid()
            ax1.legend(fontsize=10)
                  
            title = (f'EL gap={el_gap}mm, pitch={pitch}mm,' + 
                      f' fiber immersion={fiber_immersion}mm, anode distance={anode_distance}mm,' + 
                      f' holder thickness={holder_thickness}mm\n\nEvent spacing={dist}mm,' + 
                    f' Avg RL iterations={int(avg_cutoff_iter)},'  +
                    f' Avg RL relative diff={np.around(avg_rel_diff_checkout,4)}')
            

            fig.suptitle(title,fontsize=15)
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

            
# In[8.5]
## Low Pass Filter to get accurate P2V ##

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





# In[9.0]
### GRAPHS - compare P2Vs of different geometries ###

def sort_ascending_dists_P2Vs(dists, P2Vs):
    combined = list(zip(dists, P2Vs))
    combined.sort(key=lambda x: x[0])
    sorted_dists, sorted_P2Vs = zip(*combined)
    return sorted_dists, sorted_P2Vs


import matplotlib.cm as cm
# Number of unique colors needed
num_colors = len(geometry_dirs)

# Create a colormap
# 'viridis', 'plasma', 'inferno', 'magma', 'cividis' are good choices for distinct colors
# You can also use 'tab20', 'tab20b', or 'tab20c' for categorical colors
cmap = cm.get_cmap('hsv', num_colors)

# Generate colors from the colormap
colors = [cmap(i) for i in range(num_colors)]
random.shuffle(colors)
markers = ['^', '*', 's', 'o', 'x', '+', 'D']

# Parameter choices to slice to
fiber_immersion_choice = [3,6]
pitch_choice = [15.6]
el_gap_choice = [10]
anode_distance_choice = [10]
holder_thickness_choice = [10]
count = 0
SHOW_ORIGINAL_SPACING = True
POLYNOMIAL_FIT = False


# # best
# fiber_immersion_choice = [3]
# pitch_choice = [15.6]
# el_gap_choice = [10]
# anode_distance_choice = [10]
# holder_thickness_choice = [10]

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
                averaged_dists.append(averaged_dist)
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
plt.xlabel('Distance [mm]')
plt.ylabel('P2V')
plt.ylim([-1,5])
plt.xlim([0,xlim])
plt.grid()
plt.legend(loc='upper left', bbox_to_anchor=(1, 1),
           title='Datasets', fontsize='small', ncol=1)
# fig.suptitle(f'Fiber immersion = {immersion_choice}mm')
plt.show()

print(f'total geometries used: {count}')


# In[9.1]
### GRAPHS - Find optimal anode distance ###  

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
    # xlim = 30
    plt.axhline(y=1,color='red',alpha=0.8)
    plt.xlabel('Anode distance [mm]')
    plt.ylabel('P2V')
    plt.grid()
    plt.legend(loc='upper left',fontsize=10)
    title = (f'Comparison of fiber immmersions for source spacing of {dist}mm' + 
             f'\nEL gap={el_gap_choice[-1]}mm,' +
             f' pitch={pitch_choice[-1]}mm,' +
             f' holder thickness={holder_thickness_choice[-1]}mm')
    fig.suptitle(title)
    plt.show()


# In[9.2]
### GRAPHS - Find optimal pitch ###  

# Parameter choices to slice to - DO NOT CHANGE FOR CURRENT DATASET!
fiber_immersion_choice = [0]
pitch_choice = [5,10,15.6]
el_gap_choice = [10]
anode_distance_choice = [10]
holder_thickness_choice = [10]

color = iter(['red', 'green', 'blue'])
marker = iter(['^', '*', 's'])
SHOW_ORIGINAL_SPACING = True
if SHOW_ORIGINAL_SPACING is False:
    custom_spacing = 2
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
        # Fit the data to a 2nd degree polynomial
        coefficients = np.polyfit(sorted_dists, sorted_P2Vs, 2)
        
        # Generate a polynomial function from the coefficients
        polynomial = np.poly1d(coefficients)
        
        # Generate x values for the polynomial curve (e.g., for plotting)
        sorted_dists_fit = np.linspace(sorted_dists[0], sorted_dists[-1], 100)
        
        # Generate y values for the polynomial curve
        sorted_P2V_fit = polynomial(sorted_dists_fit)
        
        ax.plot(sorted_dists_fit, sorted_P2V_fit, color=next(color), ls='-', 
                    alpha=0.5, label=f'pitch={pitch}mm',linewidth=3, markersize=10)
        
    else:
        ax.plot(sorted_dists, sorted_P2Vs, color=next(color), ls='-', 
                    marker=next(marker),alpha=0.5, label=f'pitch={pitch}mm',
                    linewidth=3, markersize=10)
        
    count += 1
    
title = ('Separation for different pitch values vs sources distance' + 
         f'\nEL gap={el_gap_choice[-1]}mm,' +
         f' fiber immersion={fiber_immersion_choice[-1]}mm,' +
         f' anode distance={anode_distance_choice[-1]}mm,'+ 
         f' holder thickness={holder_thickness_choice[-1]}mm')
plt.xlabel('Distance [mm]')
plt.ylabel('P2V')
plt.grid()
plt.xlim([5,25])
plt.legend(loc='upper left',fontsize=10)
fig.suptitle(title,size=12)
plt.show()
print(f'total geometries used: {count}')


# In[9.3]
### GRAPHS - Find optimal EL gap ###  

# Parameter choices to slice to - DO NOT CHANGE FOR CURRENT DATASET!
fiber_immersion_choice = [3]
pitch_choice = [15.6]
el_gap_choice = [1,10]
anode_distance_choice = [10]
holder_thickness_choice = [10]

color = iter(['red', 'green'])
marker = iter(['^', '*'])
SHOW_ORIGINAL_SPACING = False
if SHOW_ORIGINAL_SPACING is False:
    custom_spacing = 3
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
        # Fit the data to a 2nd degree polynomial
        coefficients = np.polyfit(sorted_dists, sorted_P2Vs, 2)
        
        # Generate a polynomial function from the coefficients
        polynomial = np.poly1d(coefficients)
        
        # Generate x values for the polynomial curve (e.g., for plotting)
        sorted_dists_fit = np.linspace(sorted_dists[0], sorted_dists[-1], 100)
        
        # Generate y values for the polynomial curve
        sorted_P2V_fit = polynomial(sorted_dists_fit)
        ax.plot(sorted_dists_fit, sorted_P2V_fit, color=next(color), ls='-', 
                    marker=next(marker), alpha=0.5, label=f'EL gap={el_gap}mm',
                    linewidth=3, markersize=10)
        
    else:
        ax.plot(sorted_dists, sorted_P2Vs, color=next(color), ls='-', 
                    marker=next(marker),alpha=0.5, label=f'EL gap={el_gap}mm',
                    linewidth=3, markersize=10)
        
    count += 1
    
title = ('P2V of different EL gaps vs sources distance' + 
         f'\npitch={pitch_choice[-1]}mm,' +
         f' fiber immersion={fiber_immersion_choice[-1]}mm,' +
         f' anode distance={anode_distance_choice[-1]}mm,'+ 
         f' holder thickness={holder_thickness_choice[-1]}mm')
plt.xlabel('Distance [mm]')
plt.ylabel('P2V')
plt.grid()
plt.legend(loc='upper left',fontsize=15)
fig.suptitle(title,size=15)
plt.show()
print(f'total geometries used: {count}')

# In[9.3]
### GRAPHS - Find optimal immersion ###  

# Parameter choices to slice to - DO NOT CHANGE FOR CURRENT DATASET!
fiber_immersion_choice = [0,3,6]
pitch_choice = [15.6]
el_gap_choice = [10]
anode_distance_choice = [10]
holder_thickness_choice = [10]

color = iter(['red', 'green', 'blue'])
marker = iter(['^', '*', 's'])
SHOW_ORIGINAL_SPACING = False
if SHOW_ORIGINAL_SPACING is False:
    custom_spacing = 2
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
                print(f'data_pair dist={dist}, P2V={P2V}')
            
            # Only proceed if we have a pair, to avoid index out of range errors
            if len(dists) == custom_spacing and len(P2Vs) == custom_spacing:
                averaged_dist = sum(dists) / custom_spacing
                averaged_P2V = sum(P2Vs) / custom_spacing
                print(f'averaged_dist={averaged_dist}, averaged_P2V={averaged_P2V}',end='\n\n')
                averaged_dists.append(averaged_dist)
                averaged_P2Vs.append(averaged_P2V)
        
        # zip and sort in ascending order
        sorted_dists, sorted_P2Vs = sort_ascending_dists_P2Vs(averaged_dists,
                                                              averaged_P2Vs)
        
    if POLYNOMIAL_FIT:
        # Fit the data to a 2nd degree polynomial
        coefficients = np.polyfit(sorted_dists, sorted_P2Vs, 2)
        
        # Generate a polynomial function from the coefficients
        polynomial = np.poly1d(coefficients)
        
        # Generate x values for the polynomial curve (e.g., for plotting)
        sorted_dists_fit = np.linspace(sorted_dists[0], sorted_dists[-1], 100)
        
        # Generate y values for the polynomial curve
        sorted_P2V_fit = polynomial(sorted_dists_fit)
        
        ax.plot(sorted_dists_fit, sorted_P2V_fit, color=next(color), ls='-', 
                    alpha=0.5, label=f'immersion={fiber_immersion}mm',
                    linewidth=3, markersize=10)
        
    else:
        ax.plot(sorted_dists, sorted_P2Vs, color=next(color), ls='-', 
                    marker=next(marker),alpha=0.5, label=f'immersion={fiber_immersion}mm',
                    linewidth=3, markersize=10)
        
    count += 1
    
title = ('P2V of different immersions vs sources distance' + 
         f'\nEL gap={el_gap_choice[-1]}mm,' +
         f' pitch={pitch_choice[-1]}mm,' +
         f' anode distance={anode_distance_choice[-1]}mm,'+ 
         f' holder thickness={holder_thickness_choice[-1]}mm')
plt.xlabel('Distance [mm]')
plt.ylabel('P2V')
plt.grid()
plt.legend(loc='upper left',fontsize=10)
fig.suptitle(title,size=12)
plt.show()
print(f'total geometries used: {count}')


# In[9]
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
    m, n = 1, 2
seed = random.randint(0,10**9)
# seed = 428883858
random.seed(seed)
np.random.seed(seed)

x_match_str = r"_x=(-?\d+(?:\.\d+)?(?:e-?\d+)?)mm"
y_match_str = r"_y=(-?\d+(?:\.\d+)?(?:e-?\d+)?)mm"

# for i,geo_dir in tqdm(enumerate(geometry_dirs)):
# geo_dir = geometry_dirs[-1]


# # works bad for this geometry
# geo_dir = ('/media/amir/Extreme Pro/SquareFiberDatabase/' +
#             'ELGap=1mm_pitch=5mm_distanceFiberHolder=2mm_distanceAnodeHolder=2.5mm_holderThickness=10mm')

# # worst geometry
# geo_dir = ('/media/amir/Extreme Pro/SquareFiberDatabase/' +
#             'ELGap=1mm_pitch=15.6mm_distanceFiberHolder=-1mm_distanceAnodeHolder=2.5mm_holderThickness=10mm')

geo_dir = ('/media/amir/Extreme Pro/SquareFiberDatabase/' +
            'ELGap=1mm_pitch=5mm_distanceFiberHolder=-1mm_distanceAnodeHolder=2.5mm_holderThickness=10mm')

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

# # Save combined centered event to suitlable folder according to sources distance
# save_dir = save_dir + f'/{int(dist)}mm'
# if not os.path.isdir(save_dir):
#     os.mkdir(save_dir)
# save_path = save_dir + f'/{i}.npy'
# np.save(save_path,centered_combined_event_SR)


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
points = np.column_stack((x_grid.ravel(), y_grid.ravel())) # Flatten the grids

rotated_points = np.dot(points, rot_matrix.T) # Rotate each point

# Reshape rotated points back into 2D grid
x_rotated = rotated_points[:, 0].reshape(x_grid.shape)
y_rotated = rotated_points[:, 1].reshape(y_grid.shape)

# Perform the interpolation on the rotated grid
rotated_interp_img = griddata((hist_hits_x, hist_hits_y), hist_hits_vals,
                              (x_rotated, y_rotated),
                              method='cubic', fill_value=0)


# rotate combined event AFTER deconv
rotated_deconv = rotate(deconv, np.degrees(theta), reshape=False, mode='nearest')


print(f'rel_diff = {rel_diff_checkout}')
print(f'cut off iteration = {cutoff_iter}')


##### P2V #####
# try P2V without deconv
x_cm, y_cm = ndimage.measurements.center_of_mass(rotated_interp_img)
x_cm, y_cm = int(x_cm), int(y_cm)
interp_img_1d = rotated_interp_img[y_cm,:]
avg_P2V_interp = P2V(interp_img_1d)
print(f'avg_P2V_interp = {avg_P2V_interp}')


# try P2V with deconv
# print(f'min deconv = {np.around(np.min(deconv),3)}')
x_cm, y_cm = ndimage.measurements.center_of_mass(rotated_deconv)
x_cm, y_cm = int(x_cm), int(y_cm)
rotated_deconv_1d = rotated_deconv[y_cm,:]
avg_P2V_deconv = P2V(rotated_deconv_1d)
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
    
    # plot PSF
    plt.imshow(PSF,vmin=0)
    plt.colorbar()
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    plt.title('PSF')
    plt.show()
    
    ## plot deconv + deconv profile (with rotation) ##
    fig, ax = plt.subplots(2,2,figsize=(15,13))
    im = ax[0,0].imshow(rotated_interp_img, extent=[-size/2, size/2, -size/2, size/2])
    divider = make_axes_locatable(ax[0,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax[0,0].set_xlabel('x [mm]')
    ax[0,0].set_ylabel('y [mm]')
    ax[0,0].set_title('Rotated interpolated image')

    legend = f'Avg P2V={np.around(avg_P2V_interp,3)}'
    ax[0,1].plot(np.arange(-size/2, size/2), interp_img_1d,label=legend)
    ax[0,1].set_xlabel('x [mm]')
    ax[0,1].set_ylabel('photon hits')
    ax[0,1].set_title('Rotated interpolated image profile')
    ax[0,1].grid()
    ax[0,1].legend(fontsize=10)
    # ax[0,1].set_ylim([0,None])
    
    # deconv
    im = ax[1,0].imshow(rotated_deconv, extent=[-size/2, size/2, -size/2, size/2])
    divider = make_axes_locatable(ax[1,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax[1,0].set_xlabel('x [mm]')
    ax[1,0].set_ylabel('y [mm]')
    ax[1,0].set_title('Rotated RL deconvolution')
    # deconv profile
    legend = f'Avg P2V={np.around(avg_P2V_deconv,3)}'
    ax[1,1].plot(np.arange(-size/2, size/2), rotated_deconv_1d,label=legend)
    ax[1,1].set_xlabel('x [mm]')
    ax[1,1].set_ylabel('photon hits')
    ax[1,1].set_title('Rotated RL deconvolution profile')
    ax[1,1].grid()
    ax[1,1].legend(fontsize=10)
    # ax[1,1].set_ylim([0,None])
    geo_params = geo_dir.split('/SquareFiberDatabase/')[-1]
    title = (f'{geo_params}\nEvent spacing = {np.around(dist,3)}[mm], ' +
             f'cutoff_iter={cutoff_iter}, rel_diff={np.around(rel_diff_checkout,4)}')
    fig.suptitle(title,fontsize=15)
    fig.tight_layout()
    plt.show()

# In[7]

    



















# In[6]










