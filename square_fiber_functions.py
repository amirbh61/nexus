#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 20:14:58 2024

@author: amir
"""

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.style.use('classic')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
import glob
import re
from scipy import ndimage
from scipy.signal           import fftconvolve
from scipy.signal           import convolve
# from invisible_cities.reco.deconv_functions     import richardson_lucy
from scipy.signal import find_peaks, peak_widths

n_sipms = 25 # DO NOT CHANGE THIS VALUE
n_sipms_per_side = (n_sipms-1)/2
size = 250
bins = size

'''
FUNCTIONS for square_fiber.py
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
    if (-half_grid_length <= nearest_x <= half_grid_length and
        -half_grid_length <= nearest_y <= half_grid_length):
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
            plot_sensor_response(hitmap, bins, size,
                                 title='Single Geant4 Kr event')
        
        # Assign each hit to a SiPM before shifting the hitmap
        new_hitmap = []
        for hit in hitmap:
            sipm = assign_hit_to_SiPM(hit=hit, pitch=pitch, n=n_sipms)
            if sipm:  # if the hit belongs to a SiPM
                new_hitmap.append(sipm)
       
        new_hitmap = np.array(new_hitmap)
        
        
        if plot_sipm_assigned_event:
            plot_sensor_response(new_hitmap, bins, size,
                                 title='SR of single event')
        
        
        
        # Now, shift each event to center
        shifted_hitmap = new_hitmap - [x_event, y_event]
    
        
        if plot_shifted_event:
            plot_sensor_response(shifted_hitmap, bins, size,
                                 title='SR of shifted event')
        
        
        PSF_list.append(shifted_hitmap)
        
        if plot_accomulated_events:
            PSF = np.vstack(PSF_list)
            plot_sensor_response(PSF, bins, size,
                                 title='Accumulated events SR')
        
        

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


def plot_sensor_response(event, bins, size, title='', noise=False):
    hist, x_hist, y_hist = np.histogram2d(event[:,0], event[:,1],
                                                         range=[[-size/2, size/2], [-size/2, size/2]],
                                                         bins=[bins,bins])  
    if noise:
        hist = np.random.poisson(hist)

    # # Transpose the array to correctly align the axes
    hist = hist.T
    
    fig, ax = plt.subplots(1, figsize=(7,7), dpi=600)
    fig.set_facecolor('white')
    im = ax.imshow(hist,extent=[x_hist[0], x_hist[-1], y_hist[0], y_hist[-1]], vmin=0,
                origin='lower',interpolation=None,aspect='equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label='Photon hits')
    if title!="":
        ax.set_title(title,fontsize=13,fontweight='bold')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    plt.show()


def plot_PSF(PSF,size=100):
    total_TPB_photon_hits = int(np.sum(PSF))
    
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15,7), dpi=600)
    fig.patch.set_facecolor('white')
    title = (f'{total_TPB_photon_hits}/100M TPB hits PSF, current geometry:' +
             f'\n{os.path.basename(os.getcwd())}')
    # fig.suptitle(title, fontsize=15)
    im = ax0.imshow(PSF, extent=[-size/2, size/2, -size/2, size/2])
    ax0.set_xlabel('x [mm]', fontsize=18);
    ax0.set_ylabel('y [mm]', fontsize=18);
    ax0.set_title('PSF image', fontsize=15, fontweight='bold')

    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    
    # Format colorbar tick labels in scientific notation
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))  # You can adjust limits as needed
    cbar.ax.yaxis.set_major_formatter(formatter)
    
    y = PSF.mean(axis=0)
    peaks, _ = find_peaks(y)
    fwhm = np.max(peak_widths(y, peaks, rel_height=0.5)[0])
    ax1.plot(np.arange(-size/2,size/2,1), y, linewidth=2) #normalize
    ax1.set_xlabel('x [mm]', fontsize=18)
    ax1.set_ylabel('Photon hits', fontsize=18)
    ax1.set_title('PSF profile', fontsize=15, fontweight='bold')
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                    
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