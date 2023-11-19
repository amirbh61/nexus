#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 20:06:51 2022

@author: amir
"""

import numpy as np
from numpy import pi as pi
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.style.use(['classic'])
import time 
import os
import shutil
from scipy import ndimage
from scipy.interpolate import interp2d as interp2d
from scipy.signal import find_peaks, peak_widths
from scipy.signal           import fftconvolve
from scipy.signal           import convolve
from invisible_cities.reco.deconv_functions     import richardson_lucy
import glob
import pandas as pd
#GLOBAL VARIABLES

#create the mother directory
Mfolder = '/home/amir/Desktop/NEXT_work/Resolving_Power/Results/'
dirpath = os.path.join(f'{Mfolder}')
if not os.path.isdir(dirpath):
    os.mkdir(dirpath)



# In[0]
'''
PSF calculation and creation
'''
section_clock_start = time.process_time()

def plot_centering_range(distance_between_sources):
    '''
    This function centers the convolution 2d plot so its centered on the centroid of
    the original PSF. This centering is general for different distances between 
    sources and different angles (theta) between them.
    x axis points up, y axis apoints right
    '''  
    a = size/2
    d = distance_between_sources
    x_range = np.array([-(a-0.5*d*np.cos(theta)), a+0.5*d*np.cos(theta)])
    y_range = np.array([-(a-0.5*d*np.sin(theta)), a+0.5*d*np.sin(theta)])
    return x_range, y_range


def el_light():
    '''
    randomize photons in the EL gap.
    Calculates point of intersection with sensor wall.
    Returns x,y hit coordinates as a list.
    '''    
    sections = 1300 #marks electrons per 30 keV event
    # photons_per_section = 500 #marks photons emitted per electron in EL gap
    photons_per_section = 7700
    xhitList, yhitList = np.array([]), np.array([])
    d = np.linspace(0,el_gap,sections)
    for i in range(0,len(d)):
        cos_theta = np.random.uniform(0,1,size=photons_per_section)
        theta = np.arccos(cos_theta)
        phi = np.random.uniform(0,2*pi,size=photons_per_section)
        r = (anode_track_gap + el_gap - d[i])*np.tan(theta)
        xhit = r*np.cos(phi)
        yhit = r*np.sin(phi)
        xhitList = np.concatenate([xhitList,xhit])
        yhitList = np.concatenate([yhitList,yhit])
    return xhitList.tolist(), yhitList.tolist()
    

def smooth_PSF(PSF):
    '''
    Smoothes the PSF matrix based on the fact the ideal PSF should have radial
    symmetry.
    return the smoothed PSF matrix.
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


def psf_creator():
    '''
    Gives a picture of the SiPM (sensor wall) response to an event involving 
    the emission of light in the EL gap.
    Depends on the EL and tracking plane gaps.
    Returns the Point Spread Function (PSF) of smooth PSF.
    '''
    x,y = el_light() #MC light on wall
    
    PSF = np.zeros((100,100))
    PSF, x_hist, y_hist = np.histogram2d(x, y, range=[[-50, 50], [-50, 50]],bins=100)
    
    #Smooth the PSF
    # smoothed_PSF = smooth_PSF(PSF)
    # np.save(evt_PSF_output,smoothed_PSF)
    
    np.save(evt_PSF_output,PSF) #unsmoothed
    return 




size = 100 #figure/matrix
psf_plot_edge = 50
PSF_list = []
el_gap = 10
# anode_track_gap = 10
anode_track_gap = 2.5

# while anode_track_gap <= 3:
start = time.process_time()
   
#create all sub directories
folder = Mfolder + f'Resolving_Power_EL_gap{float(el_gap)}mm_Tracking_Plane_Gap{float(anode_track_gap)}mm/'
dirpath = os.path.join(f'{folder}')# def convolution_2dplot_centering():
if not os.path.isdir(dirpath):
    os.mkdir(dirpath)
    
Save_deconv  = f'{folder}deconv/'  
dirpath = os.path.join(f'{Save_deconv}')
if os.path.exists(dirpath) and os.path.isdir(dirpath):
    shutil.rmtree(dirpath)
if not os.path.isdir(dirpath):
    os.mkdir(dirpath)
    
Save_Conv  = f'{folder}Convolution/'  
dirpath = os.path.join(f'{Save_Conv}')
if not os.path.isdir(dirpath):
    os.mkdir(dirpath)
    
Save_SR  = f'{folder}Sensor_Response/'
dirpath = os.path.join(f'{Save_SR}')
if not os.path.isdir(dirpath):
       os.mkdir(dirpath) 

Save_PtoV  = f'{folder}PtoV/'  
dirpath = os.path.join(f'{Save_PtoV}')
if not os.path.isdir(dirpath):
    os.mkdir(dirpath)
    
Save_PSF  = f'{folder}PSF/'  
dirpath = os.path.join(f'{Save_PSF}')
if not os.path.isdir(dirpath):
    os.mkdir(dirpath) 
psf_file_name = 'PSF_matrix'
evt_PSF_output = Save_PSF + psf_file_name + '.npy'
Save_PSF_list = Mfolder + 'PSF_list_for_plots' + '.npy'
PSF_list = np.append(PSF_list,evt_PSF_output) #for PSF summary plot

psf_creator()
print(f'Time taken for PSF of EL gap {el_gap}mm and anode track gap {anode_track_gap}mm: {time.process_time() - start} seconds.')
# anode_track_gap += 2.5
    
np.save(Save_PSF_list,PSF_list)
print(f'Total time taken to produce all PSFs: {time.process_time() - section_clock_start} seconds.')

# In[1]
'''
PSF plots section
'''
section_clock_start = time.process_time()
# anode_track_gap = np.arange(0,22,2)
anode_track_gap = 2.5
el_gap = 10
psf_plot_edge = 50
big_run_mode = False
size = 100 #figure/matrix
# i = 10

#plot all PSFs on different graphs
i = anode_track_gap
# for i in anode_track_gap:
    start = time.process_time()
    #assign sub directory variables
    folder = Mfolder + f'Resolving_Power_EL_gap{el_gap}mm_Tracking_Plane_Gap{i}mm/'
    Save_PSF  = f'{folder}PSF/'  
    psf_file_name = 'PSF_matrix'
    evt_PSF_output = Save_PSF + psf_file_name + '.npy'
    PSF = np.load(evt_PSF_output)
    x_cm,y_cm = ndimage.measurements.center_of_mass(PSF)
    x_cm, y_cm = int(x_cm), int(y_cm)
    # cropped = PSF[x_cm-psf_plot_edge:x_cm+psf_plot_edge,y_cm-psf_plot_edge:y_cm+psf_plot_edge] #crop interesting area, for plot
    
    print(f'Plotting PSF for EL gap={el_gap}mm, anode gap={i}mm...')
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16.5,8))
    fig.suptitle(f'PSF - EL gap={el_gap}mm, Anode Track gap={i}mm', fontsize=15)
    im = ax0.imshow(PSF, extent=(-psf_plot_edge,psf_plot_edge,-psf_plot_edge,psf_plot_edge))
    # ax0.hist2d(x, y, range=[[-psf_plot_edge, psf_plot_edge],[-psf_plot_edge, psf_plot_edge]], bins=100) #unsmoothed
    ax0.set_xlabel('x [mm]');
    ax0.set_ylabel('y [mm]');
    ax0.set_title('smooth PSF image')
    # fig.colorbar(im, orientation='vertical', location='left')
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax1.plot()
    ax1.plot(np.arange(-psf_plot_edge,psf_plot_edge,1),
             smoothed_PSF[int(size/2),:]/np.sum(smoothed_PSF[int(size/2),:]),linewidth=2) # normalize
    
    # ax1.plot(np.arange(-psf_plot_edge,psf_plot_edge,1), cropped.sum(axis=0),linewidth=2)
    # ax1.plot(shift_to_bin_centers(x_hist), PSF.sum(axis=0)/PSF.sum()) #unsmoothed
    
    # # y = cropped.sum(axis=1) 
    y = PSF[int(size/2),:]
    peaks, _ = find_peaks(y)
    fwhm = np.max(peak_widths(y, peaks, rel_height=0.5)[0])
    fwhm_text = f"FWHM = {fwhm:.3f}"  # format to have 3 decimal places
    ax1.text(0.95, 0.95, fwhm_text, transform=ax1.transAxes, 
              verticalalignment='top', horizontalalignment='right', 
              color='red', fontsize=12, fontweight='bold',
              bbox=dict(facecolor='white', edgecolor='red',
                        boxstyle='round,pad=0.5'))
    ax1.set_xlabel('mm')
    ax1.set_ylabel('Charge')
    ax1.set_title('Charge along y axis')
    ax1.grid(linewidth=1)
    fig.tight_layout()
    fig.savefig(f'{Save_PSF}PSF_image.svg', format='svg', dpi=300, bbox_inches='tight')
    if big_run_mode: plt.close(fig)
    
    #compare smoothed vs unsmoothed
    # plt.figure()
    # plt.plot(np.arange(-psf_plot_edge,psf_plot_edge,1), cropped.sum(axis=1)/cropped.sum())
    # plot_area = PSF[psf_plot_edge:psf_plot_edge,psf_plot_edge:psf_plot_edge]
    # plt.plot(np.arange(-psf_plot_edge,psf_plot_edge,1), plot_area.sum(axis=1)/plot_area.sum())
    # plt.title(f'smoothed vs unsmoothed psf for EL gap={el_gap}mm, anode track gap={anode_track_gap}mm')
    # plt.xlabel('x [mm]')
    # plt.ylabel('Intensity')
    # plt.show()
    print(f'Time taken to plot PSF of EL gap {el_gap}mm and anode track gap {i}mm: {time.process_time() - start} seconds.')


#plot all PSFs and FWHM on one graph
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15,7.5))
fig.suptitle(f'PSF summary for EL gap={el_gap}mm', fontsize=15)
fwhm = []
anode_track_gap = np.arange(0,22,2)
Save_PSF_list = Mfolder + 'PSF_list_for_plots' + '.npy'
PSF_list = np.load(Save_PSF_list)

for i,psf_obj in enumerate(PSF_list):
    #put each psf on a graph and get its FWHM
    psf = np.load(psf_obj)
    y = psf.sum(axis=1) 
    ax0.plot(np.arange(-size/2,size/2),y, label=f"Anode gap={anode_track_gap[i]}mm")
    peaks, _ = find_peaks(y)
    fwhm = np.append(fwhm, np.max(peak_widths(y, peaks, rel_height=0.5)[0]))
    
ax0.set_xlabel('x [mm]')
ax0.set_ylabel('Charge sum along y axis')
ax0.set_title('Charge sum for various anode spacings')
ax0.legend(loc="upper right",fontsize=8)
ax0.set_xlim(-100,100)
ax0.grid(linewidth=1)
ax1.scatter(anode_track_gap,fwhm)
ax1.set_xlabel('Anode-tracking plane distance [mm]')
ax1.set_ylabel('FWHM [mm]')
ax1.set_title('FWHM values for the PSFs on the left figure')
ax1.grid(linewidth=1)
fig.tight_layout()
# fig.savefig(f'{Mfolder}PSF_summary.svg', format='svg', dpi=1200, bbox_inches='tight')
plt.show(fig)   

print(f'Total time taken to plot all PSFs: {time.process_time() - section_clock_start} seconds.')

# In[2]

'''
Convolution.
This section take the existing smooth PSFs, creates a copy of it for different
source distances and angle and saves image of the two sources on the screen.
'''
section_clock_start = time.process_time()


def matrix_centering(combined_PSF,x_cm, y_cm):
    '''
    takes the combined PSF and shifts the matrix in a way that the center of mass
    of the two sources is now the center of the matrix.
    returns the centered convolution matrix.
    '''
    psf_center = int(size/2)
    centered_combined_PSF = np.roll(combined_PSF,y_cm-psf_center,axis=0) # up/down shift - y axis
    centered_combined_PSF = np.roll(centered_combined_PSF, -(x_cm-psf_center),axis=1) # left/right shift - x axis
    return centered_combined_PSF


def combine_PSFs(anode_track_gap,distance_between_sources):
    '''
    This function take 2 PSF matrices, simulating the light from 2 sources hitting
    the tracking plane and merges them into one.
    '''
    
    PSF = np.load(evt_PSF_output)

    #shift PSF
    shifted_PSF = np.zeros((size,size))   
    x_shift = -int(distance_between_sources*np.cos(theta)) # minus so that x axis points up
    y_shift = int(distance_between_sources*np.sin(theta))
    shifted_PSF = np.roll(PSF,x_shift,axis=0) #axis=0 is up/down, axis=1 is right/left
    shifted_PSF = np.roll(shifted_PSF,y_shift,axis=1) 
    
    combined_PSF = PSF + shifted_PSF #merge two matriecs 
    
    #find plot center using convolution center of mass
    row_cm, column_cm = ndimage.measurements.center_of_mass(combined_PSF)
    #go to coordinate system where [0,0] is bottom left
    x_cm = int(column_cm)
    y_cm = int(size-row_cm)

    combined_PSF = matrix_centering(combined_PSF, x_cm, y_cm)
    np.save(Conv_output, combined_PSF) 
    return 


distance_between_sources = np.arange(0,160,10)
anode_track_gap = np.arange(0,22,2)
# j = distance_between_sources = 140
# i = anode_track_gap = 20
el_gap = 10
theta = pi/2 # theta=pi/2 -> horizontal, theta=0 -> veritcal sources
size = 500 #figure/matrix

for i in anode_track_gap:
    for j in distance_between_sources:
        iteration_start = time.process_time()
        
        #assign sub directory variables
        folder = Mfolder + f'Resolving_Power_EL_gap{el_gap}mm_Tracking_Plane_Gap{i}mm/'
        Save_PSF  = f'{folder}PSF/'  
        psf_file_name = 'PSF_matrix'
        evt_PSF_output = Save_PSF + psf_file_name + '.npy'
        Save_Conv  = f'{folder}Convolution/'  
        Conv_output = Save_Conv + f'source_spacing={j}mm' + '.npy'   
        
        combine_PSFs(i,j)
        print(f'convolutions time for EL gap={el_gap}mm, anode track gap={i}mm and source distance={j}mm: {time.process_time() - iteration_start} seconds.')

print(f'Total time taken to produce all convolutions: {time.process_time() - section_clock_start} seconds.')

# In[3]
'''
Convolution plots section
'''
section_clock_start = time.process_time()
big_run_mode = False
theta = pi/2
el_gap = 10
# anode_track_gap = np.arange(0,22,2)
# distance_between_sources = np.arange(10,160,10)
size = 500 #figure/matrix
left_edge = 100
i = anode_track_gap = 20
j = distance_between_sources = 140

# for i in anode_track_gap:
#     for j in distance_between_sources:
iteration_start = time.process_time()
        
#assign sub directory variables
folder = Mfolder + f'Resolving_Power_EL_gap{el_gap}mm_Tracking_Plane_Gap{i}mm/'
Save_Conv  = f'{folder}Convolution/'  
Conv_output = Save_Conv + f'source_spacing={j}mm' + '.npy'   
conv = np.load(Conv_output)
if theta == pi/2: 
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15,7.5), dpi=300)
    fig.suptitle(f'Convolution - source spacing={j}mm, EL gap={el_gap}mm, Anode Track gap={i}mm', fontsize=15)
    im = ax0.imshow(conv,extent=(0,size,0,size))
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax0.grid(linewidth=1)
    ax0.set_xlabel('x [mm]');
    ax0.set_ylabel('y [mm]');
    ax0.set_title('Image of the two sources')
    ax1.plot(np.arange(0,size), conv.sum(axis=0),linewidth=2)
    ax1.set_xlabel('x [mm]')
    ax1.set_ylabel('Charge sum along y axis')
    ax1.set_title('Charge sum')
    ax1.grid(linewidth=1)
    fig.tight_layout()
    fig.savefig(f'{Save_Conv}Source_spacing={j}mm.svg', format='svg', dpi=300, bbox_inches='tight')
    if big_run_mode: plt.close(fig)

else: # because it is a problem to sum hitpoints along an axis that is neither x nor y
    plt.figure()
    plt.imshow(conv,extent=(0,size,0,size))
    plt.xlabel('x [mm]')
    plt.ylabel('Charge sum along y axis')
    plt.title('Charge sum')
    plt.grid(linewidth=1)
    plt.tight_layout()
    plt.savefig(f'{Save_Conv}Source_spacing={j}mm.svg', format='svg', dpi=300, bbox_inches='tight')
    if(big_run_mode): plt.close()
   
print(f'convolution plot time for EL gap={el_gap}mm, anode track gap={i}mm and source distance={j}mm: {time.process_time() - iteration_start} seconds.')

print(f'Total time taken to plot all convolutions: {time.process_time() - section_clock_start} seconds.')

# In[4]

'''
Calculate and save sensor response.
Interpolate signal in dead space between SiPMs
'''
section_clock_start = time.process_time()

#parameters
el_gap = 10
pitch = 15
size = 500 #figure/matrix
# anode_track_gap = np.arange(0,22,2)
# distance_between_sources = np.arange(10,160,10)

#losses
TPB = 0.6 #TPB transmitivity
aimed_forward = 0.5 #we created photons going forward only, here we "pay" for it
SiPM_PDE = 0.4 
overall_losses = TPB*aimed_forward*SiPM_PDE

i = anode_track_gap = 20
j = distance_between_sources = 140



# for i in anode_track_gap:
# for j in distance_between_sources:
iteration_start = time.process_time()
folder = Mfolder + f'Resolving_Power_EL_gap{el_gap}mm_Tracking_Plane_Gap{i}mm/'
Save_SR  = f'{folder}Sensor_Response/'
conv = np.load(f'/home/amir/Desktop/NEXT_work/Resolving_Power/Results/Resolving_Power_EL_gap{el_gap}mm_Tracking_Plane_Gap{i}mm/Convolution/source_spacing={j}mm.npy','r')
SR_output = Save_SR + f'source_spacing={j}mm' + '.npy' #saves S matrix - no interpolation
SR_interpolation_output = Save_SR + f'RL_Interpolation_source_spacing={j}mm' + '.npy' #saves S matrix - with interpolation
S = np.zeros((size,size))
# S[::pitch,::pitch] = overall_losses*conv[::pitch,::pitch] #no poisson
S[::pitch,::pitch] = np.random.poisson(overall_losses*conv[::pitch,::pitch]) # fill only "pitch" spaced cells of matrix S, with losses included, poisson

#interpolation parameters
x_range = np.arange(-size/2,size/2)
y_range = np.arange(-size/2,size/2)
#interpolate from known data
conv_interp = interp2d(x_range[::pitch], y_range[::pitch],S[::pitch,::pitch], kind='cubic')
z = conv_interp(x_range, y_range) #create function z over the interpolated area
np.save(SR_interpolation_output,z) # Save interpolation matrix for RL deconv
np.save(SR_output,S) # Save no interpolation matrix for future use
print(f'Sensor ressponse time for EL gap={el_gap}mm, anode track gap={i}mm and source distance={j}mm: {time.process_time() - iteration_start} seconds.')

print(f'Total time taken to produce all convolutions: {time.process_time() - section_clock_start} seconds.')

# In[5]
'''
Plot Sensor response section
'''
section_clock_start = time.process_time()

el_gap = 10
pitch = 15
theta = pi/2 # theta=pi/2 -> horizontal, theta = 0 -> veritcal sources
size = 500 #figure/matrix
# anode_track_gap = np.arange(0,22,2)
# distance_between_sources = np.arange(10,160,10)

i = anode_track_gap = 20
j = distance_between_sources = 140


big_run_mode = False

# for i in anode_track_gap:
    # for j in distance_between_sources:
iteration_start = time.process_time()
folder = Mfolder + f'Resolving_Power_EL_gap{el_gap}mm_Tracking_Plane_Gap{i}mm/'
Save_SR  = f'{folder}Sensor_Response/'
SR_output = Save_SR + f'source_spacing={j}mm' + '.npy' #saves S matrix - no interpolation
SR_interpolation_output = Save_SR + f'RL_Interpolation_source_spacing={j}mm' + '.npy' #saves S matrix - with interpolation

S = np.load(SR_output)
z = np.load(SR_interpolation_output)
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(24.5,7), dpi=300)
fig.suptitle(f'SR - EL gap={el_gap}mm, anode track gap={i}mm, source spacing={j}mm, SiPM pitch={pitch}mm', fontsize=15)
x_cm, y_cm = ndimage.measurements.center_of_mass(S)
ax0.imshow(np.ma.masked_array(S,mask=S >= 1),'cividis',extent=(0,size,0,size)) #mask SiPM response
# ax0.imshow(S,extent=(x_cm-size/2,x_cm+size/2,y_cm-size/2,y_cm+size/2))
ax0.set_xlabel('x [mm]')
ax0.set_ylabel('y [mm]')
ax0.set_title('Response of the SiPMs to the two sources')     
im = ax1.imshow(z,extent=(0,size,0,size))
fig.colorbar(im, ax=ax1, orientation='vertical')
ax1.set_xlabel('x [mm]')
ax1.set_ylabel('y [mm]')
ax1.set_title('Bicubic interpolation in between the SiPMs')

ax2.plot(np.arange(0,size), z.sum(axis=0),linewidth=2)
ax2.set_xlabel('x [mm]')
ax2.set_ylabel('Charge sum along y axis')
ax2.set_title('Charge sum')
ax2.grid(linewidth=1)

fig.tight_layout()
print(f'Sensor ressponse plot time for EL gap={el_gap}mm, anode track gap={i}mm and source distance={j}mm: {time.process_time() - iteration_start} seconds.')

fig.savefig(f'{Save_SR}Sensor_response_source_distance={j}mm.svg', format='svg', dpi=300, bbox_inches='tight')
if big_run_mode: plt.close(fig)

# In[6]

'''
RL deconvolution calculation and save
'''
section_clock_start = time.process_time()

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


el_gap = 10
i = anode_track_gap = 20
j = distance_between_sources = 140
# anode_track_gap = np.arange(0,22,2)
# distance_between_sources = np.arange(10,150,10)
size = 500
theta = pi/2
iterations = 5000 
# iter_thr = [0.1,0.01,0.001,0.0005,0.0001,0.00005,0.00001]
iter_thr = [0.1,0.01,0.001,0.0001,0.00001]
# threshold = 0.00001
RL_summary = pd.DataFrame(columns=['threshold','iterations','Path'])

# for i in anode_track_gap: #strongest for
folder = Mfolder + f'Resolving_Power_EL_gap{el_gap}mm_Tracking_Plane_Gap{i}mm/'
Save_PSF  = f'{folder}PSF/'  
psf_file_name = 'PSF_matrix'
evt_PSF_output = Save_PSF + psf_file_name + '.npy'
Save_SR  = f'{folder}Sensor_Response/'
Save_deconv  = f'{folder}deconv/' 
Save_deconv_summary = Save_deconv + 'Deconvolution_summary_info.h5'
# if os.path.exists(Save_deconv_summary):
    # os.path.append(Save_deconv_summary)


#save deconvolutions info to hdf
RL_summary = pd.DataFrame(columns=['source spacing [mm]','threshold','iterations','path'])
thresholds = []
actual_iterations = []
paths = []
anode_spacing = []


# for j in distance_between_sources:#second strongest for  
iteration_start = time.process_time()
SR_interpolation_output = Save_SR + f'RL_Interpolation_source_spacing={j}mm' + '.npy'

image = np.load(SR_interpolation_output)
PSF = np.load(evt_PSF_output)

for threshold in iter_thr:#weakest for
    rel_diff, cutoff_iter, de_conv = richardson_lucy(PSF,image,iterations,threshold)
    rel_diff = remove_digits_beyond_first_significant(rel_diff)
    deconv_output = Save_deconv + f'Sources_spacing={j}mm_thr={rel_diff}_iterations={cutoff_iter}.npy'
    if os.path.exists(deconv_output):
        os.remove(deconv_output)     
    np.save(deconv_output, de_conv)
    
    #save deconvolutions info to lists
    anode_spacing.append(j)
    thresholds.append(rel_diff)
    actual_iterations.append(cutoff_iter)
    paths.append(deconv_output)
    print(f'RL time for EL gap={el_gap}mm, anode track gap={i}mm, source distance={j}mm, iterations={cutoff_iter}, thr={rel_diff}: {time.process_time() - iteration_start} seconds.')

#fill deconvolution summary info for that EL gap and tracking plane-anode distance
RL_summary['source spacing [mm]'] = anode_spacing
RL_summary['threshold'] = thresholds
RL_summary['iterations'] = actual_iterations
RL_summary['path'] = paths
RL_summary.to_hdf(Save_deconv_summary, key='deconv_summary')
                
print(f'Total time taken to produce all RL deconvolutions: {time.process_time() - section_clock_start} seconds.')
 
# In[7]
'''
Plot RL deconvolutions
'''
section_clock_start = time.process_time()

el_gap = 10
i = anode_track_gap = 20
j = distance_between_sources = 50
# anode_track_gap = np.arange(0,22,2)
# distance_between_sources = np.arange(10,150,10)
size = 500
edge = 125
theta = pi/2
big_run_mode = False

#plot all RL deconvolutions
# for i in anode_track_gap:
folder = Mfolder + f'Resolving_Power_EL_gap{el_gap}mm_Tracking_Plane_Gap{i}mm/'
Save_deconv  = f'{folder}deconv/' 
Save_deconv_summary = Save_deconv + 'Deconvolution_summary_info.h5'
deconv_index = pd.read_hdf(Save_deconv_summary, key='deconv_summary')

#     for j in distance_between_sources:
# iteration_start = time.process_time()

#find relevant deconvolutions to loop iteration (anode gap) and arange by threshold in descending order
relevant_deconv_files = deconv_index.where(deconv_index['source spacing [mm]']==j)
relevant_deconv_files = relevant_deconv_files.sort_values('threshold',ascending=False)


for idx in range(len(relevant_deconv_files)):
    #load file and parameters from the temp relevant list
    deconv = np.load(relevant_deconv_files.at[idx,'path'])
    threshold = relevant_deconv_files.at[idx,'threshold']
    iterations = relevant_deconv_files.at[idx,'iterations']
    
    x_cm, y_cm = ndimage.measurements.center_of_mass(deconv)
    x_cm, y_cm = int(x_cm), int(y_cm)
    deconv = deconv[x_cm-edge:x_cm+edge,y_cm-edge:y_cm+edge]
    
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20,9))
    fig.suptitle(f'RL deconvolution for: EL gap={el_gap}mm, anode track gap={i}mm, source distance={j}mm, threshold={threshold}, iterations={iterations}', fontsize=20)
    ax0.imshow(deconv,extent=(x_cm-edge,x_cm+edge,y_cm-edge,y_cm+edge))
    ax0.set_xlabel('x [mm]')
    ax0.set_ylabel('y [mm]')
    ax0.set_title('Deconvolution for 2 sources')
    ax1.plot(np.arange(x_cm-edge,x_cm+edge),deconv.sum(axis=0)/deconv.sum(),linewidth=2)
    ax1.grid(linewidth=1)
    ax1.set_ylabel('Charge fraction along y axis')
    ax1.set_title('Charge fraction')
    fig.tight_layout()
    fig.savefig(f'{Save_deconv}Sources_spacing={j}mm_thr={threshold}_iterations={iterations}.svg', format='svg', dpi=300, bbox_inches='tight' )
    if big_run_mode: plt.close(fig)



'''
plot RL deconvolutions and corresponding peaks in **ONE BIG GRAPH** to show
RL for different thresholds
'''

i = anode_track_gap = 20
j = distance_between_sources = 10
edge = 75

folder = Mfolder + f'Resolving_Power_EL_gap{el_gap}mm_Tracking_Plane_Gap{i}mm/'
Save_deconv  = f'{folder}deconv/' 
Save_deconv_summary = Save_deconv + 'Deconvolution_summary_info.h5'
deconv_index = pd.read_hdf(Save_deconv_summary, key='deconv_summary')
#find relevant deconvolutions to loop iteration (anode gap) and arange by threshold in descending order
relevant_deconv_files = deconv_index.where(deconv_index['source spacing [mm]']==j)
relevant_deconv_files = relevant_deconv_files.sort_values('threshold',ascending=False)

fig, ax = plt.subplots(2, 5, figsize =(36,16))
fig.suptitle(f'RL deconvolution for: EL gap={el_gap}mm, anode track gap={i}mm, ' \
              f'source spacing={j}mm',fontsize=22)

for plot_col in range(len(relevant_deconv_files)):
    #load deconv matrix for each threshold value
    deconv = np.load(relevant_deconv_files.at[plot_col,'path'])
    threshold = relevant_deconv_files.at[plot_col,'threshold']
    iterations = relevant_deconv_files.at[plot_col,'iterations']
    
    x_cm, y_cm = ndimage.measurements.center_of_mass(deconv)
    x_cm, y_cm = int(x_cm), int(y_cm)
    deconv = deconv[x_cm-edge:x_cm+edge, y_cm-edge:y_cm+edge]
    ax[0,plot_col].imshow(deconv,extent=(x_cm-edge,x_cm+edge,y_cm-edge,y_cm+edge))
    ax[0,plot_col].set_xlabel('x [mm]')
    ax[0,plot_col].set_ylabel('y [mm]')
    ax[0,plot_col].set_title(f'threshold={threshold}, iterations={iterations}')
    ax[1,plot_col].plot(np.arange(x_cm-edge,x_cm+edge),deconv.sum(axis=0)/deconv.sum(),linewidth=2)
    ax[1,plot_col].set_ylabel('Charge fraction along y axis')
    ax[1,plot_col].set_xlabel('x [mm]')
    ax[1,plot_col].set_xlim(left=x_cm-edge, right=x_cm+edge)
    ax[1,plot_col].grid(linewidth=1)
    fig.tight_layout()
if big_run_mode: plt.close(fig) 

fig.savefig(f'{Save_deconv}RL_summary_plot_Sources_spacing={j}mm.svg', format='svg', dpi=1200, bbox_inches='tight' )

print(f'Total time taken to plot all RL deconvolutions: {time.process_time() - section_clock_start} seconds.')

# In[8]
'''
Peak to Valley (P2V) analysis
'''
section_clock_start = time.process_time()
import itertools

def peaks(array):
    fail = 0
    peak_idx, properties = find_peaks(array, height = 20)
    if len(peak_idx) == 1:
        fail = 1
    return fail, peak_idx, properties['peak_heights']

def find_min_between_peaks(array, left, right):
    return min(array[left:right])


el_gap = 10
i = anode_track_gap = 20
j = distance_between_sources = 10
edge = 75
# anode_track_gap = np.arange(0,22,2)
# distance_between_sources = np.arange(10,150,10)
big_run_mode = False

# for i in anode_track_gap:
#     for j in distance_between_sources:
# iteration_start = time.process_time()

folder = Mfolder + f'Resolving_Power_EL_gap{el_gap}mm_Tracking_Plane_Gap{i}mm/'
Save_deconv  = f'{folder}deconv/' 
Save_deconv_summary = Save_deconv + 'Deconvolution_summary_info.h5'
deconv_index = pd.read_hdf(Save_deconv_summary, key='deconv_summary')

#find relevant deconvolutions to loop iteration (anode gap) and arange by threshold in descending order
relevant_deconv_files = deconv_index.where(deconv_index['source spacing [mm]']==j)
relevant_deconv_files = relevant_deconv_files.sort_values('threshold',ascending=False)

# colors = itertools.cycle(['r', 'g', 'b', 'm', 'k'])
colors = ['r', 'g', 'b', 'm', 'k']
fig, (ax0,ax1) = plt.subplots(1,2, figsize=(15,7.5), dpi=600)
fig.suptitle(f'Peak to Valley - source spacing={j}mm, EL gap={el_gap}mm, Anode Track gap={i}mm', fontsize=15)


for idx in range(len(relevant_deconv_files)):
    #load file and parameters from the temp relevant list
    deconv = np.load(relevant_deconv_files.at[idx,'path'])
    threshold = relevant_deconv_files.at[idx,'threshold']
    iterations = relevant_deconv_files.at[idx,'iterations']

    x_cm, y_cm = ndimage.measurements.center_of_mass(deconv)
    x_cm, y_cm = int(x_cm), int(y_cm)
    deconv_1d = deconv[y_cm,:]
    
    fail, peak_idx, heights = peaks(deconv_1d)
    if fail:
        print(f'Could not find any peaks for EL gap={el_gap}mm, anode tracking gap={i}, ' \
              f'source spacing={j}mm')
        plt.close(fig)
        continue
    valley = find_min_between_peaks(deconv_1d, peak_idx[0], peak_idx[1])
    avg_peak = np.average(heights)
    avg_P2V = avg_peak/valley

    ax0.plot(deconv.sum(axis=0)/deconv.sum(), color=colors[idx], linewidth=1.5, label=f'threshold={threshold},iterations={iterations}')
    ax0.set_xlabel('x [mm]')
    ax0.set_ylabel('Charge fraction along y axis')
    ax0.set_title('Charge fracion for multiple tresholds')
    ax0.legend(loc="upper right",fontsize=8)
    ax0.set_xlim(left=x_cm-edge, right=x_cm+edge)
    ax0.grid(linewidth=1)
    ax1.scatter(np.log10(threshold),avg_P2V, marker='o', color=colors[idx], linewidth=4, label=f'threshold={threshold},iterations={iterations}')
    ax1.set_xlabel('log(Threshold)')
    ax1.set_ylabel('Average peak to valley ratio')
    ax1.set_title('Peak to valley vs threshold')
    ax1.legend(loc="upper right",fontsize=8)
    ax1.grid(linewidth=1)
    fig.tight_layout()
    # fig.savefig(f'{Save_Conv}Source_spacing={j}mm.svg', format='svg', dpi=300, bbox_inches='tight')
    if big_run_mode: plt.close(fig)


'''
To do :
    
check ax0 y axis values on plot 
if more than one peak in between -> check its height and
maybe set it as the new threshold for the next iteration


in deconvolution -> after calculating convolutions , in next sector, 
run on all files in sub directory deconv and create a pandas df + hdf file
of all the files there
'''






























