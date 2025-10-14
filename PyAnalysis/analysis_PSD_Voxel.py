"""
Copyright (C) 2020-2024 Nico Marioni <nmarioni@utexas.edu>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# This script determines the pore size distribution based on the van der Waals volume of the system
# This code was specifically desgined to find the distribution of water-rich pores within a hydrated polymer system
# The output includes the Cumulative Pore Size Distribution, Pore Size Distribution, and Free Volume Fraction
# This code was written based on the methods used for PoreBlazer: https://github.com/SarkisovGitHub/PoreBlazer
#
# This code voxelizes the system into voxels of side length L, defined below in the iDist function. The default value of L = 0.35 Angstroms, which appears to work well for systems with very low probe-accessible free volumes.
# When implementing this code, it is recommended to test different values of L to ensure convergence of the FFV as L decreases. Note, computation time and memory usage will grow significantly as L decreases.
# If you run into memory or extreme run times, there are debugging lines throughout the code, and several values you can change to increase or decrease memory usage.
# There are two instances where .xyz files can be created to visualize 1) probe-accessible spheres of maximum radius without overlapping the van der Waals volume of the systems, and 2) voxel-centers that lie within the probe-accessible volume

import MDAnalysis as mda
import MDAnalysis.analysis.distances as dist
import MDAnalysis.lib.distances as distances
import numpy as np
import secrets

import multiprocessing as mp
import functools
import h5py
import os
from sys import argv

script, trj_file, top_file, system_name, probe_radius, t_min, t_max, N_frames, nt = argv
probe_radius = float(probe_radius); t_min = float(t_min); t_max = float(t_max); N_frames = int(N_frames); nt = int(nt)
# trj_file = .trr/.xtc file, top_file = .tpr file
# system_name = names of atoms that make up the polymer matrix in a form acceptable by MDAnalysis.select_atoms(), e.g., "moltype MOL"
# probe_radius = radius in Angstroms of the probe for obtaining the probe-accessible PSD and FFV, e.g., water molecules are 1.4 Angstroms
# t_min = start time for analysis in ps (-1 assumes the start time of the first frame)
# t_max = end time for analysis in ps (-1 assumes the end time of the last frame)
# N_frames = number of frames to analyze (-1 assumes N_frames = nt) -> for efficiency, N_frames should be a multiple of the number of threads
#            N_frames = 1 defaults to the frame at t_max
# nt = number of threads

# Example of Use: python3 analysis_PSD_Voxel.py md.xtc md.tpr "moltype MOL" 1.4 90000 100000 80 80
#                 Polymer chains are molecules with name MOL -> this defines the "system"
#                 probe_radius = 1.4 is approximately the size of a water molecule
#                 The run time of this code is highly dependent on the number of frames -> N_frames shouldn't be too many multiples of nt
#                     Note: each frame takes about the same amount of time to process, so it is efficient for the number of frames examined to be a multiple of nt

# Van der waals radii from Bondi (1964)
# https://www.knowledgedoor.com/2/elements_handbook/bondi_van_der_waals_radius.html
Size_arr = np.array([('C',1.7), ('O',1.52), ('H', 1.2), ('N', 1.55), ('S', 1.8), ('F', 1.47), ('P', 1.80)])



def iDist(frame):
    if frame%nt == 0:
        print("Frame " + str(frame))

    with h5py.File('/tmp/PSD.hdf5','r') as f:
        dset1 = f['system']; sys = dset1[frame]                                                                                                 # Position of all system atoms
        dset2 = f['sys_radii']; sys_radii = dset2[:]                                                                                            # Radius of all system atoms, where distance between sphere (see below) and polymer atom minus the radius is the distance to the van der waals surface of the atom
        dset3 = f['cells']; cell = dset3[frame]                                                                                                 # Size of the cell
        dset4 = f['frames']; frame_ids = dset4[:]; frame_ids = np.arange(0,len(frame_ids),1)





    # This part of the calculation determines the maximum size of voxel-centered free volume spheres without overlapping system atoms, where the total volume of all spheres larger than probe_radius defines the probe-accessible free volume of the system
    # This code will generate points at the center of voxels with side length L and grow these points into free volume spheres
    # Changing L, N, and d_inc can reduce run time and memory usage
    L = 0.35                                                                                                                                    # Side length of voxel in angstroms
    vox_x = np.round(np.linspace(0, cell[0], num = int(cell[0]/L)), decimals=5); vox_x = np.round((vox_x[:-1] + vox_x[1:])/2, decimals=5)       # Voxel-centers in the x direction
    vox_y = np.round(np.linspace(0, cell[1], num = int(cell[1]/L)), decimals=5); vox_y = np.round((vox_y[:-1] + vox_y[1:])/2, decimals=5)       # Voxel-centers in the y direction
    vox_z = np.round(np.linspace(0, cell[2], num = int(cell[2]/L)), decimals=5); vox_z = np.round((vox_z[:-1] + vox_z[1:])/2, decimals=5)       # Voxel-centers in the z direction

    sphere_arr = []; radii_arr = []                                                                                                             # sphere_arr and radii_arr track free volume sphere positions and size, respectively, where we are interested in spheres of radius r > probe_radius
    PSD_probes = []                                                                                                                             # PSD_probes tracks all voxel-centers not within the system van der Waals volume for later calculations, r >= 0
    
    N = 20e9                                                                                                                                    # Maximum number of distance calculations-per-loop. A lower number uses less memory.
    N_cube = int((N/len(sys))**(1/3))**3                                                                                                        # To improve efficiency, voxels are looped over in cubes of N_cube voxel-centers
    vox_inc = int(N_cube**(1/3)); vox_track = np.array((0,0,0), dtype=int); vox_track[0] = -vox_inc                                             # vox_inc = side length of voxel cube, vox_track tracks the location of the cubes in x, y, and z compared to the position in vox_x, vox_y, and vox_z

    ## Prints the number of voxels-per-cube and number of voxel cubes
    #if frame == frame_ids[-1]:
    #    print("Number of voxels-per-cube: ", N_cube)
    #    print("Number of voxel cubes: ", np.ceil(len(vox_x)/vox_inc).astype(int)*np.ceil(len(vox_y)/vox_inc).astype(int)*np.ceil(len(vox_z)/vox_inc).astype(int))
    for x_i in np.arange(vox_inc,len(vox_x)+vox_inc,vox_inc):
        vox_track[0] += vox_inc
        if x_i > len(vox_x):
            x_i = len(vox_x)

        vox_track[1] = -vox_inc
        for y_i in np.arange(vox_inc,len(vox_y)+vox_inc,vox_inc):
            vox_track[1] += vox_inc
            if y_i > len(vox_y):
                y_i = len(vox_y)
    
            vox_track[2] = -vox_inc
            for z_i in np.arange(vox_inc,len(vox_z)+vox_inc,vox_inc):
                vox_track[2] += vox_inc
                if z_i > len(vox_z):
                    z_i = len(vox_z)

                sphere_temp = np.zeros((N_cube, 3)); count = 0                                                                                  # sphere_temp contains the position of the voxel-centers within the cube of size N_cube
                for x in vox_x[vox_track[0]:x_i]:
                    for y in vox_y[vox_track[1]:y_i]:
                        for z in vox_z[vox_track[2]:z_i]:
                            sphere_temp[count] = np.round([x,y,z], decimals = 5); count += 1
                sphere_temp = np.round(sphere_temp[:count], decimals = 5)

                # Find the approximate center of the voxel cube to find the system atoms near the voxel cube (sys_mask), where system atoms define the van der Waals volume of the system. Reduces computational cost
                center = np.round([vox_x[vox_track[0] + int((x_i - vox_track[0])/2)], vox_y[vox_track[1] + int((y_i - vox_track[1])/2)], vox_z[vox_track[2] + int((z_i - vox_track[2])/2)]], decimals = 5)

                # To reduce the number of calculations and limit memory usage, the distance between voxel-centers and system atoms is done in steps of d_inc Angstroms
                d = 0.0; d_inc = 2                                                                                                              # Maximum distance to calculate between every voxel-center and every system atom
                while len(sphere_temp) > 0:
                    d += d_inc

                    sys_mask = distances.capped_distance(center, sys, d + np.sqrt(3)*vox_inc*L/2 + 0.5, box=cell)[0][:,1]                       # System atoms near the voxel cube

                    pair_arr, dist_arr = distances.capped_distance(sphere_temp, sys[sys_mask], d, box=cell)                                     # Distance between voxel-centers and system atoms

                    ## Useful print command for troubleshooting memory problems: prints the distance calulcated out to and the number of distance calculations
                    ## Decreasing d_inc or N will reduce the number of distances generated each cycle, reducing memory usage
                    #if frame == frame_ids[-1]:
                    #    if d == d_inc:
                    #        print("Voxel Block: ", (vox_track/vox_inc).astype(int))
                    #    print("Distance calculations: {:2.1f} {:.1e}".format(d, len(sphere_temp)*len(sys[sys_mask])))

                    if len(dist_arr) > 0:
                        dist_arr -= sys_radii[sys_mask][pair_arr[:,1]]                                                                          # Subtract radius of each system atom from the distance to get the distance from the voxel-center to the surface of the atom

                        # Fill sphere_arr and radii_arr for all voxel-centers that contain system atoms within d distance
                        index = 0; sph_save = pair_arr[0,0]
                        sphere_remove = []
                        for i,sph in enumerate(pair_arr[:,0]):
                            if sph > sph_save:
                                r_min = np.round(np.min(dist_arr[index:i]), decimals=5)                                                         # Minimum distance between voxel-center and system atom

                                if r_min > probe_radius:
                                    sphere_arr.append(sphere_temp[sph_save]); radii_arr.append(r_min)                                           # Sphere does not overlap the system and radius r > probe_radius

                                if r_min >= 0:                                                                                                  # Sphere does not overlap the system and radius r >= 0 -> use in later analysis
                                    PSD_probes.append(sphere_temp[sph_save])

                                sphere_remove.append(sph_save)                                                                                  # Analysis complete, remove from future distance calculations
                                index = i; sph_save = sph

                        if i > index:                                                                                                           # Check to make sure the last voxel-center is counted
                            r_min = np.round(np.min(dist_arr[index:]), decimals=5)                                                              # Minimum distance between voxel-center and system atom

                            if r_min > probe_radius:
                                sphere_arr.append(sphere_temp[sph_save]); radii_arr.append(r_min)                                               # Sphere does not overlap the system and radius r > probe_radius

                            if r_min >= 0:                                                                                                      # Sphere does not overlap the system and radius r >= 0 -> use in later analysis
                                PSD_probes.append(sphere_temp[sph_save])

                            sphere_remove.append(sph_save)                                                                                      # Analysis complete, remove from future distance calculations
                    sphere_temp = np.delete(sphere_temp, np.array(sphere_remove), axis=0)                                                       # Removed evaluated voxel-centers
    sphere_arr = np.array(sphere_arr); radii_arr = np.array(radii_arr); max_radius = np.max(radii_arr); PSD_probes = np.array(PSD_probes)

    ## Useful print command for troubleshooting problems: prints the number of voxel-centers within the free volume, the radius of the largest sphere, and the diameter of the largest sphere (pore)
    ## Also prints the number of voxels within the system van der Waals free volume, voxels containing free volume spheres of radius r >= probe_radius, and voxels that need to be assessed whether they are in the free volume or not
    #if frame == frame_ids[-1]:
    #    print("Spheres Created: ", len(radii_arr), max_radius, 2 * max_radius)
    #    print("Voxels not within the system: ", len(PSD_probes))

    ### Code to write coordinates and radius of each free volume sphere to a .xyz file, which can be visualized in Ovito, etc
    #if frame == frame_ids[-1]:
    #    with open('Free_Volume_Spheres.xyz', 'w') as anaout:
    #        print(str(len(sphere_arr)), file=anaout)
    #        print('Properties=species:S:1:pos:R:3:Radius:R:1', file=anaout)
    #        for i, sph in enumerate(sphere_arr):
    #            print('X {:10.5f} {:10.5f} {:10.5f} {:10.5f}'.format(sph[0] - cell[0]/2, sph[1] - cell[1]/2, sph[2] - cell[2]/2, radii_arr[i]), file=anaout)
    #    print('Free Volume Sphere XYZ File Printed')





    # This part of the calculation determines the free volume fraction and cumulative probe-accessible pore size distribution, where the distribution is defined as the probability that a random point (voxel) within the free volume resides within a free volume sphere of diameter d with minimum size probe_radius
    # This code will take each voxel not within the system volume (PSD_probes) and determine 1) if it lies within the free volume (FFV), and 2) the largest free volume sphere it lies within (PSD)
    # Changing L, N, and d_inc can reduce run time and memory usage

    FFV_track = 0; FFV_total = len(vox_x)*len(vox_y)*len(vox_z)                                                                                 # Track number of voxels within the free volume against the total number to get FFV
    d_max = 50; d_inc = 0.1                                                                                                                     # d_max is the largest free volume diameter to measure PSD out to; d_inc is the stepsize between free volume diameters
    d_arr = np.arange(0, d_max + d_inc, d_inc); PSD_arr = np.zeros_like(d_arr, dtype=int)                                                       # d_arr is the histogram of free volume sphere sizes; PSD_arr tracks the number of instances of voxels contained within free volume spheres of size at least d

    N = 10e9                                                                                                                                    # Maximum number of distance calculations-per-loop. Lower number reduces memory usage
    FFV_save = np.zeros((1,3)) - 1                                                                                                              # Save voxel-centers within the free volume incase you want it to be printed for visualization
    for d in np.round(np.arange(d_max, 0, -d_inc), decimals = 5):
        if d - d_inc > 2*max_radius:
            continue
        
        # For efficiency, we measure the distance between free volume spheres and the voxel-centers starting with the largest d_arr bin and moving down
        sphere_temp = sphere_arr[(radii_arr < d/2) & (radii_arr > (d - d_inc)/2)]; radii_temp = radii_arr[(radii_arr < d/2) & (radii_arr > (d - d_inc)/2)]
        if len(sphere_temp) == 0:
            continue
        
        # For efficiency, we limit the number of free volume spheres per loop to a total of N distance calculations
        count = 0
        while count < len(sphere_temp):
            count_old = count; count += min(int(N/len(PSD_probes)), len(sphere_temp)-count_old)

            sph_temp = sphere_temp[count_old:count]; rad_temp = radii_temp[count_old:count]

            ## Useful print command for troubleshooting memory problems: the number of distance calculations
            ## Decreasing N will reduce the number of distances generated each cycle, reducing memory usage
            #if frame == frame_ids[-1]:
            #    if count_old == 0:
            #        print("Radii: ", d/2)
            #    print("Distances calculated: {:.1e}".format(len(sph_temp)*len(PSD_probes)))
            
            pair_arr, dist_arr = distances.capped_distance(sph_temp, PSD_probes, d/2, box=cell)                                                 # Distance between free volume spheres and voxel-centers

            if len(dist_arr) > 0:
                dist_arr -= rad_temp[pair_arr[:,0]]                                                                                             # Subtract radius of each free volume sphere from the distance to get the distance from the voxel-center to the surface of the free volume sphere
                pair_arr = np.unique(pair_arr[:,1][dist_arr < 0])                                                                               # Only consider voxel-centers that lie within the free volume sphere (adjusted distance < 0), and only count each occurence once (unique)

                FFV_track += len(pair_arr); PSD_arr[np.where(d_arr < d)[0]] += len(pair_arr)                                                    # Voxel-centers w/n free volume sphere count towards the FFV and cumulatively towards the PSD
                if np.all(FFV_save[0] == -1):                                                                                                   # Save free volume voxel-centers for printing
                    FFV_save = PSD_probes[pair_arr]
                else:
                    FFV_save = np.append(FFV_save, PSD_probes[pair_arr], axis=0)
                PSD_probes = np.delete(PSD_probes, pair_arr, axis=0)                                                                            # No longer consider voxel-centers that are found within a free volume sphere in future loops
    
    ## Code to print the final FFV and PSD for the last frame analyzed
    #if frame == frame_ids[-1]:
    #    print("FFV: ", FFV_track/FFV_total, FFV_track, FFV_total)
    #    print("PSD Final:", PSD_arr[0])
    #    print_string=''
    #    for i in PSD_arr:
    #        if i == 0:
    #            continue
    #        print_string += str(np.round(i / PSD_arr[0], decimals=5)) + ' '
    #    print(print_string)

    ## Code to write coordinates of each voxel-center to a .xyz file, which can be visualized in Ovito
    #if frame == frame_ids[-1]:
    #    with open('Free_Volume_Voxels.xyz', 'w') as anaout:
    #        print(str(len(FFV_save)), file=anaout)
    #        print('Properties=species:S:1:pos:R:3:Radius:R:1', file=anaout)
    #        for i, sph in enumerate(FFV_save):
    #            print('X {:10.5f} {:10.5f} {:10.5f} {:10.5f}'.format(sph[0] - cell[0]/2, sph[1] - cell[1]/2, sph[2] - cell[2]/2, L/2), file=anaout)
    #    print('Free Volume Voxel XYZ File Printed')
    




    # Return the necessary information to complete the calculations: FFV_track / FFV_total gives the pore-accessible free volume, PSD_arr / PSD_arr[0] gives the pore-accessible PSD
    PSD_arr = np.insert(PSD_arr, 0, FFV_total); PSD_arr = np.insert(PSD_arr, 0, FFV_track)
    return PSD_arr





    ## OLD ALGORITHM - FFV/PSD have been significantly updated. Below versions may not work properly.
    ##
    ## This part of the calculation determines the maximum size of voxel-center spheres without overlapping system atoms, where the total volume of all spheres larger than probe_radius defines the probe-accessible free volume of the system
    ## This code will generate points at the center of voxels with side length L and grow these points into free volume spheres
    ## Changing L, N, and d_inc can reduce run time and memory usage
    #L = 0.35                                                                                                                                    # Side length of voxel in angstroms
    #vox_x = np.round(np.linspace(0, cell[0], num = int(cell[0]/L)), decimals=5); vox_x = np.round((vox_x[:-1] + vox_x[1:])/2, decimals=5)       # Voxel-centers in the x direction
    #vox_y = np.round(np.linspace(0, cell[1], num = int(cell[1]/L)), decimals=5); vox_y = np.round((vox_y[:-1] + vox_y[1:])/2, decimals=5)       # Voxel-centers in the y direction
    #vox_z = np.round(np.linspace(0, cell[2], num = int(cell[2]/L)), decimals=5); vox_z = np.round((vox_z[:-1] + vox_z[1:])/2, decimals=5)       # Voxel-centers in the z direction
    #
    #sphere_arr = []; radii_arr = []                                                                                                             # sphere_arr and radii_arr track free volume sphere positions and size, respectively, where we are interested in sphere of radius r > probe_radius
    #sphere_mask = np.zeros((len(vox_x),len(vox_y),len(vox_z)), dtype=np.int8)                                                                   # sphere_mask tracks whether a voxel-center is (-1) within the van der Waals volume of the system, (0) outside the van der Waals volume of the system, or (1) contains a free volume sphere
    #
    #N = 100000                                                                                                                                  # To improve efficiency, voxels are looped over in cubes of ~N voxel-centers
    #vox_inc = int(N**(1/3)); vox_track = np.array((0,0,0), dtype=int); vox_track[0] = -vox_inc                                                  # vox_inc = side length of voxel cube, vox_track tracks the location of the cubes in x, y, and z compared to the position in vox_x, vox_y, and vox_z
    #for x_i in np.arange(vox_inc,len(vox_x)+vox_inc,vox_inc):
    #    vox_track[0] += vox_inc
    #    if x_i > len(vox_x):
    #        x_i = len(vox_x)
    #
    #    vox_track[1] = -vox_inc
    #    for y_i in np.arange(vox_inc,len(vox_y)+vox_inc,vox_inc):
    #        vox_track[1] += vox_inc
    #        if y_i > len(vox_y):
    #            y_i = len(vox_y)
    #
    #        vox_track[2] = -vox_inc
    #        for z_i in np.arange(vox_inc,len(vox_z)+vox_inc,vox_inc):
    #            vox_track[2] += vox_inc
    #            if z_i > len(vox_z):
    #                z_i = len(vox_z)
    #
    #            sphere_temp = np.zeros((N, 3)); count = 0                                                                                       # sphere_temp contains the position of the voxel-centers within the cube of size N
    #            for x in vox_x[vox_track[0]:x_i]:
    #                for y in vox_y[vox_track[1]:y_i]:
    #                    for z in vox_z[vox_track[2]:z_i]:
    #                        sphere_temp[count] = np.round([x,y,z], decimals = 5); count += 1
    #            sphere_temp = np.round(sphere_temp[:count], decimals = 5)
    #            radii_check = np.zeros((len(sphere_temp)), dtype=bool)                                                                          # radii_check tracks whether or not the free volume sphere size has been determined
    #
    #            # Find the approximate center of the voxel cube to find the system atoms near the voxel cube, where system atoms define the van der Waals volume of the system. Reduces computational cost
    #            center = np.round([vox_x[vox_track[0] + int((x_i - vox_track[0])/2)], vox_y[vox_track[1] + int((y_i - vox_track[1])/2)], vox_z[vox_track[2] + int((z_i - vox_track[2])/2)]], decimals = 5)
    #            sys_mask = distances.capped_distance(center, sys, (np.min(cell[:3]) + np.sqrt(3)*vox_inc*L + 1)/2, box=cell)[0][:,1]
    #
    #            # To reduce the number of calculations and limit memory usage, the distance between voxel-centers and system atoms is done in steps of d_inc Angstroms
    #            d = 0.0; d_inc = 2                                                                                                               # Maximum distance to calculate between every voxel-center and every system atom
    #            remaining = np.where(radii_check == False)[0]                                                                                    # Index of voxel-centers that still need their size determined
    #            while len(remaining) > 0:
    #                d += d_inc
    #
    #                pair_arr, dist_arr = distances.capped_distance(sphere_temp[remaining], sys[sys_mask], d, box=cell)                           # Distance between voxel-centers and system atoms
    #
    #                if len(dist_arr) > 0:
    #                    dist_arr -= sys_radii[sys_mask][pair_arr[:,1]]                                                                           # Subtract radius of each system atom from the distance to get the distance from the voxel-center to the surface of the atom
    #
    #                    # Useful print command for troubleshooting memory problems: prints the distance calulcated out to, the number of voxel-centers, and the total number of distances to system atoms generated
    #                    # Decreasing d_inc or N will reduce the number of distances generated each cycle, reducing memory usage
    #                    if frame == frame_ids[-1]:
    #                        if d == d_inc:
    #                            print("Voxel Block: ", (vox_track/vox_inc).astype(int))
    #                        print("Create Spheres:", d, len(remaining), len(dist_arr))
    #
    #                    # Fill sphere_arr, radii_arr, and radii_check for all voxel-centers that contain system atoms within d distance, where only radii_check is altered if the radius r > probe_radius
    #                    index = 0; c_index = pair_arr[0,0]; skip = 0
    #                    for i,c in enumerate(pair_arr[:,0]):
    #                        if c > c_index:
    #                            r_min = np.round(np.min(dist_arr[index:i]), decimals=5)                                                         # Minimum distance between voxel-center and system atom
    #                            if r_min > probe_radius:
    #                                sphere_arr.append(sphere_temp[remaining[c_index]]); radii_arr.append(r_min)
    #                                radii_check[remaining[c_index]] = True                                                                      # Sphere does not overlap the system and radius r > probe_radius
    #                                
    #                                coords = np.floor(sphere_temp[remaining[c_index]]/np.array((vox_x[0],vox_y[0],vox_z[0]))/2).astype(int)
    #                                sphere_mask[coords[0],coords[1],coords[2]] = 1
    #                            else:
    #                                radii_check[remaining[c_index]] = True                                                                      # Sphere is within the van der waals surface of the system or radius r < probe_radius
    #
    #                                if r_min < 0:                                                                                               # voxel-center is within the van der Waals surface of the system
    #                                    coords = np.floor(sphere_temp[remaining[c_index]]/np.array((vox_x[0],vox_y[0],vox_z[0]))/2).astype(int)
    #                                    sphere_mask[coords[0],coords[1],coords[2]] = -1
    #
    #                            index = i; c_index = c
    #
    #                            if i == len(pair_arr[:,0]) - 1:                                                                                 # Check to make sure the last voxel-center is counted
    #                                skip = 1
    #
    #                    if skip == 0:
    #                        r_min = np.round(np.min(dist_arr[index:]), decimals=5)                                                              # Minimum distance between voxel-center and system atom
    #                        if r_min > probe_radius:
    #                            sphere_arr.append(sphere_temp[remaining[c_index]]); radii_arr.append(r_min)
    #                            radii_check[remaining[c_index]] = True                                                                          # Sphere does not overlap the system and radius r > probe_radius
    #                            
    #                            coords = np.floor(sphere_temp[remaining[c_index]]/np.array((vox_x[0],vox_y[0],vox_z[0]))/2).astype(int)
    #                            sphere_mask[coords[0],coords[1],coords[2]] = 1
    #                        else:
    #                            radii_check[remaining[c_index]] = True                                                                          # Sphere is within the van der waals surface of the system or radius r < probe_radius
    #                            
    #                            if r_min < 0:                                                                                                   # voxel-center is within the van der Waals surface of the system
    #                                coords = np.floor(sphere_temp[remaining[c_index]]/np.array((vox_x[0],vox_y[0],vox_z[0]))/2).astype(int)
    #                                sphere_mask[coords[0],coords[1],coords[2]] = -1
    #                del pair_arr; del dist_arr
    #
    #                remaining = np.where(radii_check == False)[0]
    #            del remaining
    #sphere_arr = np.array(sphere_arr); radii_arr = np.array(radii_arr); max_radius = np.max(radii_arr)
    #del sphere_temp; del radii_check
    #
    ## Useful print command for troubleshooting problems: prints the number of voxel-centers within the free volume, the radius of the largest sphere, and the diameter of the largest sphere (pore)
    ## Also prints the number of voxels within the system van der Waals free volume, voxels containing free volume spheres of radius r >= probe_radius, and voxels that need to be assessed whether they are in the free volume or not
    #if frame%nt == 0:
    #    print("Spheres Created:", len(radii_arr), max_radius, 2 * max_radius)
    #    print("Voxels within system: ", len(sphere_mask[sphere_mask == -1]))
    #    print("Voxels with free volume spheres: ", len(sphere_mask[sphere_mask == 1]))
    #    print("Voxels outside the system: ", len(sphere_mask[sphere_mask == 0]))
    #
    ### Code to write coordinates and radius of each free volume sphere to a .xyz file, which can be visualized in Ovito, etc
    #if frame == frame_ids[-1]:
    #    with open('Free_Volume_Spheres.xyz', 'w') as anaout:
    #        print(str(len(sphere_arr)), file=anaout)
    #        print('Properties=species:S:1:pos:R:3:Radius:R:1', file=anaout)
    #        for i, sph in enumerate(sphere_arr):
    #            print('X {:10.5f} {:10.5f} {:10.5f} {:10.5f}'.format(sph[0] - cell[0]/2, sph[1] - cell[1]/2, sph[2] - cell[2]/2, radii_arr[i]), file=anaout)
    #    print('Free Volume Sphere XYZ File Printed')
    #
    #
    #
    #
    #
    ## This part of the calculation determines the probe-accessible fractional free volume
    ## The code will take the voxelized system and determine if each voxel-center lies within the free volume or not
    ## Changing L, N, and d_inc can reduce run time and memory usage
    #
    #FFV_save = []                                                                                                                               # voxel-centers within the free volume will be re-used for the PSD calculation
    #FFV_block = []; Block_center = []; Block_track = -1                                                                                         # Track the voxel cubes and voxel vubes centers
    #FFV_track = 0; FFV_total = len(vox_x)*len(vox_y)*len(vox_z)                                                                                 # Number of voxels within the free volume; Total number of voxels
    #
    #N = 100000                                                                                                                                  # To improve efficiency, voxels are looped over in cubes of ~N voxel-centers
    #vox_inc = int(N**(1/3)); vox_track = np.array((0,0,0), dtype=int); vox_track[0] = -vox_inc                                                  # vox_inc = side length of voxel cube, vox_track tracks the location of the cubes in x, y, and z compared to the position in vox_x, vox_y, and vox_z
    #for x_i in np.arange(vox_inc,len(vox_x)+vox_inc,vox_inc):
    #    vox_track[0] += vox_inc
    #    if x_i > len(vox_x):
    #        x_i = len(vox_x)
    #
    #    vox_track[1] = -vox_inc
    #    for y_i in np.arange(vox_inc,len(vox_y)+vox_inc,vox_inc):
    #        vox_track[1] += vox_inc
    #        if y_i > len(vox_y):
    #            y_i = len(vox_y)
    #
    #        vox_track[2] = -vox_inc
    #        for z_i in np.arange(vox_inc,len(vox_z)+vox_inc,vox_inc):
    #            vox_track[2] += vox_inc
    #            if z_i > len(vox_z):
    #                z_i = len(vox_z)
    #
    #            FFV_probe = np.zeros((N, 3)); count = 0; Block_track += 1                                                                       # FFV_probe contains the position of the voxel-centers within the cube of size N that still need to be calculated
    #            for i,x in enumerate(vox_x[vox_track[0]:x_i]):
    #                for j,y in enumerate(vox_y[vox_track[1]:y_i]):
    #                    for k,z in enumerate(vox_z[vox_track[2]:z_i]):
    #
    #                        # If voxel-center is the center of a free volume sphere, count it and skip it
    #                        if sphere_mask[i,j,k] == 1:
    #                            FFV_track += 1; FFV_save.append(np.round([x,y,z], decimals = 5)); FFV_block.append(Block_track)
    #                            continue
    #                        
    #                        # If voxel-center is within the van der Waals volume of the system, skip it
    #                        elif sphere_mask[i,j,k] == -1:
    #                            continue
    #                        
    #                        FFV_probe[count] = np.round([x,y,z], decimals = 5); count += 1
    #            FFV_probe = np.round(FFV_probe[:count], decimals = 5)
    #            FFV_check = np.zeros((len(FFV_probe)), dtype=bool)                                                                              # FFV_check tracks whether or not the voxel-center has been analyzed
    #
    #            # Find the approximate center of the voxel cube to find the free volume spheres near the voxel cube, where free volume spheres define the free volume of the system. Reduces computational cost
    #            center = np.round([vox_x[vox_track[0] + int((x_i - vox_track[0])/2)], vox_y[vox_track[1] + int((y_i - vox_track[1])/2)], vox_z[vox_track[2] + int((z_i - vox_track[2])/2)]], decimals = 5); Block_center.append(center)
    #
    #            # To reduce the number of calculations and limit memory usage, the distance between voxel-centers and free volume spheres is done in steps of d_inc Angstroms
    #            d = 0; d_inc = 2                                                                                                                # Maximum distance to calculate between every voxel-center and every free volume sphere
    #            remaining = np.where(FFV_check == False)[0]                                                                                     # Index of voxel-centers that still need their location determined
    #            while d < max_radius and d < 3*probe_radius:
    #                d += d_inc
    #
    #                # Only need to measure distance from voxel-centers to free volume spheres out to the maximum radius of the free volume spheres
    #                if len(remaining) == 0:
    #                    d = max_radius + 0.5
    #                
    #                sph_mask = distances.capped_distance(center, sphere_arr, d + np.sqrt(3)*vox_inc*L/2 + 0.5, box=cell)[0][:,1]                # Free volume spheres near the voxel cube
    #
    #                pair_arr, dist_arr = distances.capped_distance(FFV_probe[remaining], sphere_arr[sph_mask], d, box=cell)                     # Distance between each voxel-center and the free volume sphere centers generated in the code above
    #                if len(dist_arr) > 0:
    #                    dist_arr -= radii_arr[sph_mask][pair_arr[:,1]]                                                                          # Subtract radius of each free volume sphere to find distance between voxel-center and the surface of the sphere
    #
    #                    # Useful print command for troubleshooting memory problems: prints the maximum distance calculated between voxel-center and free volume spheres, the number of voxel-centers, and the number of distances generated
    #                    # Decreasing d_inc or N will reduce the number of distances generated each cycle, reducing memory usage
    #                    if frame%nt == 0:
    #                        if d == d_inc:
    #                            print("Voxel Block: ", (vox_track/vox_inc).astype(int))
    #                        print("FFV Probes:", d, len(remaining), len(dist_arr))
    #
    #                    # Fill FFV_check for all voxel-centers located within a free volume sphere
    #                    index = 0; c_index = pair_arr[0,0]; skip = 0
    #                    for i,c in enumerate(pair_arr[:,0]):
    #                        if c > c_index:
    #                            if np.any(dist_arr[index:i] < 0):                                                                               # If the voxel-center lies within a free volume sphere, mark it as no longer needing to be calculated and increase the number of voxels within the free volume by 1
    #                                FFV_check[remaining[c_index]] = True
    #                                FFV_save.append(FFV_probe[remaining[c_index]]); FFV_block.append(Block_track)                               # Save the voxel-center for later use in PSD caluclations
    #                                FFV_track += 1
    #                            else:                                                                                                           # If the voxel-center is within d of at least 1 free volume sphere but it doesn't lie within it, it is likely not within the free volume of any sphere, significantly reduces computation time
    #                                FFV_check[remaining[c_index]] = True
    #                            index = i; c_index = c
    #
    #                            if i == len(pair_arr[:,0]) - 1:                                                                                 # Check to make sure the last voxel-centers is counted
    #                                skip = 1
    #                                
    #                    if skip == 0:
    #                        if np.any(dist_arr[index:] < 0):                                                                                    # If the voxel-center lies within a free volume sphere, mark it as no longer needing to be calculated and increase the number of voxels within the free volume by 1
    #                            FFV_check[remaining[c_index]] = True
    #                            FFV_save.append(FFV_probe[remaining[c_index]]); FFV_block.append(Block_track)                                   # Save the voxel-center for later use in PSD caluclations
    #                            FFV_track += 1
    #                        else:                                                                                                               # If the voxel-center is within d of at least 1 free volume sphere but it doesn't lie within it, it is likely not within the free volume of any sphere, significantly reduces computation time
    #                            FFV_check[remaining[c_index]] = True
    #                remaining = np.where(FFV_check == False)[0]
    #                del pair_arr; del dist_arr
    #        
    #            ## Useful print command to track the probe-accessible free volume every loop
    #            #if frame == frame_ids[-1]:
    #            #    print("FFV:", FFV_total, FFV_track / FFV_total)
    #
    ## Useful print command to track the final FFV
    #if frame%nt == 0:
    #    print("FFV Final:", FFV_total, FFV_track / FFV_total)
    #del remaining; del FFV_check; del sphere_mask; FFV_block = np.array(FFV_block); Block_center = np.array(Block_center)
    #
    ### Code to write coordinates of each voxel-center to a .xyz file, which can be visualized in Ovito
    ##if frame == frame_ids[-1]:
    ##    with open('Free_Volume_Voxels.xyz', 'w') as anaout:
    ##        print(str(len(FFV_save)), file=anaout)
    ##        print('Properties=species:S:1:pos:R:3:Radius:R:1', file=anaout)
    ##        for i, sph in enumerate(FFV_save):
    ##            print('X {:10.5f} {:10.5f} {:10.5f} {:10.5f}'.format(sph[0] - cell[0]/2, sph[1] - cell[1]/2, sph[2] - cell[2]/2, L/2), file=anaout)
    ##    print('Free Volume Voxel XYZ File Printed')
    #
    #
    #
    #
    #
    #
    ## This part of the calculation determines the cumulative probe-accessible pore size distribution, where the distribution is defined as the probability that a random point (voxel) within the free volume resides within a free volume sphere of diameter d with minimum size probe_radius
    ## This code will take each voxel within the probe-accessible free volume and determine the largest free volume sphere that contains that point
    ## Changing L, N, and d_inc can reduce run time and memory usage
    #
    #d_arr = np.arange(0, 50.0 + 0.10, 0.10); PSD_arr = np.zeros_like(d_arr, dtype=int)                                                      # d_arr is the histogram of free volume sphere sizes; PSD_arr tracks the number of instances of voxels contained within free volume spheres of size at least d
    #
    #PSD_probes = np.array(FFV_save)
    #
    ## To improve efficiency, voxels are looped over the cubes generated in the FFV calculation
    ## This is an expensive calculation, so cubes are looped over N voxel-centers at a time
    ## There is no d_inc variable here, as it would not be useful
    #N = 1000
    #for block, center in enumerate(Block_center):
    #    PSD_block = PSD_probes[FFV_block == block]
    #
    #    if len(PSD_block) == 0:
    #        continue
    #    
    #    # Find the approximate center of the voxel cube to find the free volume spheres near the voxel cube, where free volume spheres define the free volume of the system. Reduces computational cost
    #    sph_mask = distances.capped_distance(center, sphere_arr, max_radius + np.sqrt(3)*vox_inc*L/2 + 0.5, box=cell)[0][:,1]
    #
    #    PSD_temp = np.zeros((N, 3))                                                                                                         # PSD_temp contains the position of the N voxel-centers within the voxel cube
    #    for count in range(1, N+1):
    #        PSD_temp[(count % len(PSD_temp)) - 1] = PSD_block[count - 1]
    #
    #        if ((count != len(PSD_probes)) and (count % len(PSD_temp) > 0)):
    #            continue
    #        PSD_temp = np.round(PSD_temp[:count], decimals = 5)
    #
    #        pair_arr, dist_arr = distances.capped_distance(PSD_temp, sphere_arr[sph_mask], max_radius+0.5, box=cell)                        # Find distance between voxel-centers and free volume sphere centers
    #
    #        if len(dist_arr) > 0:
    #            dist_arr -= radii_arr[sph_mask][pair_arr[:,1]]                                                                              # Subtract radius of each free volume sphere to find distance between voxel-centers and the surface of the sphere
    #
    #            # Useful print command for troubleshooting memory problems: prints the maximum distance calculated between voxel-centers and free volume sphere centers, the number of voxels, and the number of distances generated
    #            if frame == frame_ids[-1]:
    #                if count <= len(PSD_temp):
    #                    print("Voxel Block: ", block, " of ", len(Block_center)-1)
    #                print("PSD Probes:", max_radius+0.5, len(PSD_temp), len(dist_arr))
    #
    #            # Fill PSD_arr for all voxel-centers located within a free volume sphere
    #            index = 0; c_index = pair_arr[0,0]; skip = 0
    #            for i,c in enumerate(pair_arr[:,0]):
    #                if c > c_index:
    #                    contained = pair_arr[index:i][np.where(dist_arr[index:i] < 0)[0],1]                                                 # Index of all free volume spheres containing the voxel-center
    #
    #                    if len(contained) > 0:                                                                                              # If at least 1 free volume sphere contains the voxel-center, find the largest sphere and add it to PSD_arr in a cumulative manner
    #                        max_size = 2 * np.max(radii_arr[contained])
    #                        PSD_arr[np.where(d_arr <= max_size)[0]] += 1
    #                    index = i; c_index = c
    #
    #                    if i == len(pair_arr[:,0]) - 1:                                                                                     # Check to make sure the last voxel-center is counted
    #                        skip = 1
    #
    #            if skip == 0:
    #                contained = pair_arr[index:][np.where(dist_arr[index:] < 0)[0],1]                                                       # Index of all free volume spheres containing the voxel-center
    #                if len(contained) > 0:                                                                                                  # If at least 1 free volume sphere contains the voxel-center, find the largest sphere and add it to PSD_arr in a cumulative manner
    #                    max_size = 2 * np.max(radii_arr[contained])
    #                    PSD_arr[np.where(d_arr <= max_size)[0]] += 1
    #        PSD_temp = np.zeros((N, 3))  
    #        del pair_arr; del dist_arr
    #
    #        ## Useful print command to track the probe-accessible PSD every loop
    #        #if frame == frame_ids[-1]:
    #        #    print("PSD:", PSD_arr[0])
    #        #    print_string=''
    #        #    for i in PSD_arr:
    #        #        if i == 0:
    #        #            continue
    #        #        print_string += str(np.round(i / PSD_arr[0], decimals=5)) + ' '
    #        #    print(print_string)
    #
    ## Useful print command to track the final PSD
    #if frame%nt == 0:
    #    print("PSD Final:", PSD_arr[0])
    ##    print_string=''
    ##    for i in PSD_arr:
    ##        if i == 0:
    ##            continue
    ##        print_string += str(np.round(i / PSD_arr[0], decimals=5)) + ' '
    ##    print(print_string)
    #del contained
    #
    #
    #
    #
    #
    ## Return the necessary information to complete the calculations: FFV_track / FFV_total gives the pore-accessible free volume, PSD_arr / PSD_arr[0] gives the pore-accessible PSD
    #PSD_arr = np.insert(PSD_arr, 0, FFV_total); PSD_arr = np.insert(PSD_arr, 0, FFV_track)
    #return PSD_arr
    


def load_TRR():
# loads in the trajectory and saves the necessary data to a temporary h5py file

    global t_min, t_max, N_frames

    uta = mda.Universe(top_file, trj_file)  # Load in the trajectory and topology
    system = uta.select_atoms(system_name)  # Define the system atoms
    
    # Create an array that tracks the radius of each system atom based on Size_array
    sys_names = system.names; sys_radii = []
    for name in sys_names:
        name = str(name)[0]
        if name in Size_arr[:,0]:
            sys_radii.append(float(Size_arr[np.where(Size_arr[:,0] == name)[0][0],1]))
        else:
            print("Missing Atom Name and Size in Size_arr (See Atom Name Below)")
            print(name)
            exit()
    sys_radii = np.array(sys_radii)

    # Define the system times/frames to be calculated over
    if t_min == -1:
        t_min = uta.trajectory[0].time
    if t_max == -1:
        t_max = uta.trajectory[-1].time
    if N_frames < 1:
        N_frames = nt
    dt = np.round((uta.trajectory[1].time - uta.trajectory[0].time),3)
    if N_frames == 1:
        frame_ids = np.array([int((t_max - uta.trajectory[0].time)/dt)], dtype=int)
    else:
        frame_ids = np.linspace(int((t_min - uta.trajectory[0].time)/dt), int((t_max - uta.trajectory[0].time)/dt), N_frames, dtype=int)
        print("Timestep: ~" + str(dt*(frame_ids[1] - frame_ids[0])) + " ps")
    print("Number of Frames: " + str(len(frame_ids)))

    # Load in the necessary data: "system" atom positions, cell dimensions
    r_system = []; cells = []
    for frame in frame_ids:

        if frame%5000 == 0:
            print("Frame " + str(frame))

        ts = uta.trajectory[frame]
        cell = ts.dimensions
        cells.append(cell)

        r_system.append(system.positions)

    # Save necessary infomration to a .hdf5 file for later use in the calculation
    with h5py.File('/tmp/PSD.hdf5','w') as f:
        dset1 = f.create_dataset("system", data=r_system)
        dset2 = f.create_dataset("cells", data = cells)
        dset3 = f.create_dataset("frames", data = frame_ids)
        dset4 = f.create_dataset("sys_radii", data = sys_radii)



def main(trj_file, top_file, system_name, probe_radius, t_min, t_max, N_frames, nt):

    # Load in the trajectory file and exit the code to purge the memory before multi-threading
    if not os.path.exists('/tmp/PSD.hdf5'):
        load_TRR()
        exit()

    with h5py.File('/tmp/PSD.hdf5','r') as f:
        dset1 = f['frames']; frame_ids = dset1[:]
    frame_ids = np.arange(0,len(frame_ids),1)

    print("FFV/PSD Analysis")
    # Perform the analysis using multi-threading
    pool = mp.Pool(processes=nt)
    func = functools.partial(iDist)
    radii_arr = pool.map(func, list(frame_ids))
    pool.close()
    pool.join()
    radii_arr = np.array(radii_arr)

    # Return the average and standard deviation (over the frames processed) of the probe-accessible fractional free volume
    FFV = radii_arr[:,:2]; FFV = FFV[:,0] / FFV[:,1]; FFV = np.array([np.mean(FFV), np.std(FFV)])
    with open('FFV.xvg', 'w') as anaout:
        print("# FFV Std", file=anaout)
        print('0.0 {:10.5f} {:10.5f}'.format(FFV[0], FFV[1]), file=anaout)

    # Return the average and standard deviation (over the frames processed) of the probe-accessible pore size ditribution
    d_arr = np.arange(0, 50.0 + 0.10, 0.10)
    PSD_all = radii_arr[:,2:]; PSD_all = np.divide(PSD_all.T, PSD_all[:,0], dtype=float).T
    PSD_Cumulative = np.array([np.mean(PSD_all, axis=0), np.std(PSD_all, axis = 0)])
    # PSD is the derivative of the cumulative PSD
    PSD = np.array([np.mean((PSD_all[:,:len(d_arr)-2] - PSD_all[:,2:])/(d_arr[2:] - d_arr[:len(d_arr)-2]), axis=0), np.std((PSD_all[:,:len(d_arr)-2] - PSD_all[:,2:])/(d_arr[2:] - d_arr[:len(d_arr)-2]), axis=0)])

    with open('Cumulative_PSD.xvg', 'w') as anaout:
        print("# Cumulative_PSD Std", file=anaout)
        for i in range(len(d_arr)):
            print(' {:10.5f} {:10.5f} {:10.5f}'.format(np.round(d_arr[i], decimals=3), PSD_Cumulative[0,i], PSD_Cumulative[1,i]), file=anaout)

    with open('PSD.xvg', 'w') as anaout:
        print("# PSD Std", file=anaout)
        for i in range(len(d_arr)-1):
            if i == 0:
                print(' {:10.5f} {:10.5f} {:10.5f}'.format(d_arr[i], 0.0, 0.0), file=anaout)
            else:
                print(' {:10.5f} {:10.5f} {:10.5f}'.format(np.round(d_arr[i], decimals=3), PSD[0,i-1], PSD[1,i-1]), file=anaout)

    # Deletes the temporary h5py file
    os.remove('/tmp/PSD.hdf5')

if __name__ == "__main__":
    main(trj_file, top_file, system_name, probe_radius, t_min, t_max, N_frames, nt)
