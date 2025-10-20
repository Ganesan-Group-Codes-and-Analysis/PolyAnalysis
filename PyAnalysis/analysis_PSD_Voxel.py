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

# This script determines the free volume (pore) size distribution based on the van der Waals volume of the system
# This code was specifically desgined to find the distribution of water-rich pores within a hydrated polymer system
# The output includes the Cumulative Pore Size Distribution (Cumulative PSD), Pore Size Distribution (PSD), and Free Volume Fraction (Fractional Free Volume, FFV).
# This code was written based on the methods used for PoreBlazer: https://github.com/SarkisovGitHub/PoreBlazer
#
# As written, this code reads in GROMACS trajectory data or PoreBlazer XYZ and DAT data using MDAnalysis
# As written, this code is designed for 3D-periodic rectangular simulations
#
# When implementing this code, it is recommended to test different values of L_voxel to ensure convergence of the FFV as L_voxel decreases. Note, computation time and memory usage will grow significantly as L_voxel decreases.
# If you run into memory or extreme run times, there are debugging lines throughout the code, and several values you can change to increase or decrease memory usage.
# There are two instances where .xyz files can be created to visualize 1) probe-accessible spheres of maximum radius without overlapping the van der Waals volume of the systems, and 2) voxel-centers that lie within the probe-accessible volume

import MDAnalysis as mda
import MDAnalysis.analysis.distances as dist
import MDAnalysis.lib.distances as distances
import numpy as np

import multiprocessing as mp
import functools
import h5py
import os
from sys import argv

script, trj_file, top_file, system_name, probe_radius, t_min, t_max, N_frames, nt = argv
probe_radius = float(probe_radius); t_min = float(t_min); t_max = float(t_max); N_frames = int(N_frames); nt = int(nt)
# trj_file     = .trr/.xtc/.gro file, see below for XYZ files adapted from PoreBlazer analyses
# top_file     = .tpr file, see below for DAT files adapted from PoreBlazer analyses
# system_name  = names of atoms that make up the system matrix in a form acceptable by MDAnalysis.select_atoms(), e.g., "moltype MOL", "moltype MOL or resname LI", "moltype MOL and not name LI", "all"
# probe_radius = radius in Angstroms of the probe for obtaining the probe-accessible PSD and FFV, e.g., water molecules are 1.4 Angstroms
#                minimum probe_radius is the voxel size
# t_min        = start time for analysis in ps (-1 assumes the start time of the first frame)
# t_max        = end time for analysis in ps (-1 assumes the end time of the last frame)
# N_frames     = number of frames to analyze (-1 assumes N_frames = nt) -> for efficiency, N_frames should be a multiple of the number of threads
#                N_frames = 1 defaults to the frame at t_max
# nt           = number of threads

# To adapt to PoreBlazer input data
# trj_file = .xyz file
# top_file = 'input.dat'
# system_name = '', XYZ atoms do not have defined residues. Therefore, ALL atoms are assumed to define the system, and solvent atoms must be removed to probe solvent free volume
# e.g., python3 analysis_PSD_Voxel.py md.xyz input.dat "" 1.4 -1 -1 -1 1
#
# input.dat example
# polymer_matrix.xyz            # .xyz file name
# 96.65307 96.65307 96.65307    # Box dimensions: box length
# 90 90 90                      # Box dimensions: rectangular

# Example of Use: python3 analysis_PSD_Voxel.py md.xtc md.tpr "moltype MOL" 1.4 90000 100000 80 80
#                 Polymer chains are molecules with name MOL -> this defines the "system" domain
#                 probe_radius = 1.4 is approximately the size of a water molecule
#                 The run time of this code is highly dependent on the number of frames -> N_frames shouldn't be too many multiples of nt
#                     Note: each frame takes about the same amount of time to process, so it is efficient for the number of frames examined to be a multiple of nt

# Van der waals radii from Bondi (1964)
# https://www.knowledgedoor.com/2/elements_handbook/bondi_van_der_waals_radius.html
Size_arr = np.array([('C',  1.70), ('O',  1.52), ('H',  1.20), ('N',  1.55), ('S',  1.80), ('F',  1.47), ('P',  1.80),
                     ('LI', 1.82), ('NA', 2.27), ('K',  2.75), ('MG', 1.73), ('CL', 1.75), ('BR', 1.85), ('I',  1.98)])
# Define dummy atom namess within your system which should not be included in the system analysis. MW is common for TIP4P.
Dummy_atoms = np.array(['MW'])

# Voxel side length in angstroms, where larger values sacrifice accuracy for efficiency.
L_voxel = 0.20
if probe_radius < L_voxel:
    probe_radius = L_voxel

# d_max is the largest free volume diameter to measure PSD out to; d_step is the stepsize between free volume diameter bins in the PSD
d_max = 50; d_step = 0.50

# Efficiency parameters: maximum number of distance calculations-per-loop, where sph and PSD refer to the generation of free volume spheres and calculating the PSD, respectively. Incremental increase in cutoff for distance calculations in the generation of free volume spheres
# Lower numbers use less memory. 10e9 distance calculations was found to be a good number, where N_calc_sph can be larger due to code efficiencies such that N_calc_true < N_calc_sph under all conditions
# Use print statements within the loops to see number of distance calculations-per-loop and adjust as needed.
N_calc_sph = 20e9      # Maximum number of distance calculations-per-loop. A lower number uses less memory.
N_calc_PSD = 10e9      # Maximum number of distance calculations-per-loop. A lower number uses less memory.
d_inc = 2              # Incremental size of distance cutoff while generating free volume spheres. A lower number uses less memory.



def iDist(frame):
    if frame%nt == 0:
        print("Frame " + str(frame))

    with h5py.File('/tmp/PSD.hdf5','r') as f:
        dset1 = f['system']; sys = dset1[frame]                                                                                                 # Position of all system atoms
        dset2 = f['sys_radii']; sys_radii = dset2[:]                                                                                            # Radius of all system atoms, where distance between sphere (see below) and polymer atom minus the radius is the distance to the van der waals surface of the atom
        dset3 = f['cells']; cell = dset3[frame]                                                                                                 # Size of the cell
        dset4 = f['frames']; frame_ids = dset4[:]; frame_ids = np.arange(0,len(frame_ids),1)





    # This part of the calculation determines the maximum size of voxel-centered free volume spheres without overlapping system atoms, where the total volume of all spheres larger than probe_radius defines the probe-accessible free volume of the system
    # This code will generate points at the center of voxels with side length L_voxel and grow these points into free volume spheres
    # Changing L_voxel, N_calc_sph, and d_inc can reduce run time and memory usage
    vox_x = np.round(np.linspace(0, cell[0], num = int(cell[0]/L_voxel)), decimals=5); vox_x = np.round((vox_x[:-1] + vox_x[1:])/2, decimals=5) # Voxel-centers in the x direction
    vox_y = np.round(np.linspace(0, cell[1], num = int(cell[1]/L_voxel)), decimals=5); vox_y = np.round((vox_y[:-1] + vox_y[1:])/2, decimals=5) # Voxel-centers in the y direction
    vox_z = np.round(np.linspace(0, cell[2], num = int(cell[2]/L_voxel)), decimals=5); vox_z = np.round((vox_z[:-1] + vox_z[1:])/2, decimals=5) # Voxel-centers in the z direction

    radii_arr = np.zeros((len(vox_x),len(vox_y),len(vox_z)))                                                                                    # radii_arr tracks free volume sphere indices (position in array = position in voxelized system) and radius (value at that position), where we are interested in spheres of radius r >= probe_radius. All probes of r > 0 are saved for later use.
    
    N_cube = int((N_calc_sph/len(sys))**(1/3))**3                                                                                               # To improve efficiency, voxels are looped over in cubes of N_cube voxel-centers
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

                sphere_temp = np.vstack(np.meshgrid(vox_x[vox_track[0]:x_i],vox_y[vox_track[1]:y_i],vox_z[vox_track[2]:z_i])).reshape(3,-1).T   # sphere_temp contains the position of the voxel-centers within the cube of size N_cube

                # Find the approximate center of the voxel cube to find the system atoms near the voxel cube (sys_mask), where system atoms define the van der Waals volume of the system. Reduces computational cost
                center = np.round([vox_x[vox_track[0] + int((x_i - vox_track[0])/2)], vox_y[vox_track[1] + int((y_i - vox_track[1])/2)], vox_z[vox_track[2] + int((z_i - vox_track[2])/2)]], decimals = 5)

                # To reduce the number of calculations and limit memory usage, the distance between voxel-centers and system atoms is done in steps of d_inc Angstroms
                d = 0.0                                                                                                                         # Maximum distance to calculate between every voxel-center and every system atom
                while len(sphere_temp) > 0:
                    d += d_inc

                    sys_mask = distances.capped_distance(center, sys, d + np.sqrt(3)*vox_inc*L_voxel/2 + 2*L_voxel, box=cell)[0][:,1]           # System atoms near the voxel cube

                    pair_arr, dist_arr = distances.capped_distance(sphere_temp, sys[sys_mask], d, box=cell)                                     # Distance between voxel-centers and system atoms

                    ## Useful print command for troubleshooting memory problems
                    ## Decreasing d_inc or N_calc_sph will reduce the number of distances generated each cycle, reducing memory usage
                    #if frame == frame_ids[-1]:
                    #    if d == d_inc:
                    #        print("Voxel Block: ", (vox_track/vox_inc).astype(int))
                    #    print("Distance, Calculations, Writes: {:2.1f} {:.1e} {:.1e}".format(d, len(sphere_temp)*len(sys[sys_mask]), len(dist_arr)))

                    if len(dist_arr) > 0:
                        dist_arr -= sys_radii[sys_mask][pair_arr[:,1]]                                                                          # Subtract radius of each system atom from the distance to get the distance from the voxel-center to the surface of the atom

                        # Fill radii_arr for all voxel-centers that contain system atoms within d distance, where the smallest distance is the radius of the free volume sphere centered on the voxel
                        index = 0; sph_save = pair_arr[0,0]
                        sphere_remove = []
                        for i,sph in enumerate(pair_arr[:,0]):
                            if sph > sph_save:
                                r_min = np.round(np.min(dist_arr[index:i]), decimals=5)                                                         # Minimum distance between voxel-center and system surface

                                if r_min > 0:                                                                                                   # Sphere does not overlap the system and radius >= 0
                                    coords = np.divide(sphere_temp[sph_save], np.array([vox_x[1] - vox_x[0],vox_y[1] - vox_y[0],vox_z[1] - vox_z[0]])).astype(int)
                                    radii_arr[coords[0],coords[1],coords[2]] = r_min

                                sphere_remove.append(sph_save)                                                                                  # Analysis complete, remove from future distance calculations
                                index = i; sph_save = sph

                        if i > index:                                                                                                           # Check to make sure the last voxel-center is counted
                            r_min = np.round(np.min(dist_arr[index:]), decimals=5)                                                              # Minimum distance between voxel-center and system surface

                            if r_min > 0:                                                                                                       # Sphere does not overlap the system and radius >= 0
                                coords = np.divide(sphere_temp[sph_save], np.array([vox_x[1] - vox_x[0],vox_y[1] - vox_y[0],vox_z[1] - vox_z[0]])).astype(int)
                                radii_arr[coords[0],coords[1],coords[2]] = r_min

                            sphere_remove.append(sph_save)                                                                                      # Analysis complete, remove from future distance calculations
                    sphere_temp = np.delete(sphere_temp, np.array(sphere_remove), axis=0)                                                       # Remove evaluated voxel-centers
    max_radius = np.max(radii_arr)

    ## Useful print command for troubleshooting problems
    ## Also prints the number of voxels within the system van der Waals free volume, voxels containing free volume spheres of radius r >= probe_radius, and voxels that need to be assessed whether they are in the free volume or not
    #if frame == frame_ids[-1]:
    #    print("\nMaximum pore diameter: ", 2 * max_radius)
    #    print("Spheres within the free volume: ", len(radii_arr[radii_arr >= probe_radius]))
    #    print("Voxels not within the system: ", len(radii_arr[radii_arr != 0]))

    ### Code to write coordinates and radius of each free volume sphere to a .xyz file, which can be visualized in Ovito, etc
    #if frame == frame_ids[-1]:
    #    idx_x, idx_y, idx_z = np.where(radii_arr >= probe_radius)
    #    with open('Free_Volume_Spheres.xyz', 'w') as anaout:
    #        print(str(len(idx_x)), file=anaout)
    #        print('Properties=species:S:1:pos:R:3:Radius:R:1', file=anaout)
    #        for i in range(len(idx_x)):
    #            if '.xyz' in trj_file:
    #                print('X {:10.5f} {:10.5f} {:10.5f} {:10.5f}'.format(vox_x[idx_x[i]], vox_y[idx_y[i]], vox_z[idx_z[i]], radii_arr[idx_x[i],idx_y[i],idx_z[i]]), file=anaout)
    #            else:
    #                print('X {:10.5f} {:10.5f} {:10.5f} {:10.5f}'.format(vox_x[idx_x[i]] - cell[0]/2, vox_y[idx_y[i]] - cell[1]/2, vox_z[idx_z[i]] - cell[2]/2, radii_arr[idx_x[i],idx_y[i],idx_z[i]]), file=anaout)
    #    print('Free Volume Sphere XYZ File Printed')





    # This part of the calculation determines the free volume fraction and cumulative probe-accessible pore size distribution, where the distribution is defined as the probability that a random point (voxel) within the free volume resides within a free volume sphere of diameter d with minimum size probe_radius
    # This code will take each voxel not within the system volume (PSD_probes) and determine 1) if it lies within the free volume (FFV), and 2) the largest free volume sphere it lies within (PSD)
    # Changing L_voxel, N_calc_PSD, and d_step can reduce run time and memory usage

    idx_x, idx_y, idx_z = np.where(radii_arr != 0); radii_arr[radii_arr < probe_radius] *= 0                                                    # Indices of all free volume voxels
    PSD_probes = np.vstack((vox_x[idx_x],vox_y[idx_y],vox_z[idx_z])).T                                                                          # Position of all free volume voxels

    FFV_track = 0; FFV_total = len(vox_x)*len(vox_y)*len(vox_z)                                                                                 # Track number of voxels within the free volume against the total number to get FFV
    d_arr = np.arange(0, d_max + d_step, d_step); PSD_arr = np.zeros_like(d_arr, dtype=int)                                                     # d_arr is the histogram of free volume sphere sizes; PSD_arr tracks the number of instances of voxels contained within free volume spheres of size at least d

    FFV_save = np.zeros((1,3)) - 1                                                                                                              # Save voxel-centers within the free volume incase you want it to be printed for visualization
    
    # Starting from the largest free volume spheres, find all free volume voxels within the desired free volume domain for FFV and PSD calulcations
    for d in np.round(np.arange(d_max, 0, -d_step), decimals = 5):
        if 2*max_radius > d:
            print("Largest free volume element lies outside the defined bounds for the PSD. Update d_max accordingly.")
            break

        if d - d_step > 2*max_radius:
            continue
        
        if d < 2*probe_radius:
            break
        
        if len(PSD_probes) == 0:
            break
        
        # For efficiency, we measure the distance between free volume spheres and the voxel-centers starting with the largest d_arr bin and moving down
        idx_x, idx_y, idx_z = np.where((radii_arr <= d/2) & (radii_arr > (d - d_step)/2))
        sphere_temp = np.vstack((vox_x[idx_x],vox_y[idx_y],vox_z[idx_z])).reshape(3,-1).T; radii_temp = radii_arr[idx_x, idx_y, idx_z]          # Positions (sphere_temp) and radii (radii_temp) of free volume spheres in the current PSD bin, radius (d - d_step)/2 < r <= d/2
        if len(sphere_temp) == 0:
            continue
        
        # For efficiency, we limit the number of free volume spheres per loop to a total of N_calc_PSD distance calculations
        count = 0
        while count < len(sphere_temp) and len(PSD_probes) > 0:
            count_old = count; count += min(int(N_calc_PSD/len(PSD_probes)), len(sphere_temp)-count_old)

            sph_temp = sphere_temp[count_old:count]; rad_temp = radii_temp[count_old:count]
            
            pair_arr, dist_arr = distances.capped_distance(sph_temp, PSD_probes, d/2 + 0.5, box=cell)                                           # Distance between free volume spheres and voxel-centers

            ## Useful print command for troubleshooting memory problems
            ## Decreasing N_calc_PSD will reduce the number of distances generated each cycle, reducing memory usage
            #if frame == frame_ids[-1]:
            #    if count_old == 0:
            #        print("Diameter: {} < d <= {}".format(d - d_step, d))
            #    print("Calulcations, Writes: {:.1e}".format(len(sph_temp)*len(PSD_probes), len(dist_arr)))

            if len(dist_arr) > 0:
                dist_arr -= rad_temp[pair_arr[:,0]]                                                                                             # Subtract radius of each free volume sphere from the distance to get the distance from the voxel-center to the surface of the free volume sphere
                pair_arr = np.unique(pair_arr[:,1][dist_arr < 0])                                                                               # Only consider voxel-centers that lie within the free volume sphere (adjusted distance < 0), and only count each occurence once (unique)

                FFV_track += len(pair_arr); PSD_arr[np.where(d_arr < d)[0]] += len(pair_arr)                                                    # Voxel-centers w/n free volume sphere count towards the FFV and cumulatively towards the PSD
                if np.all(FFV_save[0] == -1):                                                                                                   # Save free volume voxel-centers for printing
                    FFV_save = PSD_probes[pair_arr]
                else:
                    FFV_save = np.append(FFV_save, PSD_probes[pair_arr], axis=0)
                PSD_probes = np.delete(PSD_probes, pair_arr, axis=0)                                                                            # No longer consider voxel-centers that are found within a free volume sphere in future loops (prevent double-counting)
    
    ## Code to print the final FFV and PSD for the last frame analyzed
    #if frame == frame_ids[-1]:
    #    print("\nFFV: ", FFV_track/FFV_total, FFV_track, FFV_total)
    #    print("\nPSD Final:", PSD_arr[0])
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
    #            if '.xyz' in trj_file:
    #                print('X {:10.5f} {:10.5f} {:10.5f} {:10.5f}'.format(sph[0], sph[1], sph[2], L_voxel/2), file=anaout)
    #            else:
    #                print('X {:10.5f} {:10.5f} {:10.5f} {:10.5f}'.format(sph[0] - cell[0]/2, sph[1] - cell[1]/2, sph[2] - cell[2]/2, L_voxel/2), file=anaout)
    #    print('Free Volume Voxel XYZ File Printed')
    




    # Return the necessary information to complete the calculations: FFV_track / FFV_total gives the pore-accessible free volume, PSD_arr / PSD_arr[0] gives the pore-accessible PSD
    PSD_arr = np.insert(PSD_arr, 0, FFV_total); PSD_arr = np.insert(PSD_arr, 0, FFV_track)
    return PSD_arr
    


def load_TRR():
# loads in the trajectory and saves the necessary data to a temporary h5py file

    global t_min, t_max, N_frames, nt

    if '.xyz' in trj_file:
        uta = mda.Universe(trj_file)         # Load in the .xyz trajectory
        system = uta.select_atoms('all')     # Define the system atoms - ALL atoms

        # Define the simulation cell from the input.dat file created for PoreBlazer
        cell = np.zeros(6); cell[3:] = 90.0
        with open(top_file, 'r') as file:
            lines = file.readlines()[1]
            cell[:3] += np.array(lines.split(), dtype=float)
    else:
        uta = mda.Universe(top_file, trj_file)        # Load in the trajectory and topology
        system = uta.select_atoms(system_name)        # Define the system atoms

    print("If the following is incorrect, there may be inconsistencies between your atom ID name and the Element name in this script")
    print("\nSYSTEM ATOMS")

    # Remove dummy atoms from the system
    if len(Dummy_atoms) > 0 and len(system) > 0:
        for dummy in Dummy_atoms:
            if np.sum(system.names == dummy) > 0:
                print("Removed {} {} atoms from system analysis".format(np.sum(system.names == dummy), dummy))
                system = system[system.names != dummy]
    
    # Create an array that tracks the radius of each system atom based on Size_array
    sys_names = system.names; sys_radii = []; sys_count = np.zeros((len(Size_arr)), dtype=int)
    for name in sys_names:
        name = str(name)
        if name in Size_arr[:,0]:
            sys_radii.append(float(Size_arr[np.where(Size_arr[:,0] == name)[0][0],1]))
            sys_count[np.where(Size_arr[:,0] == name)[0][0]] += 1
        elif name[0] in Size_arr[:,0]:
            sys_radii.append(float(Size_arr[np.where(Size_arr[:,0] == name[0])[0][0],1]))
            sys_count[np.where(Size_arr[:,0] == name[0])[0][0]] += 1
        else:
            print("Missing Atom Name and Size in Size_arr (See Atom Name Below)")
            print(name)
            exit()
    sys_radii = np.array(sys_radii)

    # Print out system atom information
    print("Element N-in-System")
    for i,j in enumerate(sys_count):
        if j > 0:
            print("{:>7s} {:11d}".format(Size_arr[i,0], j))

    # Define the system times/frames to be calculated over
    print()
    if '.gro' in trj_file or '.xyz' in trj_file:
        nt = 1; N_frames = 1; frame_ids = np.array([0], dtype=int)
    else:
        if t_min == -1:
            t_min = uta.trajectory[0].time
        if t_max == -1:
            t_max = uta.trajectory[-1].time
        if N_frames == -1:
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
        ts = uta.trajectory[frame]

        r_system.append(system.positions)

        if '.xyz' in trj_file:
            cells.append(cell)
        else:
            cell = ts.dimensions
            cells.append(cell)

    # Save necessary infomration to a temporary .hdf5 file for later use in the calculation
    with h5py.File('/tmp/PSD.hdf5','w') as f:
        dset1 = f.create_dataset("system", data=r_system)
        dset2 = f.create_dataset("sys_radii", data = sys_radii)
        dset4 = f.create_dataset("cells", data = cells)
        dset5 = f.create_dataset("frames", data = frame_ids)



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
    d_arr = np.arange(0, d_max + d_step, d_step)
    PSD_all = radii_arr[:,2:]; PSD_all = np.divide(PSD_all.T, PSD_all[:,0], dtype=float).T
    PSD_Cumulative = np.array([np.mean(PSD_all, axis=0), np.std(PSD_all, axis = 0)])
    # PSD is the negative derivative of the cumulative sum
    PSD = np.array([np.mean(-(PSD_all[:,1:] - PSD_all[:,:len(d_arr)-1])/(d_arr[1:] - d_arr[:len(d_arr)-1]), axis=0), np.std(-(PSD_all[:,1:] - PSD_all[:,:len(d_arr)-1])/(d_arr[1:] - d_arr[:len(d_arr)-1]), axis=0)])

    with open('Cumulative_PSD.xvg', 'w') as anaout:
        print("# Cumulative_PSD Std", file=anaout)
        for i in range(len(PSD_Cumulative[0,:])):
            print(' {:10.5f} {:10.5f} {:10.5f}'.format(np.round(d_arr[i], decimals=3), PSD_Cumulative[0,i], PSD_Cumulative[1,i]), file=anaout)

    with open('PSD.xvg', 'w') as anaout:
        print("# PSD Std", file=anaout) 
        for i in range(len(PSD[0,:])):
                if i == 0:
                    print(' {:10.5f} {:10.5f} {:10.5f}'.format(np.round(d_arr[i], decimals=3), 0.0, 0.0), file=anaout)
                else:
                    print(' {:10.5f} {:10.5f} {:10.5f}'.format(np.round(d_arr[i], decimals=3), PSD[0,i], PSD[1,i]), file=anaout)

    # Deletes the temporary .hdf5 file
    os.remove('/tmp/PSD.hdf5')

if __name__ == "__main__":
    main(trj_file, top_file, system_name, probe_radius, t_min, t_max, N_frames, nt)
