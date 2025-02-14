"""
Original code developed by Kara Fong, modified by Nico Marioni

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

# Original Code can be found here: https://github.com/kdfong/transport-coefficients-MSD
# Adjustments were made to load data similarly to my other codes. The base FFT methods have not been changed.
# This script calculates Green-Kubo "MSDs" for retrieving Onsager Coefficients

import multiprocessing as mp
import functools

import numpy as np
import h5py
import MDAnalysis as mda
import MDAnalysis.lib.distances as distances
import os

from sys import argv
script, trj_file, top_file, a_list, t_min, t_max, step, Center_of_Reference = argv
a_list = np.array(a_list.split()); t_min = float(t_min); t_max = float(t_max); step = int(step)
print(a_list)
# trj_file = .trr/.xtc file, top_file = .tpr file
# a_list = list of atom names, e.g., 'LI CL'
#     NOTE: Coded for up to 4 different atoms (minimum of 2), which can characterize a 5-component system
# t_min = start time for analysis in ps (-1 assumes the start time of the first frame)
# t_max = end time for analysis in ps (-1 assumes the end time of the last frame)
# step = frame step-size (-1 assumes a step-size of 1)
# Center_of_Reference (CoR) = 0 for System Center of Mass (CoM), 1 for Polymer Center of Mass (PoM), 2 for Solvent Center of Mass (CoM), 3 for Center of Volume (CoV), 4 if System Center of Mass data was saved using Sys_CoM.py
# NOTE: Make sure to have unwrapped coordinates (gmx trjconv with -pbc nojump)

# Example: python3 ${Path}/Onsager_GK.py /tmp/unwrap.xtc md.tpr 'LI K CL' -1 -1 -1 0

# Van der waals radii from Bondi (1964)
# https://www.knowledgedoor.com/2/elements_handbook/bondi_van_der_waals_radius.html
Size_arr = np.array([('H', 1.20), ('C',1.70), ('O',1.52), ('N',1.55), ('S',1.80), ('P',1.80), ('LI', 1.82), ('NA', 2.27), ('K', 2.75), ('MG', 1.73), ('F', 1.47), ('CL', 1.85), ('BR', 1.47), ('I', 1.98), ('MW',0.00)])

def outPrint(filename, dt, msd):  # print msd
    with open('{}.xvg'.format(filename), 'w') as anaout:
        print('# time msd', file=anaout)
        for i in range(0, len(dt)):
            print(' {: 1.5e} {: 1.5e}'.format(dt[i], msd[i]), file=anaout)


def create_mda(top, trj):
    """
    Creates MDAnalysis universe with trajectory data.
    :param top: string, path to Gromacs topology file with atom coordinates and topology
    :param trj: string or list[string], path(s) to Gromacs xtc files with trajectory data
    :return uta: MDAnalysis universe
    """
    uta = mda.Universe(top, trj)
    return uta

# Algorithms in this section are adapted from DOI: 10.1051/sfn/201112010 and
# https://stackoverflow.com/questions/34222272/computing-mean-square-displacement-using-python-and-fft

def autocorrFFT(x):
    """
    Calculates the autocorrelation function using the fast Fourier transform.
    
    :param x: array[float], function on which to compute autocorrelation function
    :return: acf: array[float], autocorrelation function
    """
    N = len(x)
    F = np.fft.fft(x, n = 2 * N)  
    PSD = F * F.conjugate()
    res = np.fft.ifft(PSD)
    res = (res[:N]).real   
    n = N * np.ones(N) - np.arange(0, N) 
    acf = res / n
    return acf


def msd_fft(r):
    """
    Computes mean square displacement using the fast Fourier transform.
    
    :param r: array[float], atom positions over time
    :return: msd: array[float], mean-squared displacement over time
    """
    N = len(r)
    D = np.square(r).sum(axis = 1) 
    D = np.append(D, 0) 
    S2 = sum([autocorrFFT(r[:, i]) for i in range(r.shape[1])])
    Q = 2 * D.sum()
    S1 = np.zeros(N)
    for m in range(N):
        Q = Q - D[m - 1] - D[N - m]
        S1[m] = Q / (N - m)
    msd = S1 - 2 * S2
    return msd

def cross_corr(x, y):
    """
    Calculates cross-correlation function of x and y using the 
    fast Fourier transform.
    :param x: array[float], data set 1
    :param y: array[float], data set 2
    :return: cf: array[float], cross-correlation function
    """
    N = len(x)
    F1 = np.fft.fft(x, n = 2**(N * 2 - 1).bit_length())
    F2 = np.fft.fft(y, n = 2**(N * 2 - 1).bit_length())
    PSD = F1 * F2.conjugate()
    res = np.fft.ifft(PSD)
    res = (res[:N]).real   
    n = N * np.ones(N) - np.arange(0, N)
    cf = res / n
    return cf

def msd_fft_cross(r, k):
    """
    Calculates "MSD" (cross-correlations) using the fast Fourier transform.
    :param r: array[float], positions of atom type 1 over time
    :param k: array[float], positions of atom type 2 over time
    :return: msd: array[float], "MSD" over time
    """
    N = len(r)
    D = np.multiply(r,k).sum(axis=1) 
    D = np.append(D,0) 
    S2 = sum([cross_corr(r[:, i], k[:,i]) for i in range(r.shape[1])])
    S3 = sum([cross_corr(k[:, i], r[:,i]) for i in range(k.shape[1])])
    Q = 2 * D.sum()
    S1 = np.zeros(N)
    for m in range(N):
        Q = Q - D[m - 1] - D[N - m]
        S1[m] = Q / (N - m)
    msd = S1 - S2 - S3
    return msd

def calc_Lii_self(atom_positions, times):
    """ 
    Calculates the "MSD" for the self component for a diagonal transport coefficient (L^{ii}).
    :param atom_positions: array[float,float,float], position of each atom over time.
    Indices correspond to time, ion index, and spatial dimension (x,y,z), respectively.
    :param times: array[float], times at which position data was collected in the simulation
    :return msd: array[float], "MSD" corresponding to the L^{ii}_{self} transport 
    coefficient at each time
    """
    Lii_self = np.zeros(len(times))
    n_atoms = np.shape(atom_positions)[1]
    for atom_num in (range(n_atoms)):
        r = atom_positions[:, atom_num, :]
        msd_temp = msd_fft(np.array(r))
        Lii_self += msd_temp
    msd = np.array(Lii_self)
    return msd


def calc_Lii(atom_positions, times):
    """ 
    Calculates the "MSD" for the diagonal transport coefficient L^{ii}. 
    :param atom_positions: array[float,float,float], position of each atom over time.
    Indices correspond to time, ion index, and spatial dimension (x,y,z), respectively.
    :param times: array[float], times at which position data was collected in the simulation
    :return msd: array[float], "MSD" corresponding to the L^{ii} transport 
    coefficient at each time
    """
    r_sum = np.sum(atom_positions, axis = 1)
    msd = msd_fft(r_sum)
    return np.array(msd)


def compute_all_Lij(a1_positions, a2_positions, a3_positions, a4_positions, times):
    """
    Computes the "MSDs" for all transport coefficients.
    :param a1_positions, a2_positions: array[float,float,float], position of each 
    atom (a2 or a1, respectively) over time. Indices correspond to time, ion index,
    and spatial dimension (x,y,z), respectively.
    :param times: array[float], times at which position data was collected in the simulation
    :param volume: float, volume of simulation box
    :return msds_all: list[array[float]], the "MSDs" corresponding to each transport coefficient,
    L^{++}, L^{++}_{self}, L^{--}, L^{--}_{self}, L^{+-}
    """
    print("MSD Self")
    msd_self_a1 = calc_Lii_self(a1_positions, times)
    msd_self_a2 =  calc_Lii_self(a2_positions, times)
    if len(a_list) >= 3:
        msd_self_a3 =  calc_Lii_self(a3_positions, times)
    else:
        msd_self_a3 = np.zeros_like(msd_self_a1)
    if len(a_list) == 4:
        msd_self_a4 =  calc_Lii_self(a4_positions, times)
    else:
        msd_self_a4 = np.zeros_like(msd_self_a1)

    print("MSD Distinct")
    msd_a1 = calc_Lii(a1_positions, times)
    msd_a2 = calc_Lii(a2_positions, times)
    if len(a_list) >= 3:
        msd_a3 = calc_Lii(a3_positions, times)
    else:
        msd_a3 = np.zeros_like(msd_a1)
    if len(a_list) == 4:
        msd_a4 = calc_Lii(a4_positions, times)
    else:
        msd_a4 = np.zeros_like(msd_a1)

    print("MSD Cross")
    msd_cross_a1a2 = calc_Lij(a1_positions, a2_positions, times)
    if len(a_list) >= 3:
        msd_cross_a1a3 = calc_Lij(a1_positions, a3_positions, times)
        msd_cross_a2a3 = calc_Lij(a2_positions, a3_positions, times)
    else:
        msd_cross_a1a3 = np.zeros_like(msd_cross_a1a2)
        msd_cross_a2a3 = np.zeros_like(msd_cross_a1a2)
    if len(a_list) == 4:
        msd_cross_a1a4 = calc_Lij(a1_positions, a4_positions, times)
        msd_cross_a2a4 = calc_Lij(a2_positions, a4_positions, times)
        msd_cross_a3a4 = calc_Lij(a3_positions, a4_positions, times)
    else:
        msd_cross_a1a4 = np.zeros_like(msd_cross_a1a2)
        msd_cross_a2a4 = np.zeros_like(msd_cross_a1a2)
        msd_cross_a3a4 = np.zeros_like(msd_cross_a1a2)

    msds_all = [msd_a1, msd_self_a1, msd_a2, msd_self_a2, msd_a3, msd_self_a3, msd_a4, msd_self_a4, msd_cross_a1a2, msd_cross_a1a3, msd_cross_a1a4, msd_cross_a2a3, msd_cross_a2a4, msd_cross_a3a4]
    return msds_all


def calc_Lij(a1_positions, a2_positions, times):
    """
    Calculates the "MSD" for the off-diagonal transport coefficient L^{ij}, i \neq j.
    :param a1_positions, a2_positions: array[float,float,float], position of each 
    atom (a2 or a1, respectively) over time. Indices correspond to time, ion index,
    and spatial dimension (x,y,z), respectively.
    :param times: array[float], times at which position data was collected in the simulation
    :return msd: array[float], "MSD" corresponding to the L^{ij} transport coefficient at 
    each time.
    """
    r_a1 = np.sum(a1_positions, axis = 1)
    r_a2 = np.sum(a2_positions, axis = 1)
    msd = msd_fft_cross(np.array(r_a1), np.array(r_a2))
    return np.array(msd)


def main(trj_file, top_file, a_list, t_min, t_max, step, Center_of_Reference):

    # Load trajectory
    uta = create_mda(top_file, trj_file)

    # Retrieve array of frame ids
    if t_min == -1:
        t_min = uta.trajectory[0].time
    if t_max == -1:
        t_max = uta.trajectory[-1].time
    if step < 1:
        step = 1
    dt = np.round((uta.trajectory[1].time - uta.trajectory[0].time),2)
    frame_ids = np.arange(int((t_min - uta.trajectory[0].time)/dt), int((t_max - uta.trajectory[0].time)/dt + 1), step)
    dt = step*dt
    print("Timestep " + str(dt))
    times = frame_ids*dt

    # Load in CoM data, if applicable
    if Center_of_Reference == '4':
        with h5py.File('CoM.hdf5','r') as f:
            times = f['times'][:]
            CoM_ar = f['CoM'][:,:]

    # Match atoms to van der Waals radii, if applicable
    if Center_of_Reference == '3':
        # Create an array that tracks the radius of each system atom based on Size_array
        sys_names = uta.select_atoms("all").names; sys_radii = []
        for name in sys_names:
            name = str(name)
            if name in Size_arr[:,0]:
                sys_radii.append(float(Size_arr[np.where(Size_arr[:,0] == name)[0][0],1]))
            elif name[0] in Size_arr[:,0]:
                sys_radii.append(float(Size_arr[np.where(Size_arr[:,0] == name[0])[0][0],1]))
            else:
                print("Missing Atom Name and Size in Size_arr (See Atom Name Below)")
                print(name)
                exit()
        sys_radii = np.array(sys_radii)
    
    # Define atom types for analysis
    a1 = uta.select_atoms("name " + a_list[0]); a1_positions = np.zeros((len(frame_ids), len(a1), 3))
    a2 = uta.select_atoms("name " + a_list[1]); a2_positions = np.zeros((len(frame_ids), len(a2), 3))
    if len(a1) == 0 or len(a2) == 0:
        exit()
    if len(a_list) >= 3:
        a3 = uta.select_atoms("name " + a_list[2]); a3_positions = np.zeros((len(frame_ids), len(a3), 3))
        if len(a3) == 0:
            exit()
    else:
        a3_positions = 0
    if len(a_list) == 4:
        a4 = uta.select_atoms("name " + a_list[3]); a4_positions = np.zeros((len(frame_ids), len(a4), 3))
        if len(a4) == 0:
            exit()
    else:
        a4_positions = 0

    # Retrieve atom positions relative to CoR
    for i, frame in enumerate(frame_ids):
        if frame%5000 == 0:
            print("Frame " + str(frame))

        ts = uta.trajectory[frame]
        cell = ts.dimensions

        if Center_of_Reference == '0':
            CoR = uta.atoms.center_of_mass()
        elif Center_of_Reference == '1':
            CoR = uta.select_atoms("moltype MOL").center_of_mass()
        elif Center_of_Reference == '2':
            CoR = uta.select_atoms("moltype SOL").center_of_mass()
        elif Center_of_Reference == '3':
            CoR = np.sum(np.array(uta.select_atoms("all").positions * (sys_radii[:, np.newaxis]**3)), axis=0)/np.sum(sys_radii**3)
        elif Center_of_Reference == '4':
            CoR = CoM_ar[np.where(times == ts.time)]
        else:
            print('Error in Reference Frame Removal')
            exit()

        a1_positions[i, :, :] = (a1.positions - CoR)/10.0
        a2_positions[i, :, :] = (a2.positions - CoR)/10.0

        if len(a_list) >= 3:
            a3_positions[i, :, :] = (a3.positions - CoR)/10.0

        if len(a_list) == 4:
            a4_positions[i, :, :] = (a4.positions - CoR)/10.0

    # Compute Green-Kubo "MSDs"
    print('Compute Lijs')
    msds_all = compute_all_Lij(a1_positions, a2_positions, a3_positions, a4_positions, times)
    
    # Print data
    outPrint('msd_self_'+a_list[0], times, msds_all[1])
    outPrint('msd_dis_'+a_list[0], times, msds_all[0] - msds_all[1])
    outPrint('msd_all_'+a_list[0], times, msds_all[0])
    
    outPrint('msd_self_'+a_list[1], times, msds_all[3])
    outPrint('msd_dis_'+a_list[1], times, msds_all[2] - msds_all[3])
    outPrint('msd_all_'+a_list[1], times, msds_all[2])

    outPrint('msd_cross_'+a_list[0]+'_'+a_list[1], times, msds_all[8])    

    if len(a_list) >= 3:
        outPrint('msd_self_'+a_list[2], times, msds_all[5])
        outPrint('msd_dis_'+a_list[2], times, msds_all[4] - msds_all[5])
        outPrint('msd_all_'+a_list[2], times, msds_all[4])

        outPrint('msd_cross_'+a_list[0]+'_'+a_list[2], times, msds_all[9])
        outPrint('msd_cross_'+a_list[1]+'_'+a_list[2], times, msds_all[11])
    
    if len(a_list) == 4:
        outPrint('msd_self_'+a_list[3], times, msds_all[7])
        outPrint('msd_dis_'+a_list[3], times, msds_all[6] - msds_all[7])
        outPrint('msd_all_'+a_list[3], times, msds_all[6])

        outPrint('msd_cross_'+a_list[0]+'_'+a_list[3], times, msds_all[10])
        outPrint('msd_cross_'+a_list[1]+'_'+a_list[3], times, msds_all[12])
        outPrint('msd_cross_'+a_list[2]+'_'+a_list[3], times, msds_all[13])

if __name__ == "__main__":
    main(trj_file, top_file, a_list, t_min, t_max, step, Center_of_Reference)
