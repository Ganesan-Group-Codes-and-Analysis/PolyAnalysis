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

#This script calculates the 1-D Displacement over time (DIS)

import MDAnalysis as mda
import MDAnalysis.lib.distances as distances
import numpy as np
import h5py

import multiprocessing as mp
import functools
import os

from sys import argv
script, trj_file, top_file, a_name, dimension, t_min, t_max, step, nt, Center_of_Reference = argv
dimension = dimension.split(); t_min = float(t_min); t_max = float(t_max); step = int(step); nt = int(nt)
# trj_file = .trr/.xtc file, top_file = .tpr file
# a_name = name of atom to be analyzed
# dimension = string containing axes top calculate displacement in. If a single axis is desired, 'x', 'y', or 'z'. If you want to average over multiple axes, 'x y', 'x z', 'x y z', etc
# t_min = start time for analysis in ps (-1 assumes the start time of the first frame)
# t_max = end time for analysis in ps (-1 assumes the end time of the last frame)
# step = frame step-size (-1 assumes a step-size of 1)
# nt = number of threads
# NOTE: Make sure to have unwrapped coordinates (gmx trjconv with -pbc nojump)
# Center_of_Reference (CoR) = 0 for System Center of Mass (CoM), 1 for Polymer Center of Mass (PoM), 2 for Solvent Center of Mass (CoM), 3 for Center of Volume (CoV), 4 if System Center of Mass data was saved using Sys_CoM.py

# Example: python3 ${Path}/analysis_dis.py unwrap.trr md.tpr NA 'x' -1 -1 -1 128 0

# Van der waals radii from Bondi (1964)
# https://www.knowledgedoor.com/2/elements_handbook/bondi_van_der_waals_radius.html
Size_arr = np.array([('H', 1.20), ('C',1.70), ('O',1.52), ('N',1.55), ('S',1.80), ('P',1.80), ('LI', 1.82), ('NA', 2.27), ('K', 2.75), ('MG', 1.73), ('F', 1.47), ('CL', 1.85), ('BR', 1.47), ('I', 1.98), ('MW',0.00)])

print(dimension)
dim_ar = np.array(['x', 'y', 'z']); dim_temp = []
for i in dimension:
    dim_temp.append(np.where(dim_ar == i)[0][0])
dimension = np.array(dim_temp)
print(dimension)



def dis_analysis(dimension, dt, dt_max):
# calculate the DIS with CoR removed
#
# Inputs: dim = index of the desired dimension, dt = timestep in ps; dt_max = max time to calculate DIS over

    nframe = int(round(dt_max/dt))

    pool = mp.Pool(processes=nt)
    func = functools.partial(dis_calc, dimension, nframe)
    dis = pool.map(func, list(range(nframe)))
    pool.close()
    pool.join()

    dis_bins = np.arange(0,nframe*dt,dt)
    dis = np.array(dis)

    with open('dis_{}_{}.xvg'.format(dim_ar[dimension[0]], a_name), 'w') as anaout:
        print('# Time DIS (nm)', file=anaout)
        for i, bin_i in enumerate(dis_bins):
            if i >= len(dis):
                print('{:10.3f} {:10.5f}'.format(bin_i, 0.0), file=anaout)
            else:
                print('{:10.3f} {:10.5f}'.format(bin_i, dis[i]), file=anaout)



def dis_calc(dimension, nframe, df):
# calculate the DIS with CoR removed
#
# Inputs: dim = index of the desired dimension, nframe = total number of frames, df = frame step to calculate the DIS over

    if df == 0:
        return 0.0

    if df%5000 == 0:
        print("dFrame "+ str(df))

    DIS_file = h5py.File('/tmp/r_DIS_'+a_name+'.hdf5','r'); r = DIS_file['r'][:,:,dimension]

    dis = 0.0; count = 0
    for j in range(0, nframe-1, 1):# Statistical enchancement by calculating over every frame
        if df + j >= nframe:
            break
            
        dis += np.mean(r[j+df] - r[j])
        count += 1

    DIS_file.close()

    return dis / count



def load_TRR():
# Load in the trajectory file and write necessary data to a h5py file
# For very large files, it is recommended to dump only the atoms you are interested in to a separate .trr/.xtc file for analysis
# NOTE: Must supply with unwrapped coordinates

    global t_min, t_max, step

    # Load trajectory
    uta = mda.Universe(top_file, trj_file, tpr_resid_from_one=True)

    # Define atom types for analysis
    a = uta.select_atoms("name " + a_name)
    if len(a) == 0:
        exit()

    # Retrieve array of frame ids
    if t_min == -1:
        t_min = uta.trajectory[0].time
    if t_max == -1:
        t_max = uta.trajectory[-1].time
    if step < 1:
        step = 1
    dt = np.round((uta.trajectory[1].time - uta.trajectory[0].time),3)
    frame_ids = np.arange(int((t_min - uta.trajectory[0].time)/dt), int((t_max - uta.trajectory[0].time)/dt + 1), step)
    dt = dt*step
    print("Timestep " + str(dt))

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
    
    # Retrieve atom positions relative to CoR
    r = []
    for frame in frame_ids:
        ts = uta.trajectory[frame]
        cell = ts.dimensions

        if Center_of_Reference == '0':
            CoR = uta.atoms.center_of_mass()
        elif Center_of_Reference == '1':
            CoR = uta.select_atoms("moltype MOL").center_of_mass()
        elif Center_of_Reference == '2':
            CoR = uta.select_atoms("resname SOL").center_of_mass()
        elif Center_of_Reference == '3':
            CoR = np.sum(np.array(uta.select_atoms("all").positions * (sys_radii[:, np.newaxis]**3)), axis=0)/np.sum(sys_radii**3)
        elif Center_of_Reference == '4':
            CoR = CoM_ar[np.where(times == ts.time)]
        else:
            print('Error in Reference Frame Removal')
            exit()
    
        if ts.time%5000 == 0:
            print("Time "+ str(ts.time))

        r.append((a.positions - CoR)/10)

    # Save data in I/O file
    with h5py.File('/tmp/r_DIS_'+a_name+'.hdf5','w') as f:
        dset1 = f.create_dataset("r", data=r)
        dset2 = f.create_dataset("dt", data=[dt])
        dset3 = f.create_dataset("frames", data=frame_ids)



def main(trj_file, top_file, a_name, dimension, t_min, t_max, step, nt, Center_of_Reference):

    # If there is not a position h5py file, then create one and end the program
    # This is done to avoid memory problems during multiprocessing
    if not os.path.exists('/tmp/r_DIS_'+a_name+'.hdf5'):
        load_TRR()
        exit()
    with h5py.File('/tmp/r_DIS_'+a_name+'.hdf5','r') as f:
        dset1 = f['dt']; dt = dset1[0]
        dset2 = f['frames']; frame_ids = dset2[:]
   
    print("DIS Analysis")
    dt_max = len(frame_ids)*dt# as written dt_max is the total time frame
    #dt_max=10000
    dis_analysis(dimension, dt, dt_max)

    os.remove('/tmp/r_DIS_'+a_name+'.hdf5')



if __name__ == "__main__":
    main(trj_file, top_file, a_name, dimension, t_min, t_max, step, nt, Center_of_Reference)
