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

# This script creates a HDF5 I/O file containing MSD and a1-aX coordination data as a function time.
# This file is meant to be used to create coordination vs time plots using PLOT_coordination.py

import MDAnalysis as mda
import MDAnalysis.analysis.distances as dist
import MDAnalysis.lib.distances as distances
import numpy as np
import networkx as nx

import multiprocessing as mp
import functools
import h5py
import os
from sys import argv

script, trj_file, top_file, a1_name, aX_list, coord_list, t_min, t_max, step, nt = argv
aX_list = np.array(aX_list.split(), dtype = str); coord_list = np.array(coord_list.split(), dtype = float); t_min = float(t_min); t_max = float(t_max); step = int(step); nt = int(nt)
# trj_file = .trr/.xtc file, top_file = .tpr file
# a1_name = name of reference atom to be analyzed
# aX_list = list of selection atoms to be analyzed
# coord_list = list of coordination distance between a1 and aX (Angstroms) - if a single distance is provided, it will be applied to all a1-aX coordinations
# t_min = start time for analysis in ps (-1 assumes the start time of the first frame)
# t_max = end time for analysis in ps (-1 assumes the end time of the last frame)
# step = frame step-size (-1 assumes a step-size of 1)
# nt = number of threads

# Example: python3 ${Path}/dynamic_coorindation.py md.xtc md.tpr LI 'OT OI' '2.75' -1 -1 -1 128



def Coordination(frame):
# finds all a1-aX pairs
#
# Inputs: frame = time frame to be analyzed

    if frame%5000 == 0:
        print("Frame " + str(frame))

    rX = {}
    with h5py.File('/tmp/Coordination.hdf5','r') as f:
        dset1 = f['r1']; a1 = dset1[frame]
        for a in aX_list:
            dset2 = f[a]; rX[a] = dset2[frame]
        dset3 = f['cells']; cell = dset3[frame]
    
    pair = np.zeros((len(a1),15), dtype = int) - 1; count = np.zeros((len(a1)), dtype = int); add = 0
    for i, a in enumerate(aX_list):
        if len(coord_list) == 1:
            coord = coord_list[0]
        else:
            coord = coord_list[i]

        distpair = distances.capped_distance(rX[a], a1, coord, box=cell)[0]

        for index, i in enumerate(distpair[:,0]):
            pair[distpair[index,1],count[distpair[index,1]]] = i + add; count[distpair[index,1]] += 1
        
        add += len(rX[a])

    return pair



def msd_calc(frame):
# calculate the MSD with CoM removed
#
# Inputs: frame = frame to calculate the MSD over

    if frame%5000 == 0:
        print("dFrame "+ str(frame))

    with h5py.File('/tmp/Coordination.hdf5','r') as f:
        dset1 = f['r1_uw']; a1_0 = dset1[0]; a1_f = dset1[frame]
    
    msd = np.sum(np.square(a1_f - a1_0), axis = 1)

    return msd



def load_TRR():
# loads in the trajectory and saves the necessary data to a temporary h5py file

    global t_min, t_max, step

    uta = mda.Universe(top_file, trj_file)

    a1 = uta.select_atoms("name " + a1_name)
    aX = {}; rX = {}; indexes = []
    for a in aX_list:
        aX[a] = uta.select_atoms("name " + a)
        rX[a] = []
        indexes.append([len(aX[a]), int(len(aX[a]) / aX[a].n_residues)])

    if t_min == -1:
        t_min = uta.trajectory[0].time
    if t_max == -1:
        t_max = uta.trajectory[-1].time
    if step < 1:
        step = 1
    dt = np.round((uta.trajectory[1].time - uta.trajectory[0].time),3)
    frame_ids = np.arange(int((t_min - uta.trajectory[0].time)/dt), int((t_max - uta.trajectory[0].time)/dt + 1), step)
    dt = dt*step

    r1 = []; cells = []
    for i, frame in enumerate(frame_ids):

        if frame%5000 == 0:
            print("Frame " + str(frame))
        
        ts = uta.trajectory[frame]
        cell = ts.dimensions
        cells.append(cell)

        r1.append(a1.positions)
        for a in aX_list:
            rX[a].append(aX[a].positions)



    print("Load Unwrapped Trajectory")
    uta = mda.Universe(top_file, '/tmp/unwrap.xtc')

    a1 = uta.select_atoms("name " + a1_name)

    r1_uw = []
    for frame in frame_ids:

        if frame%5000 == 0:
            print("Frame " + str(frame))
        
        ts = uta.trajectory[frame]

        r1_uw.append(a1.positions)

    with h5py.File('/tmp/Coordination.hdf5','w') as f:
        dset1 = f.create_dataset("cells", data = cells)
        dset2 = f.create_dataset("r1", data=r1)
        for a in aX_list:
            dset3 = f.create_dataset(a, data=np.array(rX[a]))
        dset4 = f.create_dataset("frames", data=frame_ids)
        dset5 = f.create_dataset("indexes", data=indexes)
        dset6 = f.create_dataset("r1_uw", data=r1_uw)



def main(trj_file, top_file, a1_name, aX_list, coord_list, t_min, t_max, step, nt):

    # Load in the trajectory file
    if not os.path.exists('/tmp/Coordination.hdf5'):
        if not os.path.exists('/tmp/unwrap.xtc'):
            print("Unwrap Trajectory")
            os.system("printf '0\n' | gmx trjconv -f md.xtc -s md.tpr -o /tmp/unwrap.xtc -pbc nojump")

            print("Load Trajectory")
            load_TRR()
            os.remove('/tmp/unwrap.xtc')
        else:
            print("Load Trajectory")
            load_TRR()
        exit()

    with h5py.File('/tmp/Coordination.hdf5','r') as f:
        dset1 = f['frames']; frame_ids = dset1[:]
        dset2 = f['indexes']; indexes = dset2[:]

    print("Coordination Analysis")
    pool = mp.Pool(processes=nt)
    func = functools.partial(Coordination)
    return_arr = pool.map(func, range(len(frame_ids)))
    pool.close()
    pool.join()
    pair_arr = np.array(return_arr)

    with h5py.File('Dyn_Coord.hdf5','w') as f:
        dset1 = f.create_dataset("pair", data = pair_arr)
        dset2 = f.create_dataset("indexes", data = indexes)
    del pair_arr, indexes, return_arr

    
    
    print("MSD Analysis")
    pool = mp.Pool(processes=nt)
    func = functools.partial(msd_calc)
    return_arr = pool.map(func, range(len(frame_ids)))
    pool.close()
    pool.join()
    msd_arr = np.array(return_arr).T

    with h5py.File('Dyn_Coord.hdf5','a') as f:
        dset3 = f.create_dataset("msd", data = msd_arr)
    
    # Deletes the temporary h5py file
    os.remove('/tmp/Coordination.hdf5')

if __name__ == "__main__":
    main(trj_file, top_file, a1_name, aX_list, coord_list, t_min, t_max, step, nt)
