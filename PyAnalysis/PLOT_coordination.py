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

# This script plots an atoms RMSD and coordination data as a function of time.
# See PLOT_coordination.pdf for an example plot.

#!/usr/bin/python
import itertools
import numpy as np
from math import *
import matplotlib.pyplot as plt
import colormaps
import matplotlib.ticker as mticker
from matplotlib.ticker import FormatStrFormatter
import matplotlib.lines as mlines
import os
import h5py
from sys import argv

script, a1_name, aX_list, mol_list = argv
aX_list = np.array(aX_list.split(), dtype = str); mol_list = np.array(mol_list.split(), dtype = int)
# a1_name = name of reference atom to be analyzed
# aX_list = list of selection atoms to be analyzed
# mol_list = list defining whether coordination should be per-atom (0) or per-molecule (1)

# Example: python3 ${Path}/PLOT_coordination.py LI 'OT OI' '0 1'

###################################################################
	###OPEN XMING ON LAPTOP IF GETTING DISPLAY ERROR###
###################################################################

labels = ['0','0.0.625','0.125','0.167','0.250','0.500']
atom = np.array([1,1,1,1,1,1])
atom *= 1

for lab_i,lab in enumerate(labels):

    with h5py.File('phi_'+lab+'/sample_1/Dyn_Coord.hdf5','r') as f:
        dset1 = f['pair']; a1 = dset1[:,atom[lab_i],:]
        dset2 = f['indexes']; indexes = dset2[:]
        dset3 = f['msd']; a1_rmsd = np.sqrt(dset3[atom[lab_i]]/10/10) # nm
    print('phi_'+lab)





    # aX_coord tracks the index of aX coordinated to a1 at each frame
    # aX_res tracks the residue index for each atom aX
    aX_coord = {}; aX_res = {}
    for i, a in enumerate(aX_list):
        aX_coord[a] = np.zeros((len(a1),10), dtype=int) - 1
        aX_res[a] = np.zeros((indexes[i,0]), dtype=int)
    
    # fill aX_coord
    c_aX = {} # counter
    for i, a1_i in enumerate(a1):
        for a in aX_list:
            c_aX[a] = 0

        for j, a1_j in enumerate(a1_i):
            if a1_j == -1:
                break

            track = 0
            for k, a in enumerate(aX_list):
                track += indexes[k,0]
                if a1_j < track:
                    if mol_list[k] == 0:
                        aX_coord[a][i,c_aX[a]] = a1_j - (track - indexes[k,0]); c_aX[a] += 1
                    else:
                        aX_coord[a][i,c_aX[a]] = int((a1_j - (track - indexes[k,0])) / indexes[k,1]); c_aX[a] += 1
                    break
    


    # Fill aX_res
    for j, a in enumerate(aX_list):
        if mol_list[j] == 0:
            c = 0
            for i in range(len(aX_res[a])):
                aX_res[a][i] = c
                if (i+1) % (indexes[j,1]) == 0:
                    c += 1
    


    #Define coordination as being coordinated for 50 ps out of each 100 ps interval (25 of every 50 frames)
    aX_coord_new = {}
    for a in aX_list:
        aX_coord_new[a] = np.zeros((int(len(a1)/50),10), dtype=int) - 1; c_new = 0
        for i in range(50, len(aX_coord[a]), 50):
            aX_temp = aX_coord[a][i-50:i]; c = 0
            for j in np.unique(aX_temp):
                if j == -1:
                    continue
                if len(aX_temp[aX_temp == j]) >= 25:
                    aX_coord_new[a][c_new, c] = j
                    c += 1
            c_new += 1
        aX_coord[a] = aX_coord_new[a]



    #Remove un-coordinated atoms/molecules from indexing
    aX_dres = {} # Defines the atom indexes where the residue changes
    aX_coord_x = {}; aX_coord_y = {} # Defines the x and y axis for creating the scatterplots
    for l, a in enumerate(aX_list):
        c = 0; rm = []
        for i in range(indexes[l,0]):
            n = indexes[l,0] - i - 1
            if (c != 0) and (n in aX_coord[a]):
                if mol_list[l] == 0:
                    aX_res[a] = np.delete(aX_res[a], np.array(rm))
                j = np.where(aX_coord[a] > n)[0]; k = np.where(aX_coord[a] > n)[1]
                for x,y in enumerate(j):
                    aX_coord[a][y,k[x]] -= c
                c = 0; rm = []
                continue
            c += 1; rm.append(n)
        if (c != 0):
            if mol_list[l] == 0:
                aX_res[a] = np.delete(aX_res[a], np.array(rm))
            j = np.where(aX_coord[a] > n)[0]; k = np.where(aX_coord[a] > n)[1]
            for x,y in enumerate(j):
                aX_coord[a][y,k[x]] -= c
            c = 0; rm = []

        #Define break between molecule indexes
        if mol_list[l] == 0:
            aX_dres[a] = np.diff(aX_res[a])

        #Prepare data for scatterplot
        count = np.zeros((aX_coord[a].shape[1])); aX_coord_x[a] = np.array([]); aX_coord_y[a] = np.array([])
        for i in aX_coord[a]:
            aX_coord_x[a] = np.hstack((aX_coord_x[a], count[:len(i[i != -1])])); aX_coord_y[a] = np.hstack((aX_coord_y[a], i[i != -1]))
            count += (100/1000) # Plots every 100 ps (0.1 ns)
    







    fig, ax = plt.subplots(len(aX_list) + 1, 1, sharex=True)
    fig.subplots_adjust(hspace=0.1)

    #Plot MSDs
    ax[0].plot(np.arange(0, len(a1_rmsd)*2/1000, 2/1000), a1_rmsd, linewidth=0.50, color='k')

    ax[0].set_ylim(ymin=0)
    ax[0].set_xlim(0,len(a1_rmsd)*2/1000)
    ax[0].set_ylabel(r'RMSD $\mathrm{(nm)}$',fontsize=15,rotation='vertical')
    ax[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    for k, a in enumerate(aX_list):
        # Create Scatterplot
        ax[k+1].scatter(aX_coord_x[a],aX_coord_y[a], color = 'k', marker = '|', sizes=np.zeros_like(aX_coord_y[a])+5)

        #Plot dashed lines separating molecules
        if mol_list[k] == 0:
            for i,j in enumerate(aX_res[a]):
                if i == 0:
                    continue
                if aX_dres[a][i-1] != 0:
                    ax[k+1].plot([0,np.amax(aX_coord_x[a])],[i-0.5,i-0.5], color='k', linewidth=0.5, linestyle='--')

        ax[k+1].set_ylim(-0.5,np.amax(aX_coord_y[a])+0.5)
        locs = ax[k+1].get_yticks(); dloc = int(np.ceil(locs[1] - locs[0]))
        new_yticks = np.arange(0, np.amax(aX_coord_y[a])+1, dloc, dtype=int)
        ax[k+1].set_yticks(new_yticks)

        if a == 'OT':
            y_label = r'$\mathrm{Li^+-EO}$'
        elif a == 'OI':
            y_label = r'$\mathrm{Li^+-TFSI^-}$'
        elif a == 'OW':
            y_label = r'$\mathrm{Li^+-H_2O}$'
        else:
            y_label = a
        ax[k+1].set_ylabel(y_label,fontsize=15,rotation='vertical')
    ax[k+1].set_xlabel(r'$t$ (ns)',fontsize=20)

    for i in range(len(aX_list) + 1):
        ax[i].tick_params(axis='x',labelsize=15)
        ax[i].tick_params(axis='y',labelsize=12)
        ax[i].yaxis.set_label_coords(-0.09,0.5)

    fig.subplots_adjust(bottom=0.15)
    fig.subplots_adjust(top=0.95)
    #fig.subplots_adjust(left=0.15)
    #fig.subplots_adjust(right=0.875)
    fig.savefig('O_'+lab+'.pdf')
    plt.close(fig)
