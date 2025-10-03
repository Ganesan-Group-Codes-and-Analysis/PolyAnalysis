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

# This script calculates the average and standard deviation of the drift velocity over X samples and Y field strengths

from scipy import stats
from scipy import signal
from scipy.linalg import lstsq
from scipy.optimize import curve_fit
import numpy as np
from sys import argv
import os
import math
import numbers
import h5py

script, filenames, columns, runs, labels, F_labels, beg_list, end_list= argv
filenames = filenames.split(); columns = columns.split(); runs = int(runs); labels = labels.split(); F_labels = F_labels.split(); beg_list = beg_list.split(', '); end_list = end_list.split(', ')
 # filenames = string of msd files to be analyzed, separated by a ' ': e.g. 'msd_self_NA.xvg msd_self_CL.xvg'
 #             NOTE: This file is meant to analyze msd_self, NOT msd_all, msd_distinct, or msd_cross
 # columns = string of numbers denoting the number of columns in the msd file (not including the time): e.g. '1 1'
 #           NOTE: This will be a string of 1's separated by ' ' nearly always. Never tested with multiple columns of data: e.g. 5
 # runs = number of samples being averaged over. This code is built to work with my file structure and will probably need to be tweaked
 # labels = string of concentrations separated by ' ', used for my file strucutre: e.g. '0.04 0.10 0.20 0.50 1.00'

beg = []; end = []
for i in beg_list:
    beg.append(np.array(i.split(), dtype=int))
beg_list = np.array(beg)
for i in end_list:
    end.append(np.array(i.split(), dtype=int))
end_list = np.array(end)

def rip_data(columns,corr):
 # reads in data from files
 #
 # Inputs: columns = number of columns of data (not including the x axis data); corr = name of the file
    t = []
    dat = [[] for c in range(1,columns+1)]
    with open(corr,'r') as data:
        for raw in data:
            if str(raw[0])=='#' or str(raw[0])=='@':
                continue
            d = raw.strip().split()
            t.append(float(d[0]))
            for c in range(1,columns+1):
                dat[c-1].append(float(d[c]))
    return t, dat

 # used to create the directory to my files based on my file structure
prefix = ''
suffix1 = '/sample_'
suffix2 = '/F_'

dat_save = np.zeros((len(filenames),len(F_labels),runs))
for j, name in enumerate(filenames):
 # Loop over files to be analyzed
    #if j != 2: # Used for bug fixing
    #   continue

    col = int(columns[j])

    print(name)

    for k, lb in enumerate(labels):
     # Loop over concentrations to be analyzed
        #if k != 0: # Used for bug fixing
        #    continue

        beg = beg_list[j]; end = end_list[j]
        if len(beg) != 1:
            beg = beg[k]
        else:
            beg = beg[0]
        if len(end) != 1:
            end = end[k]
        else:
            end = end[0]

        if name[:4] == 'md1/':
            writefile = open(prefix+lb+'/dis_md1_' + name[8:],'w')
        elif name[:4] == 'md2/':
            writefile = open(prefix+lb+'/dis_md2_' + name[8:],'w')
        else:
            writefile = open(prefix+lb+'/dis_'+name[4:],'w')
        writefile.write('# dift velocity (cm/s)\n')

        dis_master = np.zeros((len(F_labels), runs))
        for l, F_lb in enumerate(F_labels):
        # Loop over electric field strengths to be analyzed
            #if l != 0: # Used for bug fixing
            #    continue

            dat_master = []
            for i in range(1,runs+1):
             # Loop over samples to be analyzed
                #if i != 1: # Used for bug fixing
                #    continue
                directory = prefix+lb+suffix2+F_lb+suffix1+str(i)+'/'+name

                if os.path.isfile(directory): # If file exists, do analysis, else print the directory
                    t, dat = rip_data(col, directory)
                    t = np.array(t); dat = np.array(dat[0])
                     # Rip data into the x-axis (time, t) and the y axis (MSD, dat)

                    dat_master.append(dat)

                    t_temp = np.array(t[np.where(t==beg)[0][0]:np.where(t==end)[0][0]] - t[np.where(t==beg)[0][0]])
                    dat_temp = np.array(dat[np.where(t==beg)[0][0]:np.where(t==end)[0][0]] - dat[np.where(t==beg)[0][0]])

                    ln = stats.linregress(t_temp,dat_temp)
                    dis_master[l,i-1] = ln.slope # nm/ps
                else:
                    print(directory)
            dat_master = np.array(dat_master)

            #print(beg)
            #print(end)

            dat = np.mean(dat_master, axis = 0); err = np.std(dat_master, axis = 0)

            t_temp = np.array(t[np.where(t==beg)[0][0]:np.where(t==end)[0][0]] - t[np.where(t==beg)[0][0]])
            dat_temp = np.array(dat[np.where(t==beg)[0][0]:np.where(t==end)[0][0]] - dat[np.where(t==beg)[0][0]])

            ln = stats.linregress(np.log(t[np.where(t==beg)[0][0]:np.where(t==end)[0][0]]),np.log(np.abs(dat[np.where(t==beg)[0][0]:np.where(t==end)[0][0]])))
            print('Slope {}     R^2 {}'.format(ln.slope, ln.rvalue*ln.rvalue))

            #ln = stats.linregress(np.log(t_temp[np.where(dat_temp == 0)[0][-1] + 1:]),np.log(np.abs(dat_temp[np.where(dat_temp == 0)[0][-1] + 1:])))
            #print('Slope {}     R^2 {}'.format(ln.slope, ln.rvalue*ln.rvalue))
            ln = stats.linregress(t_temp,dat_temp) # slope -> nm/ps
            dis_best = ln.slope*100000 # cm/s

            for i in range(1,runs+1):
                #dat_temp = np.array(np.log(dat_master[i-1,np.where(t==beg)[0][0]:np.where(t==end)[0][0]]))
                #ln = stats.linregress(np.log(t[np.where(t==beg)[0][0]:np.where(t==end)[0][0]]),dat_temp)
                #slope_save=ln.slope
                #print(i, slope_save)

                dat_temp = np.array(dat_master[i-1,np.where(t==beg)[0][0]:np.where(t==end)[0][0]] - dat_master[i-1,np.where(t==beg)[0][0]])
                ln = stats.linregress(t_temp,dat_temp) # slope -> nm/ps
                dat_save[j,l,i-1] = ln.slope*100000 # cm/s
                #print(ln.slope**100000)
            std_best = np.std(dat_save[j,l,:])
            #print(np.mean(dat_save[j,k,:]), np.std(dat_save[j,k,:]))

            print('\t{:s},{:s}   {:1.3e}   {:1.3e}'.format(lb, F_lb, dis_best, std_best))
            writefile.write(str(F_lb) + ' ' + str(dis_best) + ' ' + str(std_best) + '\n')

        writefile.close()

with h5py.File(lb + '/dis.hdf5','w') as f:
    dset1 = f.create_dataset("dis", data=dat_save)
    dset2 = f.create_dataset("F", data=F_labels)
