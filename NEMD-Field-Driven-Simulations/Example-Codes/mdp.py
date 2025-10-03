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

# This code creates GROMACS .mdp run files to perform NEMD simulations using the GROMACS accelerate command

import numpy as np
import os

from sys import argv
script, a_list, N_list, drc, EF = argv
a_list = a_list.split(" "); N_list = np.array(N_list.split(" "), dtype=int); EF = EF.split()

# python3 mdp.py 'LI K CL CL SOL' '18 54 72 4000' ../mdp

# Book keeping to decide what type of system this is - binary, ternary, quaternary
a_temp = []; N_temp = []; check = 0
if a_list[0] == a_list[1]:
    a_temp.append(a_list[0]); N_temp.append(N_list[2])
    check = 1
else:
    a_temp.append(a_list[0]); N_temp.append(N_list[0])
    a_temp.append(a_list[1]); N_temp.append(N_list[1])
if a_list[2] == a_list[3]:
    a_temp.append(a_list[2]); N_temp.append(N_list[2])
    if check == 1:
        check = 3
    else:
        check = 2
else:
    a_temp.append(a_list[2]); N_temp.append(N_list[0])
    a_temp.append(a_list[3]); N_temp.append(N_list[1])
a_temp.append(a_list[-1]); N_temp.append(N_list[-1])
a_list = np.array(a_temp); N_list = np.array(N_temp)
print(a_list); print(N_list)

files = np.array(['F.mdp'])

#EF = np.array(['0.02','0.04','0.06','0.08','0.10'])

# List of molecular weights
M_list = np.array([
    ['LI', '6.941'],
    ['NA', '22.98977'],
    ['K', '39.0983'],
    ['RB', '85.46780'],
    ['CS', '132.90540'],
    ['F', '18.998403'],
    ['CL', '35.453'],
    ['BR', '79.904'],
    ['I', '126.90447'],
    ['SOL', '18.0154']]) # g/mol

# Convert electric field strengths to artifical field strengths in J/m, where each ion has the charge of an electron, 1.602E-19 C
F = np.array(EF, dtype=float)*1e9*(1.602E-19) # J/m

# Calculate the acceleration required for each ion and field in the x, y, and z directions in nm/ps/ps
F_arr = np.zeros((max(3,len(a_list) - 1),len(a_list),len(F)))
for i in range(max(3,len(a_list) - 1)):
    for j, b in enumerate(a_list):
        M_j = np.where(M_list[:,0] == b)[0][0]

        if check == 1:
            if i == 0:
                if j == 0 or j == 1 or j == 2:
                    F_arr[i,j] = F / (float(M_list[M_j,1])/1000/6.022e23) * (1000000000/1000000000000/1000000000000) # nm/ps^2
            elif i == 1:
                if j == 0 or j == 2:
                    F_arr[i,j] = F / (float(M_list[M_j,1])/1000/6.022e23) * (1000000000/1000000000000/1000000000000) # nm/ps^2
                elif j == 1:
                    F_arr[i,j] = -F / (float(M_list[M_j,1])/1000/6.022e23) * (1000000000/1000000000000/1000000000000) # nm/ps^2
            elif i == 2:
                if j == 1 or j == 2:
                    F_arr[i,j] = F / (float(M_list[M_j,1])/1000/6.022e23) * (1000000000/1000000000000/1000000000000) # nm/ps^2
                elif j == 0:
                    F_arr[i,j] = -F / (float(M_list[M_j,1])/1000/6.022e23) * (1000000000/1000000000000/1000000000000) # nm/ps^2
            else:
                continue
        elif check == 2:
            if i == 0:
                if j == 0 or j == 1 or j == 2:
                    F_arr[i,j] = F / (float(M_list[M_j,1])/1000/6.022e23) * (1000000000/1000000000000/1000000000000) # nm/ps^2
            elif i == 1:
                if j == 0 or j == 2:
                    F_arr[i,j] = F / (float(M_list[M_j,1])/1000/6.022e23) * (1000000000/1000000000000/1000000000000) # nm/ps^2
                elif j == 1:
                    F_arr[i,j] = -F / (float(M_list[M_j,1])/1000/6.022e23) * (1000000000/1000000000000/1000000000000) # nm/ps^2
            elif i == 2:
                if j == 1 or j == 2:
                    F_arr[i,j] = F / (float(M_list[M_j,1])/1000/6.022e23) * (1000000000/1000000000000/1000000000000) # nm/ps^2
                elif j == 0:
                    F_arr[i,j] = -F / (float(M_list[M_j,1])/1000/6.022e23) * (1000000000/1000000000000/1000000000000) # nm/ps^2
            else:
                continue
        elif check == 3:
            if i == 0:
                if j == 0 or j == 1:
                    F_arr[i,j] = F / (float(M_list[M_j,1])/1000/6.022e23) * (1000000000/1000000000000/1000000000000) # nm/ps^2
            elif i == 1:
                if j == 0:
                    F_arr[i,j] = F / (float(M_list[M_j,1])/1000/6.022e23) * (1000000000/1000000000000/1000000000000) # nm/ps^2
                elif j == 1:
                    F_arr[i,j] = -F / (float(M_list[M_j,1])/1000/6.022e23) * (1000000000/1000000000000/1000000000000) # nm/ps^2
            else:
                continue
        else:
            if i == 0:
                if j == 0 or j == 1 or j == 2 or j == 3:
                    F_arr[i,j] = F / (float(M_list[M_j,1])/1000/6.022e23) * (1000000000/1000000000000/1000000000000) # nm/ps^2
            elif i == 1:
                if j == 0 or j == 1:
                    F_arr[i,j] = F / (float(M_list[M_j,1])/1000/6.022e23) * (1000000000/1000000000000/1000000000000) # nm/ps^2
                elif j == 2 or j == 3:
                    F_arr[i,j] = -F / (float(M_list[M_j,1])/1000/6.022e23) * (1000000000/1000000000000/1000000000000) # nm/ps^2
            elif i == 2:
                if j == 0 or j == 2:
                    F_arr[i,j] = F / (float(M_list[M_j,1])/1000/6.022e23) * (1000000000/1000000000000/1000000000000) # nm/ps^2
                elif j == 1 or j == 3:
                    F_arr[i,j] = -F / (float(M_list[M_j,1])/1000/6.022e23) * (1000000000/1000000000000/1000000000000) # nm/ps^2
            elif i == 3:
                if j == 0 or j == 3:
                    F_arr[i,j] = F / (float(M_list[M_j,1])/1000/6.022e23) * (1000000000/1000000000000/1000000000000) # nm/ps^2
                elif j == 1 or j == 2:
                    F_arr[i,j] = -F / (float(M_list[M_j,1])/1000/6.022e23) * (1000000000/1000000000000/1000000000000) # nm/ps^2
            else:
                continue




# Create GROMACS .mdp file with the appropriate "fields" applied using GROMACS accelerate
if check > 0:
    for k in files:
        with open(drc + '/' + k,'r') as f:
            mdp = f.readlines()

        for i, EF_i in enumerate(EF):
            write_file = open('F_' + str(EF_i) + '/' + k, 'w')
            for line in mdp:
                if line == 'acc-grps                 = \n':
                    line = 'acc-grps                 ='
                    for a_i, a in enumerate(a_list):
                        if N_list[a_i] == 0:
                            continue
                        line += ' ' + a
                    line += '\n'
                if line == 'accelerate               = \n':
                    line = 'accelerate               ='
                    for x in range(F_arr[:,:,i].shape[1]):
                        if N_list[x] == 0:
                            continue
                        for y in range(F_arr[:,:,i].shape[0]):
                            line += ' ' + str(F_arr[y,x,i])
                        line += '  '
                    line += '\n'
                if check == 3 and line == 'electric-field-z         = 0 0 0 0\n':
                    line = 'electric-field-z         = {} 0 0 0\n'.format(EF_i)
                write_file.write(line)
            write_file.close()
elif check == 0:
    for k in files:
        with open(drc + '/' + k,'r') as f:
            mdp = f.readlines()
    
        for i, EF_i in enumerate(EF):
            write_file = open('F_' + str(EF_i) + '/F1.mdp', 'w')
            for line in mdp:
                if line == 'acc-grps                 = \n':
                    line = 'acc-grps                 ='
                    for a_i, a in enumerate(a_list):
                        line += ' ' + a
                    line += '\n'
                if line == 'accelerate               = \n':
                    if N_list[0] <= N_list[1]:
                        line = 'accelerate               = {} {} {}  {} {} {}  {} {} {}  {} {} {}  {} {} {}\n'.format(F_arr[0,0,i], F_arr[1,0,i], F_arr[0,0,i], F_arr[0,1,i], F_arr[1,1,i], F_arr[0,1,i], F_arr[0,2,i], F_arr[1,2,i], F_arr[0,2,i], F_arr[0,3,i], F_arr[1,3,i], F_arr[0,3,i], F_arr[0,4,i], F_arr[1,4,i], F_arr[0,4,i])
                    else:
                        line = 'accelerate               = {} {} {}  {} {} {}  {} {} {}  {} {} {}  {} {} {}\n'.format(F_arr[0,0,i], F_arr[1,0,i], F_arr[1,0,i], F_arr[0,1,i], F_arr[1,1,i], F_arr[1,1,i], F_arr[0,2,i], F_arr[1,2,i], F_arr[1,2,i], F_arr[0,3,i], F_arr[1,3,i], F_arr[1,3,i], F_arr[0,4,i], F_arr[1,4,i], F_arr[1,4,i])
                write_file.write(line)
            write_file.close()

            write_file = open('F_' + str(EF_i) + '/F2.mdp', 'w')
            for line in mdp:
                if line == 'acc-grps                 = \n':
                    line = 'acc-grps                 ='
                    for a_i, a in enumerate(a_list):
                        line += ' ' + a
                    line += '\n'
                if line == 'accelerate               = \n':
                    if N_list[2] <= N_list[3]:
                        line = 'accelerate               = {} {} {}  {} {} {}  {} {} {}  {} {} {}  {} {} {}\n'.format(F_arr[2,0,i], F_arr[3,0,i], F_arr[2,0,i], F_arr[2,1,i], F_arr[3,1,i], F_arr[2,1,i], F_arr[2,2,i], F_arr[3,2,i], F_arr[2,2,i], F_arr[2,3,i], F_arr[3,3,i], F_arr[2,3,i], F_arr[2,4,i], F_arr[3,4,i], F_arr[2,4,i])
                    else:
                        line = 'accelerate               = {} {} {}  {} {} {}  {} {} {}  {} {} {}  {} {} {}\n'.format(F_arr[2,0,i], F_arr[3,0,i], F_arr[3,0,i], F_arr[2,1,i], F_arr[3,1,i], F_arr[3,1,i], F_arr[2,2,i], F_arr[3,2,i], F_arr[3,2,i], F_arr[2,3,i], F_arr[3,3,i], F_arr[3,3,i], F_arr[2,4,i], F_arr[3,4,i], F_arr[3,4,i])
                write_file.write(line)
            write_file.close()
else:
    exit()
