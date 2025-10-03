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

# This code calculates the Onsager coefficients from the drift velocity of all ions due to a range of applied field strengths

import numpy as np
from scipy import stats
from sys import argv
import os
import h5py

from scipy.linalg import lstsq

script, a_list, N_list, runs = argv
a_list = a_list.split(" "); N_list = np.array(N_list.split(" "), dtype=int); runs = int(runs)

# python3 Onsager.py 'LI K CL CL SOL' '18 54 72 4000' 5

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
print(a_list); print(N_list); print(check)

# Constants
Faraday = 1.602176634E-19 # C
kB = 1.380e-23 # J/K
T = 298 # K
kBT = kB*T # J (kg m^2/s^2)

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

# Read in drift-velocity data
with h5py.File('dis.hdf5','r') as f:
    dset1 = f['dis']; dat_save = dset1[:] # cm/s
    dset2 = f['F']; F = dset2[:] # V/nm

# Define drift velocity of each ion in each applied field direction
F = np.array(F, dtype=float) # V/nm
if check == 1:
    if N_list[1] == 0:
        x_a1 = dat_save[0]; x_a3 = dat_save[1] # cm/s
        z_a1 = dat_save[2]; z_a3 = dat_save[3] # cm/s
    elif N_list[2] == 0:
        x_a1 = dat_save[0]; x_a2 = dat_save[1] # cm/s
        y_a1 = dat_save[2]; y_a2 = dat_save[3] # cm/s
    else:
        x_a1 = dat_save[0]; x_a2 = dat_save[1]; x_a3 = dat_save[2] # cm/s
        y_a1 = dat_save[3]; y_a2 = dat_save[4]; y_a3 = dat_save[5] # cm/s
        z_a1 = dat_save[6]; z_a2 = dat_save[7]; z_a3 = dat_save[8] # cm/s
elif check == 2:
    if N_list[0] == 0:
        y_a2 = dat_save[0]; y_a3 = dat_save[1] # cm/s
        z_a2 = dat_save[2]; z_a3 = dat_save[3] # cm/s
    elif N_list[1] == 0:
        x_a1 = dat_save[0]; x_a3 = dat_save[1] # cm/s
        z_a1 = dat_save[2]; z_a3 = dat_save[3] # cm/s
    else:
        x_a1 = dat_save[0]; x_a2 = dat_save[1]; x_a3 = dat_save[2] # cm/s
        y_a1 = dat_save[3]; y_a2 = dat_save[4]; y_a3 = dat_save[5] # cm/s
        z_a1 = dat_save[6]; z_a2 = dat_save[7]; z_a3 = dat_save[8] # cm/s
elif check == 3:
    x_a1 = dat_save[0]; x_a2 = dat_save[1] # cm/s
    y_a1 = dat_save[2]; y_a2 = dat_save[3] # cm/s
    z_a1 = dat_save[4]; z_a2 = dat_save[5] # cm/s -> EF
else:
    cx_a1 = dat_save[0];  cx_a2 = dat_save[1];  cx_a3 = dat_save[2];  cx_a4 = dat_save[3] # cm/s
    cy_a1 = dat_save[4];  cy_a2 = dat_save[5];  cy_a3 = dat_save[6];  cy_a4 = dat_save[7] # cm/s
    ax_a1 = dat_save[8];  ax_a2 = dat_save[9];  ax_a3 = dat_save[10]; ax_a4 = dat_save[11] # cm/s
    ay_a1 = dat_save[12]; ay_a2 = dat_save[13]; ay_a3 = dat_save[14]; ay_a4 = dat_save[15] # cm/s

    
# Convert electric field strengths to artifical field strengths in J/m, where each ion has the charge of an electron, 1.602E-19 C
F = (F*1e9)*(Faraday) # J/m

# Define the field-strength array, i.e., in each direction x, y, z, is the field positive or negative for each ion
F_arr = np.zeros((max(3,len(a_list) - 1),len(a_list),len(F)))
for i in range(max(3,len(a_list) - 1)):
    for j, b in enumerate(a_list):
        M_j = np.where(M_list[:,0] == b)[0][0]

        if check == 1:
            if i == 0:
                if j == 0 or j == 1 or j == 2:
                    F_arr[i,j] = F # J/m
            elif i == 1:
                if j == 0 or j == 2:
                    F_arr[i,j] = F # J/m
                elif j == 1:
                    F_arr[i,j] = -F # J/m
            elif i == 2:
                if j == 1 or j == 2:
                    F_arr[i,j] = F # J/m
                elif j == 0:
                    F_arr[i,j] = -F # J/m
            else:
                continue
        elif check == 2:
            if i == 0:
                if j == 0 or j == 1 or j == 2:
                    F_arr[i,j] = F # J/m
            elif i == 1:
                if j == 0 or j == 2:
                    F_arr[i,j] = F # J/m
                elif j == 1:
                    F_arr[i,j] = -F # J/m
            elif i == 2:
                if j == 1 or j == 2:
                    F_arr[i,j] = F # J/m
                elif j == 0:
                    F_arr[i,j] = -F # J/m
            else:
                continue
        elif check == 3:
            if i == 0:
                if j == 0 or j == 1:
                    F_arr[i,j] = F # J/m
            elif i == 1:
                if j == 0:
                    F_arr[i,j] = F # J/m
                elif j == 1:
                    F_arr[i,j] = -F # J/m
            else:
                continue
        else:
            if i == 0:
                if j == 0 or j == 1 or j == 2 or j == 3:
                    F_arr[i,j] = F # J/m
            elif i == 1:
                if j == 0 or j == 1:
                    F_arr[i,j] = F # J/m
                elif j == 2 or j == 3:
                    F_arr[i,j] = -F # J/m
            elif i == 2:
                if j == 0 or j == 2:
                    F_arr[i,j] = F # J/m
                elif j == 1 or j == 3:
                    F_arr[i,j] = -F # J/m
            elif i == 3:
                if j == 0 or j == 3:
                    F_arr[i,j] = F # J/m
                elif j == 1 or j == 2:
                    F_arr[i,j] = -F # J/m
            else:
                continue


# Get average system volume
Vol_arr = []
for r_i in range(runs):
    samp = r_i + 1
    #with open('sample_' + str(samp) + '/md.gro') as f:
    with open('sample_' + str(samp) + '/md.gro') as f:
        V = (float(str(f.readlines()[-1]).split()[-1])**3)*(1e-27) # m^3
    Vol_arr.append(V)
Vol_arr = np.array(Vol_arr); Vol = np.mean(Vol_arr)



# Convert drift velocities into ion fluxes
if check == 1:
    if N_list[1] == 0:
        x_a1 *= N_list[0]/Vol/100; x_a3 *= N_list[2]/Vol/100 # 1/m^2/s
        z_a1 *= N_list[0]/Vol/100; z_a3 *= N_list[2]/Vol/100 # 1/m^2/s
    elif N_list[2] == 0:
        x_a1 *= N_list[0]/Vol/100; x_a2 *= N_list[1]/Vol/100 # 1/m^2/s
        y_a1 *= N_list[0]/Vol/100; y_a2 *= N_list[1]/Vol/100 # 1/m^2/s
    else:
        x_a1 *= N_list[0]/Vol/100; x_a2 *= N_list[1]/Vol/100; x_a3 *= N_list[2]/Vol/100 # 1/m^2/s
        y_a1 *= N_list[0]/Vol/100; y_a2 *= N_list[1]/Vol/100; y_a3 *= N_list[2]/Vol/100 # 1/m^2/s
        z_a1 *= N_list[0]/Vol/100; z_a2 *= N_list[1]/Vol/100; z_a3 *= N_list[2]/Vol/100 # 1/m^2/s
elif check == 2:
    if N_list[0] == 0:
        y_a2 *= N_list[1]/Vol/100; y_a3 *= N_list[2]/Vol/100 # 1/m^2/s
        z_a2 *= N_list[1]/Vol/100; z_a3 *= N_list[2]/Vol/100 # 1/m^2/s
    elif N_list[1] == 0:
        x_a1 *= N_list[0]/Vol/100; x_a3 *= N_list[2]/Vol/100 # 1/m^2/s
        z_a1 *= N_list[0]/Vol/100; z_a3 *= N_list[2]/Vol/100 # 1/m^2/s
    else:
        x_a1 *= N_list[0]/Vol/100; x_a2 *= N_list[1]/Vol/100; x_a3 *= N_list[2]/Vol/100 # 1/m^2/s
        y_a1 *= N_list[0]/Vol/100; y_a2 *= N_list[1]/Vol/100; y_a3 *= N_list[2]/Vol/100 # 1/m^2/s
        z_a1 *= N_list[0]/Vol/100; z_a2 *= N_list[1]/Vol/100; z_a3 *= N_list[2]/Vol/100 # 1/m^2/s
elif check == 3:
    x_a1 *= N_list[0]/Vol/100; x_a2 *= N_list[1]/Vol/100 # 1/m^2/s
    y_a1 *= N_list[0]/Vol/100; y_a2 *= N_list[1]/Vol/100 # 1/m^2/s
else:
    cx_a1 *= N_list[0]/Vol/100; cx_a2 *= N_list[1]/Vol/100; cx_a3 *= N_list[2]/Vol/100; cx_a4 *= N_list[3]/Vol/100 # 1/m^2/s
    cy_a1 *= N_list[0]/Vol/100; cy_a2 *= N_list[1]/Vol/100; cy_a3 *= N_list[2]/Vol/100; cy_a4 *= N_list[3]/Vol/100 # 1/m^2/s
    ax_a1 *= N_list[0]/Vol/100; ax_a2 *= N_list[1]/Vol/100; ax_a3 *= N_list[2]/Vol/100; ax_a4 *= N_list[3]/Vol/100 # 1/m^2/s
    ay_a1 *= N_list[0]/Vol/100; ay_a2 *= N_list[1]/Vol/100; ay_a3 *= N_list[2]/Vol/100; ay_a4 *= N_list[3]/Vol/100 # 1/m^2/s



# Create flux and driving force matrices
L_arr = []
for i, F_i in enumerate(F):
    if check == 1:
        if N_list[1] == 0:
            X_x = np.array([[F_arr[0,0,i]],[F_arr[0,2,i]]])
            X_z = np.array([[F_arr[2,0,i]],[F_arr[2,2,i]]])
        elif N_list[2] == 0:
            X_x = np.array([[F_arr[0,0,i]],[F_arr[0,1,i]]])
            X_y = np.array([[F_arr[1,0,i]],[F_arr[1,1,i]]])
        else:
            X_x = np.array([[F_arr[0,0,i]],[F_arr[0,1,i]],[F_arr[0,2,i]]])
            X_y = np.array([[F_arr[1,0,i]],[F_arr[1,1,i]],[F_arr[1,2,i]]])
            X_z = np.array([[F_arr[2,0,i]],[F_arr[2,1,i]],[F_arr[2,2,i]]])
    elif check == 2:
        if N_list[0] == 0:
            X_y = np.array([[F_arr[1,1,i]],[F_arr[1,2,i]]])
            X_z = np.array([[F_arr[2,1,i]],[F_arr[2,2,i]]])
        elif N_list[1] == 0:
            X_x = np.array([[F_arr[0,0,i]],[F_arr[0,2,i]]])
            X_z = np.array([[F_arr[2,0,i]],[F_arr[2,2,i]]])
        else:
            X_x = np.array([[F_arr[0,0,i]],[F_arr[0,1,i]],[F_arr[0,2,i]]])
            X_y = np.array([[F_arr[1,0,i]],[F_arr[1,1,i]],[F_arr[1,2,i]]])
            X_z = np.array([[F_arr[2,0,i]],[F_arr[2,1,i]],[F_arr[2,2,i]]])
    elif check == 3:
        X_x = np.array([[F_arr[0,0,i]],[F_arr[0,1,i]]])
        X_y = np.array([[F_arr[1,0,i]],[F_arr[1,1,i]]])
    else:
        X_cx = np.array([[F_arr[0,0,i]],[F_arr[0,1,i]],[F_arr[0,2,i]],[F_arr[0,3,i]]])
        X_cy = np.array([[F_arr[1,0,i]],[F_arr[1,1,i]],[F_arr[1,2,i]],[F_arr[1,3,i]]])
        X_ax = np.array([[F_arr[2,0,i]],[F_arr[2,1,i]],[F_arr[2,2,i]],[F_arr[2,3,i]]])
        X_ay = np.array([[F_arr[3,0,i]],[F_arr[3,1,i]],[F_arr[3,2,i]],[F_arr[3,3,i]]])

    # Extract Onsager coefficients from the flux and driving force matrices
    L_temp = []
    for j in range(runs):
        if check == 1:
            if N_list[1] == 0:
                J_x = np.array([[x_a1[i,j]],[x_a3[i,j]]])
                J_z = np.array([[z_a1[i,j]],[z_a3[i,j]]])
    
                L = np.matmul( np.matmul(J_x, X_x.T) + np.matmul(J_z, X_z.T), np.linalg.inv(np.matmul(X_x, X_x.T) + np.matmul(X_z, X_z.T)) ) # s/kg/m^3
                L_temp.append([L[0,0], 0.0, L[1,1], 0.0, 0.0, L[0,1], L[1,0], 0.0, 0.0])
                #print('{: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e}'.format(0.0, L[0,0], L[1,1], 0.0, 0.0, 0.0, 0.0, L[0,1], L[1,0]))
            elif N_list[2] == 0:
                J_x = np.array([[x_a1[i,j]],[x_a2[i,j]]])
                J_y = np.array([[y_a1[i,j]],[y_a2[i,j]]])
    
                L = np.matmul( np.matmul(J_x, X_x.T) + np.matmul(J_y, X_y.T), np.linalg.inv(np.matmul(X_x, X_x.T) + np.matmul(X_y, X_y.T)) ) # s/kg/m^3
                L_temp.append([L[0,0], L[1,1], 0.0, L[0,1], L[1,0], 0.0, 0.0, 0.0, 0.0])
                #print('{: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e}'.format(L[0,0], 0.0, L[1,1], L[0,1], L[1,0], 0.0, 0.0, 0.0, 0.0))
            else:
                J_x = np.array([[x_a1[i,j]],[x_a2[i,j]],[x_a3[i,j]]])
                J_y = np.array([[y_a1[i,j]],[y_a2[i,j]],[y_a3[i,j]]])
                J_z = np.array([[z_a1[i,j]],[z_a2[i,j]],[z_a3[i,j]]])
    
                L = np.matmul( np.matmul(J_x, X_x.T) + np.matmul(J_y, X_y.T) + np.matmul(J_z, X_z.T), np.linalg.inv(np.matmul(X_x, X_x.T) + np.matmul(X_y, X_y.T) + np.matmul(X_z, X_z.T)) ) # s/kg/m^3
                L_temp.append([L[0,0], L[1,1], L[2,2], L[0,1], L[1,0], L[0,2], L[2,0], L[1,2], L[2,1]])
                #print('{: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e}'.format(L[0,0], L[1,1], L[2,2], L[0,1], L[1,0], L[0,2], L[2,0], L[1,2], L[2,1]))
        elif check == 2:
            if N_list[0] == 0:
                J_y = np.array([[y_a2[i,j]],[y_a3[i,j]]])
                J_z = np.array([[z_a2[i,j]],[z_a3[i,j]]])

                L = np.matmul( np.matmul(J_y, X_y.T) + np.matmul(J_z, X_z.T), np.linalg.inv(np.matmul(X_y, X_y.T) + np.matmul(X_z, X_z.T)) ) # s/kg/m^3
                L_temp.append([0.0, L[0,0], L[1,1], 0.0, 0.0, 0.0, 0.0, L[0,1], L[1,0]])
                #print('{: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e}'.format(0.0, L[0,0], L[1,1], 0.0, 0.0, 0.0, 0.0, L[0,1], L[1,0]))
            elif N_list[1] == 0:
                J_x = np.array([[x_a1[i,j]],[x_a3[i,j]]])
                J_z = np.array([[z_a1[i,j]],[z_a3[i,j]]])

                L = np.matmul( np.matmul(J_x, X_x.T) + np.matmul(J_z, X_z.T), np.linalg.inv(np.matmul(X_x, X_x.T) + np.matmul(X_z, X_z.T)) ) # s/kg/m^3
                L_temp.append([L[0,0], 0.0, L[1,1], 0.0, 0.0, L[0,1], L[1,0], 0.0, 0.0])
                #print('{: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e}'.format(L[0,0], 0.0, L[1,1], L[0,1], L[1,0], 0.0, 0.0, 0.0, 0.0))
            else:
                J_x = np.array([[x_a1[i,j]],[x_a2[i,j]],[x_a3[i,j]]])
                J_y = np.array([[y_a1[i,j]],[y_a2[i,j]],[y_a3[i,j]]])
                J_z = np.array([[z_a1[i,j]],[z_a2[i,j]],[z_a3[i,j]]])

                L = np.matmul( np.matmul(J_x, X_x.T) + np.matmul(J_y, X_y.T) + np.matmul(J_z, X_z.T), np.linalg.inv(np.matmul(X_x, X_x.T) + np.matmul(X_y, X_y.T) + np.matmul(X_z, X_z.T)) ) # s/kg/m^3
                L_temp.append([L[0,0], L[1,1], L[2,2], L[0,1], L[1,0], L[0,2], L[2,0], L[1,2], L[2,1]])
                #print('{: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e}'.format(L[0,0], L[1,1], L[2,2], L[0,1], L[1,0], L[0,2], L[2,0], L[1,2], L[2,1]))
        elif check == 3:
            J_x = np.array([[x_a1[i,j]],[x_a2[i,j]]])
            J_y = np.array([[y_a1[i,j]],[y_a2[i,j]]])

            L = np.matmul( np.matmul(J_x, X_x.T) + np.matmul(J_y, X_y.T), np.linalg.inv(np.matmul(X_x, X_x.T) + np.matmul(X_y, X_y.T)) ); L_temp.append([L[0,0], L[1,1], L[0,1], L[1,0]]) # s/kg/m^3
        else:
            J_cx = np.array([[cx_a1[i,j]],[cx_a2[i,j]],[cx_a3[i,j]],[cx_a4[i,j]]])
            J_cy = np.array([[cy_a1[i,j]],[cy_a2[i,j]],[cy_a3[i,j]],[cy_a4[i,j]]])
            J_ax = np.array([[ax_a1[i,j]],[ax_a2[i,j]],[ax_a3[i,j]],[ax_a4[i,j]]])
            J_ay = np.array([[ay_a1[i,j]],[ay_a2[i,j]],[ay_a3[i,j]],[ay_a4[i,j]]])

            L = np.matmul( np.matmul(J_cx, X_cx.T) + np.matmul(J_cy, X_cy.T) + np.matmul(J_ax, X_ax.T) + np.matmul(J_ay, X_ay.T), np.linalg.inv(np.matmul(X_cx, X_cx.T) + np.matmul(X_cy, X_cy.T) + np.matmul(X_ax, X_ax.T) + np.matmul(X_ay, X_ay.T)) ) # s/kg/m^3
            L_temp.append([L[0,0], L[1,1], L[2,2], L[3,3], L[0,1], L[1,0], L[0,2], L[2,0], L[0,3], L[3,0], L[1,2], L[2,1], L[1,3], L[3,1], L[2,3], L[3,2]])
            #print('{: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e}'.format(L[0,0], L[1,1], L[2,2], L[3,3], L[0,1], L[1,0], L[0,2], L[2,0], L[0,3], L[3,0], L[1,2], L[2,1], L[1,3], L[3,1], L[2,3], L[3,2]))

    L_arr.append(L_temp)
L_arr = np.array(L_arr)

# Extract zero-field limit Onsager coefficients from Onsager coefficients as a function of field strength
L_temp = []
for i in range(L_arr.shape[1]):
    L_sub = []
    for j in range(L_arr.shape[2]):
        L_0 = stats.linregress(F,L_arr[:,i,j]).intercept
        L_sub.append(L_0)
    L_temp.append(L_sub)
L_temp = np.array(L_temp)
F = np.insert(F, 0, [0.0])
L_arr = np.insert(L_arr, 0, L_temp, axis=0)





# Print data
if check == 1 or check == 2:
    L = []; L.append(np.mean(L_arr[:,:,0], axis=1)); L.append(np.mean(L_arr[:,:,1], axis=1)); L.append(np.mean(L_arr[:,:,2], axis=1)); L.append(np.mean(np.concatenate((L_arr[:,:,3], L_arr[:,:,4]), axis = 1), axis=1)); L.append(np.mean(np.concatenate((L_arr[:,:,5], L_arr[:,:,6]), axis = 1), axis=1)); L.append(np.mean(np.concatenate((L_arr[:,:,7], L_arr[:,:,8]), axis = 1), axis=1)); L = np.array(L).T
    L_std = []; L_std.append(np.std(L_arr[:,:,0], axis=1)); L_std.append(np.std(L_arr[:,:,1], axis=1)); L_std.append(np.std(L_arr[:,:,2], axis=1)); L_std.append(np.std(np.concatenate((L_arr[:,:,3], L_arr[:,:,4]), axis = 1), axis=1)); L_std.append(np.std(np.concatenate((L_arr[:,:,5], L_arr[:,:,6]), axis = 1), axis=1)); L_std.append(np.std(np.concatenate((L_arr[:,:,7], L_arr[:,:,8]), axis = 1), axis=1)); L_std = np.array(L_std).T

    L_save = []; L_save.append(L_temp[:,0]); L_save.append(L_temp[:,1]); L_save.append(L_temp[:,2]); L_save.append(np.mean(L_temp[:,3:5], axis = 1)); L_save.append(np.mean(L_temp[:,5:7], axis = 1)); L_save.append(np.mean(L_temp[:,7:9], axis = 1)); L_save = np.array(L_save)

    writefile = open('Lij.xvg','w')
    writefile.write('# L11 L22 L33 L12 L13 L23 (s/kg/m^3)\n')
    for i, F_i in enumerate(F):
        writefile.write(' {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e}\n'.format(F_i, L[i,0], L[i,1], L[i,2], L[i,3], L[i,4], L[i,5]))
    writefile.close()

    writefile = open('Lij_std.xvg','w')
    writefile.write('# L11 L22 L33 L12 L13 L23 (s/kg/m^3)\n')
    for i, F_i in enumerate(F):
        writefile.write(' {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e}\n'.format(F_i, L_std[i,0], L_std[i,1], L_std[i,2], L_std[i,3], L_std[i,4], L_std[i,5]))
    writefile.close()
elif check == 3:
    L = []; L.append(np.mean(L_arr[:,:,0], axis=1)); L.append(np.mean(L_arr[:,:,1], axis=1)); L.append(np.mean(np.concatenate((L_arr[:,:,2], L_arr[:,:,3]), axis = 1), axis=1)); L = np.array(L).T
    L_std = []; L_std.append(np.std(L_arr[:,:,0], axis=1)); L_std.append(np.std(L_arr[:,:,1], axis=1)); L_std.append(np.std(np.concatenate((L_arr[:,:,2], L_arr[:,:,3]), axis = 1), axis=1)); L_std = np.array(L_std).T

    L_save = []; L_save.append(L_temp[:,0]); L_save.append(L_temp[:,1]); L_save.append(np.mean(L_temp[:,2:4], axis = 1)); L_save = np.array(L_save)

    writefile = open('Lij.xvg','w')
    writefile.write('# L11 L22 L12 (s/kg/m^3)\n')
    for i, F_i in enumerate(F):
        writefile.write(' {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e}\n'.format(F_i, L[i,0], L[i,1], L[i,2]))
    writefile.close()

    writefile = open('Lij_std.xvg','w')
    writefile.write('# L11 L22 L12 (s/kg/m^3)\n')
    for i, F_i in enumerate(F):
        writefile.write(' {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e}\n'.format(F_i, L_std[i,0], L_std[i,1], L_std[i,2]))
    writefile.close()
else:
    L = []; L.append(np.mean(L_arr[:,:,0], axis=1)); L.append(np.mean(L_arr[:,:,1], axis=1)); L.append(np.mean(L_arr[:,:,2], axis=1)); L.append(np.mean(L_arr[:,:,3], axis=1)); L.append(np.mean(np.concatenate((L_arr[:,:,4],  L_arr[:,:,5]),  axis = 1), axis=1)); L.append(np.mean(np.concatenate((L_arr[:,:,6],  L_arr[:,:,7]),  axis = 1), axis=1)); L.append(np.mean(np.concatenate((L_arr[:,:,8],  L_arr[:,:,9]),  axis = 1), axis=1)); L.append(np.mean(np.concatenate((L_arr[:,:,10], L_arr[:,:,11]), axis = 1), axis=1)); L.append(np.mean(np.concatenate((L_arr[:,:,12], L_arr[:,:,13]), axis = 1), axis=1)); L.append(np.mean(np.concatenate((L_arr[:,:,14], L_arr[:,:,15]), axis = 1), axis=1)); L = np.array(L).T
    L_std = []; L_std.append(np.std(L_arr[:,:,0], axis=1)); L_std.append(np.std(L_arr[:,:,1], axis=1)); L_std.append(np.std(L_arr[:,:,2], axis=1)); L_std.append(np.std(L_arr[:,:,3], axis=1)); L_std.append(np.std(np.concatenate((L_arr[:,:,4],  L_arr[:,:,5]),  axis = 1), axis=1)); L_std.append(np.std(np.concatenate((L_arr[:,:,6],  L_arr[:,:,7]),  axis = 1), axis=1)); L_std.append(np.std(np.concatenate((L_arr[:,:,8],  L_arr[:,:,9]),  axis = 1), axis=1)); L_std.append(np.std(np.concatenate((L_arr[:,:,10], L_arr[:,:,11]), axis = 1), axis=1)); L_std.append(np.std(np.concatenate((L_arr[:,:,12], L_arr[:,:,13]), axis = 1), axis=1)); L_std.append(np.std(np.concatenate((L_arr[:,:,14], L_arr[:,:,15]), axis = 1), axis=1)); L_std = np.array(L_std).T

    L_save = []; L_save.append(L_temp[:,0]); L_save.append(L_temp[:,1]); L_save.append(L_temp[:,2]); L_save.append(L_temp[:,3]); L_save.append(np.mean(L_temp[:,4:6], axis = 1)); L_save.append(np.mean(L_temp[:,6:8], axis = 1)); L_save.append(np.mean(L_temp[:,8:10], axis = 1)); L_save.append(np.mean(L_temp[:,10:12], axis = 1)); L_save.append(np.mean(L_temp[:,12:14], axis = 1)); L_save.append(np.mean(L_temp[:,14:16], axis = 1)); L_save = np.array(L_save)

    writefile = open('Lij.xvg','w')
    writefile.write('# L11 L22 L33 L44 L12 L13 L14 L23 L24 L34 (s/kg/m^3)\n')
    for i, F_i in enumerate(F):
        writefile.write(' {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e}\n'.format(F_i, L[i,0], L[i,1], L[i,2], L[i,3], L[i,4], L[i,5], L[i,6], L[i,7], L[i,8], L[i,9]))
    writefile.close()

    writefile = open('Lij_std.xvg','w')
    writefile.write('# L11 L22 L33 L44 L12 L13 L14 L23 L24 L34 (s/kg/m^3)\n')
    for i, F_i in enumerate(F):
        writefile.write(' {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e} {: 1.3e}\n'.format(F_i, L_std[i,0], L_std[i,1], L_std[i,2], L_std[i,3], L_std[i,4], L_std[i,5], L_std[i,6], L_std[i,7], L_std[i,8], L_std[i,9]))
    writefile.close()

with h5py.File('L.hdf5','w') as f:
    dset1 = f.create_dataset("L", data=L_save)