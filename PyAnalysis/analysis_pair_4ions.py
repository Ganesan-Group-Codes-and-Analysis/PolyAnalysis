# This script creates a 2D array specifying all clusters based on the number of a1 and a2
# This script can be used to create custom scripts for different purposes. The 2D array is output because it can be used to gain a lot of information in post

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

script, trj_file, top_file, c1_name, c2_name, a1_name, a2_name, coord_c1a1, coord_c1a2, coord_c2a1, coord_c2a2, coord_c1W, coord_c2W, coord_a1W, coord_a2W, t_min, t_max, step, nt = argv
coord_c1a1 = float(coord_c1a1); coord_c1a2 = float(coord_c1a2); coord_c2a1 = float(coord_c2a1); coord_c2a2 = float(coord_c2a2); coord_c1W = float(coord_c1W); coord_c2W = float(coord_c2W); coord_a1W = float(coord_a1W); coord_a2W = float(coord_a2W); t_min = float(t_min); t_max = float(t_max); step = int(step); nt = int(nt)
# trj_file = .trr/.xtc file, top_file = .tpr file
# cX_name = name of X cation. Define c1_name and c2_name as the same thing if there is only 1 cation.
# aX_name = name of X anion. Define a1_name and a2_name as the same thing if there is only 1 aation.
# coord_cXaX = coordination distance between cX and aX (Angstroms). If c1 == c2 or a1 == a2, coord_c2aX == -1 or coord_cXa2 == -1.
# coord_cXW = coordination distance between cX and water OXYGEN (Angstroms). If c1 == c2, coord_c2W == -1
# coord_aXW = coordination distance between aX and water HYDROGEN (Angstroms). If c1 == c2, coord_c2W == -1
# t_min = start time for analysis in ps (-1 assumes the start time of the first frame)
# t_max = end time for analysis in ps (-1 assumes the end time of the last frame)
# step = frame step-size (-1 assumes a step-size of 1)
# nt = number of threads
#
# Example of use: python3 analysis_pair_4ions.py md.xtc md.tpr LI K CL CL 3.01 -1 4.15 -1 2.83 3.61 2.98 -1 75000 100000 1 128



def pairPrint(filename, pair):# print ion pair proability distribution
    with open('{}.dat'.format(filename), 'w') as anaout:
        print('# ' + filename + ' Probability', file=anaout)
        for i in range(0, len(pair)):
            print('{:10.3f} {:10.5f}'.format(i, pair[i]), file=anaout)

def printCluster2D(filename, cluster):# print ion pair proability distribution
    with open('{}.dat'.format(filename), 'w') as anaout:
        print('# Number of ion cluster with x {} and y {}'.format(a1_name, a2_name), file=anaout)
        print('# x refers to the row, starting with x = 0; y refers to the columns, starting with y = 0'.format(a1_name, a2_name), file=anaout)
        print('\n'.join(['\t'.join(['{:5.5f}'.format(cell) for cell in row]) for row in cluster]), file=anaout)



def iPair(frame):
# finds all contact pairing (CP) and solvent-separated (SS) pairing in a given frame
#
# Inputs: frame = time frame to be analyzed

    if frame%5000 == 0:
        print("Frame " + str(frame))

     # Load in data for current frame
    with h5py.File('/tmp/Cluster.hdf5','r') as f:
        if c1_name == c2_name:
            dset1 = f['r_c1']; c1 = dset1[frame]; l_c1 = len(c1)
        else:
            dset1 = f['r_c1']; c1 = dset1[frame]; l_c1 = len(c1)
            dset2 = f['r_c2']; c2 = dset2[frame]; l_c2 = len(c2)
        if a1_name == a2_name:
            dset3 = f['r_a1']; a1 = dset3[frame]; l_a1 = len(a1)
        else:
            dset3 = f['r_a1']; a1 = dset3[frame]; l_a1 = len(a1)
            dset4 = f['r_a2']; a2 = dset4[frame]; l_a2 = len(a2)
        dset5 = f['r_H2O']; H2O = dset5[frame]; l_H2O = len(H2O)    # Water oxygen atoms
        dset6 = f['r_H2O_H']; H2O_H = dset6[frame]                  # Water hydrogen atoms
        dset7 = f['cells']; cell = dset7[frame]

     # Find all cX, aX, water coordinations; Contact Pairs (CP)
     # distpair_XY[:,0] represent X indexes from 0 to (l_X - 1), distpair_XY[:,1] represents Y indexes from 0 to (l_Y - 1)
    distpair_c1a1 = distances.capped_distance(c1, a1, coord_c1a1, box=cell)[0]                                                                      # c1-a1
    distpair_c1W = distances.capped_distance(c1, H2O, coord_c1W, box=cell)[0]                                                                       # c1-H2O O
    distpair_a1W = distances.capped_distance(a1, H2O_H, coord_a1W, box=cell)[0]; distpair_a1W[:,1] = np.array(distpair_a1W[:,1]/2,dtype=int)        # a1-H2O H
    l_tot = l_c1 + l_a1 + l_H2O                                                                                                                     # Tracks total number of ions/water
    if not c1_name == c2_name:
        distpair_c2a1 = distances.capped_distance(c2, a1, coord_c2a1, box=cell)[0]                                                                  # c2-a1
        distpair_c2W = distances.capped_distance(c2, H2O, coord_c2W, box=cell)[0]                                                                   # c2-H2O O

        l_tot = l_c1 + l_c2 + l_a1 + l_H2O                                                                                                          # Tracks total number of ions/water
    if not a1_name == a2_name:
        distpair_c1a2 = distances.capped_distance(c1, a2, coord_c1a2, box=cell)[0]                                                                  # c1-a2
        distpair_a2W = distances.capped_distance(a2, H2O_H, coord_a2W, box=cell)[0]; distpair_a2W[:,1] = np.array(distpair_a2W[:,1]/2,dtype=int)    # a2-H2O H

        l_tot = l_c1 + l_a1 + l_a2 + l_H2O
    if (not c1_name == c2_name) and (not a1_name == a2_name):
        distpair_c2a2 = distances.capped_distance(c2, a2, coord_c2a2, box=cell)[0]                                                                  # c2-a2

        l_tot = l_c1 + l_c2 + l_a1 + l_a2 + l_H2O                                                                                                   # Tracks total number of ions/water



     # Create networkx graph of size sum(l_tot)
     # Note, within the networkx graph
       # X is indexed from 0 to (l_X - 1), e.g., 0 to 99 for 100 atoms
       # Y is indexed from l_X to (l_X + l_Y - 1), e.g., 100 to 199 for 100 atoms, etc
    G = nx.Graph()
    G.add_nodes_from(range(l_tot))

     # Create connections (pairs) between all ion-ion and ion-water pairs
     # Must convert indexes from distpairs to networkx graph
    if c1_name == c2_name:
        distpair_c1a1[:,1] += l_c1
        distpair_c1W[:,1] += l_tot - l_H2O
        distpair_a1W[:,0] += l_c1
        distpair_a1W[:,1] += l_tot - l_H2O

        G.add_edges_from(distpair_c1a1)
        G.add_edges_from(distpair_c1W)
        G.add_edges_from(distpair_a1W)
    else:
        distpair_c1a1[:,1] += l_c1 + l_c2
        distpair_c1W[:,1] += l_tot - l_H2O
        distpair_a1W[:,0] += l_c1 + l_c2
        distpair_a1W[:,1] += l_tot - l_H2O
        distpair_c2a1[:,0] += l_c1
        distpair_c2a1[:,1] += l_c1 + l_c2
        distpair_c2W[:,0] += l_c1
        distpair_c2W[:,1] += l_tot - l_H2O

        G.add_edges_from(distpair_c1a1)
        G.add_edges_from(distpair_c2a1)
        G.add_edges_from(distpair_c1W)
        G.add_edges_from(distpair_c2W)
        G.add_edges_from(distpair_a1W)
    
    if not a1_name == a2_name:
        if c1_name == c2_name:
            distpair_c1a2[:,1] += l_c1 + l_a1
            distpair_a2W[:,0] += l_c1 + l_a1
            distpair_a2W[:,1] += l_tot - l_H2O

            G.add_edges_from(distpair_c1a2)
            G.add_edges_from(distpair_a2W)
        else:
            distpair_c1a2[:,1] += l_c1 + l_c2 + l_a1
            distpair_c2a2[:,0] += l_c1
            distpair_c2a2[:,1] += l_c1 + l_c2 + l_a1
            distpair_a2W[:,0] += l_c1 + l_c2 + l_a1
            distpair_a2W[:,1] += l_tot - l_H2O

            G.add_edges_from(distpair_c1a2)
            G.add_edges_from(distpair_c2a2)
            G.add_edges_from(distpair_a2W)


    
     # CP -> Contact Pairing; SS -> Solvent-Separated Pairing
    CP_c1 = np.zeros((l_c1,2)); SS_c1 = np.zeros((l_c1,2))
    for i in range(l_c1): # loop over c1
        subgraph = nx.ego_graph(G,i,radius=1) # create a graph of c1 with all contact pairs. Radius defines maximum number of connections between c1 and a molecule within its cluster
        if c1_name == c2_name:
            n = np.array(list(subgraph.nodes())); n = n[n >= l_c1]; n = n[n < l_c1 + l_a1] # isolate number of a1 contact pairs
        else:
            n = np.array(list(subgraph.nodes())); n = n[n >= l_c1 + l_c2]; n = n[n < l_c1 + l_c2 + l_a1] # isolate number of a1 contact pairs
        CP_c1[i,0] = len(n)

        if not a1_name == a2_name:
            if c1_name == c2_name:
                n = np.array(list(subgraph.nodes())); n = n[n >= l_c1 + l_a1]; n = n[n < l_c1 + l_a1 + l_a2] # isolate number of a2 contact pairs
            else:
                n = np.array(list(subgraph.nodes())); n = n[n >= l_c1 + l_c2 + l_a1]; n = n[n < l_c1 + l_c2 + l_a1 + l_a2] # isolate number of a2 contact pairs
            CP_c1[i,1] = len(n) # number SS = (number CP+SS) - (number CP)



        subgraph = nx.ego_graph(G,i,radius=2) # create a graph of c1 with all contact and water separated pairs. Radius defines maximum number of connections between c1 and a molecule within its cluster
        if c1_name == c2_name:
            n = np.array(list(subgraph.nodes())); n = n[n >= l_c1]; n = n[n < l_c1 + l_a1] # isolate number of a1 pairs (either contact or solvent-separated). Note, a1 must be contact paired OR solvent-separated within a radius of 2 since c1-c1 pairing is not considered
        else:
            n = np.array(list(subgraph.nodes())); n = n[n >= l_c1 + l_c2]; n = n[n < l_c1 + l_c2 + l_a1] # isolate number of a1 pairs (either contact or solvent-separated). Note, a1 must be contact paired OR solvent-separated within a radius of 2 since c1-c1 pairing is not considered
        SS_c1[i,0] = (len(n) - CP_c1[i,0])

        if not a1_name == a2_name:
            if c1_name == c2_name:
                n = np.array(list(subgraph.nodes())); n = n[n >= l_c1 + l_a1]; n = n[n < l_c1 + l_a1 + l_a2] # isolate number of a2 pairs (either contact or solvent-separated). Note, a2 must be contact paired OR solvent-separated within a radius of 2 since c1-c1 pairing is not considered
            else:
                n = np.array(list(subgraph.nodes())); n = n[n >= l_c1 + l_c2 + l_a1]; n = n[n < l_c1 + l_c2 + l_a1 + l_a2] # isolate number of a2 pairs (either contact or solvent-separated). Note, a2 must be contact paired OR solvent-separated within a radius of 2 since c1-c1 pairing is not considered
            SS_c1[i,1] = (len(n) - CP_c1[i,1]) # number SS = (number CP+SS) - (number CP)
    


     # Repeat above for all ions
    
     # c2
    if not c1_name == c2_name:
        CP_c2 = np.zeros((l_c2,2)); SS_c2 = np.zeros((l_c2,2))
        for i in range(l_c2):
            subgraph = nx.ego_graph(G,i + l_c1,radius=1)
            n = np.array(list(subgraph.nodes())); n = n[n >= l_c1 + l_c2]; n = n[n < l_c1 + l_c2 + l_a1]
            CP_c2[i,0] = len(n)

            if not a1_name == a2_name:
                n = np.array(list(subgraph.nodes())); n = n[n >= l_c1 + l_c2 + l_a1]; n = n[n < l_c1 + l_c2 + l_a1 + l_a2]
                CP_c2[i,1] = len(n)



            subgraph = nx.ego_graph(G,i + l_c1,radius=2)
            n = np.array(list(subgraph.nodes())); n = n[n >= l_c1 + l_c2]; n = n[n < l_c1 + l_c2 + l_a1]
            SS_c2[i,0] = (len(n) - CP_c2[i,0])

            if not a1_name == a2_name:
                n = np.array(list(subgraph.nodes())); n = n[n >= l_c1 + l_c2 + l_a1]; n = n[n < l_c1 + l_c2 + l_a1 + l_a2]
                SS_c2[i,1] = (len(n) - CP_c2[i,1])



    # a1
    CP_a1 = np.zeros((l_a1,2)); SS_a1 = np.zeros((l_a1,2))
    for i in range(l_a1):
        if c1_name == c2_name:
            subgraph = nx.ego_graph(G,i + l_c1,radius=1)
        else:
            subgraph = nx.ego_graph(G,i + l_c1 + l_c2,radius=1)
        n = np.array(list(subgraph.nodes())); n = n[n < l_c1]
        CP_a1[i,0] = len(n)

        if not c1_name == c2_name:
            n = np.array(list(subgraph.nodes())); n = n[n >= l_c1]; n = n[n < l_c1 + l_c2]
            CP_a1[i,1] = len(n)



        if c1_name == c2_name:
            subgraph = nx.ego_graph(G,i + l_c1,radius=2)
        else:
            subgraph = nx.ego_graph(G,i + l_c1 + l_c2,radius=2)
        n = np.array(list(subgraph.nodes())); n = n[n < l_c1]
        SS_a1[i,0] = (len(n) - CP_a1[i,0])

        if not c1_name == c2_name:
            n = np.array(list(subgraph.nodes())); n = n[n >= l_c1]; n = n[n < l_c1 + l_c2]
            SS_a1[i,1] = (len(n) - CP_a1[i,1])
    


    # a2
    if not a1_name == a2_name:
        CP_a2 = np.zeros((l_a2,2)); SS_a2 = np.zeros((l_a2,2))
        for i in range(l_a2):
            if c1_name == c2_name:
                subgraph = nx.ego_graph(G,i + l_c1 + l_a1,radius=1)
            else:
                subgraph = nx.ego_graph(G,i + l_c1 + l_c2 + l_a1,radius=1)
            n = np.array(list(subgraph.nodes())); n = n[n < l_c1]
            CP_a2[i,0] = len(n)

            if not c1_name == c2_name:
                n = np.array(list(subgraph.nodes())); n = n[n >= l_c1]; n = n[n < l_c1 + l_c2]
                CP_a2[i,1] = len(n)



            if c1_name == c2_name:
                subgraph = nx.ego_graph(G,i + l_c1 + l_a1,radius=2)
            else:
                subgraph = nx.ego_graph(G,i + l_c1 + l_c2 + l_a1,radius=2)
            n = np.array(list(subgraph.nodes())); n = n[n < l_c1]
            SS_a2[i,0] = (len(n) - CP_a2[i,0])

            if not c1_name == c2_name:
                n = np.array(list(subgraph.nodes())); n = n[n >= l_c1]; n = n[n < l_c1 + l_c2]
                SS_a2[i,1] = (len(n) - CP_a2[i,1])



    G.clear()

    if c1_name == c2_name and a1_name == a2_name:
        return [np.concatenate((CP_c1, CP_a1)), np.concatenate((SS_c1, SS_a1))]
    elif (not c1_name == c2_name) and a1_name == a2_name:
        return [np.concatenate((CP_c1, CP_c2, CP_a1)), np.concatenate((SS_c1, SS_c2, SS_a1))]
    elif c1_name == c2_name and (not a1_name == a2_name):
        return [np.concatenate((CP_c1, CP_a1, CP_a2)), np.concatenate((SS_c1, SS_a1, SS_a2))]
    else:
        return [np.concatenate((CP_c1, CP_c2, CP_a1, CP_a2)), np.concatenate((SS_c1, SS_c2, SS_a1, SS_a2))]



def pair_dist(pair_ar):
# create the ion pair probability distribution
#
# Inputs: pair_ar = array of atoms a1 over all time frames containing the number of bound atoms a2 or vice versa

    npair = np.zeros(30)# Size of the distribution. Can adjust as needed
    for pair in pair_ar:
        for i in range(len(npair)):
            npair[int(i)] += np.count_nonzero(pair == i)

    return npair / np.shape(pair_ar)[0] / np.shape(pair_ar[1])



def load_TRR():
# loads in the trajectory and saves the necessary data to a temporary h5py file

    global t_min, t_max, step

    uta = mda.Universe(top_file, trj_file)  # Load in trajectory and topology

     # Define ions and water atoms
    if c1_name == c2_name:
        c1 = uta.select_atoms("name " + c1_name)
    else:
        c1 = uta.select_atoms("name " + c1_name)
        c2 = uta.select_atoms("name " + c2_name)
    if a1_name == a2_name:
        a1 = uta.select_atoms("name " + a1_name)
    else:
        a1 = uta.select_atoms("name " + a1_name)
        a2 = uta.select_atoms("name " + a2_name)
    H2O = uta.select_atoms("name OW")
    H2O_H = uta.select_atoms("name HW1 or name HW2")

     # Define the system times/frames to be calculated over
    if t_min == -1:
        t_min = uta.trajectory[0].time
    if t_max == -1:
        t_max = uta.trajectory[-1].time
    if step < 1:
        step = 1
    dt = np.round((uta.trajectory[1].time - uta.trajectory[0].time),3)
    frame_ids = np.arange(int((t_min - uta.trajectory[0].time)/dt), int((t_max - uta.trajectory[0].time)/dt + 1), step)
    dt = dt*step

     # Load in the necessary data: ion/water atom positions, cell dimensions
    r_c1 = []; r_c2 = []; r_a1 = []; r_a2 = []; r_H2O = []; r_H2O_H = []; cells = []
    for frame in frame_ids:

        if frame%5000 == 0:
            print("Frame " + str(frame))
        
        ts = uta.trajectory[frame]
        cell = ts.dimensions
        cells.append(cell)

        if c1_name == c2_name:
            r_c1.append(c1.positions)
        else:
            r_c1.append(c1.positions)
            r_c2.append(c2.positions)

        if a1_name == a2_name:
            r_a1.append(a1.positions)
        else:
            r_a1.append(a1.positions)
            r_a2.append(a2.positions)

        r_H2O.append(H2O.positions)
        r_H2O_H.append(H2O_H.positions)

     # Save necessary infomration to a .hdf5 file for later use in the calculation
    with h5py.File('/tmp/Cluster.hdf5','w') as f:
        dset1 = f.create_dataset("cells", data = cells)
        if c1_name == c2_name:
            dset2 = f.create_dataset("r_c1", data=r_c1)
        else:
            dset2 = f.create_dataset("r_c1", data=r_c1)
            dset3 = f.create_dataset("r_c2", data=r_c2)
        if a1_name == a2_name:
            dset4 = f.create_dataset("r_a1", data=r_a1)
        else:
            dset4 = f.create_dataset("r_a1", data=r_a1)
            dset5 = f.create_dataset("r_a2", data=r_a2)
        dset6 = f.create_dataset("r_H2O", data=r_H2O)
        dset7 = f.create_dataset("r_H2O_H", data=r_H2O_H)
        dset8 = f.create_dataset("frames", data=frame_ids)



def main(trj_file, top_file, c1_name, c2_name, a1_name, a2_name, coord_c1a1, coord_c1a2, coord_c2a1, coord_c2a2, coord_c1W, coord_c2W, coord_a1W, coord_a2W, t_min, t_max, step, nt):

    # Load in the trajectory file
    if not os.path.exists('/tmp/Cluster.hdf5'):
        load_TRR()
        exit()

    with h5py.File('/tmp/Cluster.hdf5','r') as f:
        if c1_name == c2_name:
            dset1 = f['r_c1']; l_c1 = len(dset1[0])
        else:
            dset1 = f['r_c1']; l_c1 = len(dset1[0])
            dset2 = f['r_c2']; l_c2 = len(dset2[0])
        if a1_name == a2_name:
            dset3 = f['r_a1']; l_a1 = len(dset3[0])
        else:
            dset3 = f['r_a1']; l_a1 = len(dset3[0])
            dset4 = f['r_a2']; l_a2 = len(dset4[0])
        dset5 = f['frames']; frame_ids = dset5[:]

    print("CP and SS Pairing Analysis")
     # Perform the analysis using multi-threading
    pool = mp.Pool(processes=nt)
    func = functools.partial(iPair)
    pairing = pool.map(func, range(len(frame_ids)))
    pool.close()
    pool.join()
    pairing = np.array(pairing)

    CP = pairing[:,0,:,:]; SS = pairing[:,1,:,:]

     # Print out data
    if c1_name == c2_name and a1_name == a2_name:
        CP_c1a1 = pair_dist(CP[:,:l_c1,0]); SS_c1a1 = pair_dist(SS[:,:l_c1,0]); Pair_c1 = pair_dist(CP[:,:l_c1,0] + SS[:,:l_c1,0])
        CP_a1c1 = pair_dist(CP[:,l_c1:,0]); SS_a1c1 = pair_dist(SS[:,l_c1:,0]); Pair_a1 = pair_dist(CP[:,l_c1:,0] + SS[:,l_c1:,0])

        pairPrint("aPair_{}_{}".format(c1_name, a1_name), CP_c1a1)
        pairPrint("aPair_{}_{}".format(a1_name, c1_name), CP_a1c1)
        pairPrint("aPair_{}_{}_{}".format(c1_name, a1_name, "OW"), SS_c1a1)
        pairPrint("aPair_{}_{}_{}".format(a1_name, c1_name, "OW"), SS_a1c1)
        pairPrint("aPairT_{}".format(c1_name), Pair_c1)
        pairPrint("aPairT_{}".format(a1_name), Pair_a1)
    elif (not c1_name == c2_name) and a1_name == a2_name:
        CP_c1a1 = pair_dist(CP[:,:l_c1,0]); SS_c1a1 = pair_dist(SS[:,:l_c1,0]); Pair_c1 = pair_dist(CP[:,:l_c1,0] + SS[:,:l_c1,0])
        CP_c2a1 = pair_dist(CP[:,l_c1:l_c1+l_c2,0]); SS_c2a1 = pair_dist(SS[:,l_c1:l_c1+l_c2,0]); Pair_c2 = pair_dist(CP[:,l_c1:l_c1+l_c2,0] + SS[:,l_c1:l_c1+l_c2,0])
        CP_a1c1 = pair_dist(CP[:,l_c1+l_c2:,0]); SS_a1c1 = pair_dist(SS[:,l_c1+l_c2:,0])
        CP_a1c2 = pair_dist(CP[:,l_c1+l_c2:,1]); SS_a1c2 = pair_dist(SS[:,l_c1+l_c2:,1])
        Pair_a1c1 = pair_dist(CP[:,l_c1+l_c2:,0] + SS[:,l_c1+l_c2:,0])
        Pair_a1c2 = pair_dist(CP[:,l_c1+l_c2:,1] + SS[:,l_c1+l_c2:,1])
        Pair_a1 = pair_dist(CP[:,l_c1+l_c2:,0] + CP[:,l_c1+l_c2:,1] + SS[:,l_c1+l_c2:,0] + SS[:,l_c1+l_c2:,1]); Pair_CP_a1 = pair_dist(CP[:,l_c1+l_c2:,0] + CP[:,l_c1+l_c2:,1])

        pairPrint("aPair_{}_{}".format(c1_name, a1_name), CP_c1a1)
        pairPrint("aPair_{}_{}".format(c2_name, a1_name), CP_c2a1)
        pairPrint("aPair_{}_{}".format(a1_name, c1_name), CP_a1c1)
        pairPrint("aPair_{}_{}".format(a1_name, c2_name), CP_a1c2)
        pairPrint("aPair_{}_{}_{}".format(c1_name, a1_name, "OW"), SS_c1a1)
        pairPrint("aPair_{}_{}_{}".format(c2_name, a1_name, "OW"), SS_c2a1)
        pairPrint("aPair_{}_{}_{}".format(a1_name, c1_name, "OW"), SS_a1c1)
        pairPrint("aPair_{}_{}_{}".format(a1_name, c2_name, "OW"), SS_a1c2)
        pairPrint("aPairT_{}".format(c1_name), Pair_c1)
        pairPrint("aPairT_{}".format(c2_name), Pair_c2)
        pairPrint("aPairT_{}_{}".format(a1_name, c1_name), Pair_a1c1)
        pairPrint("aPairT_{}_{}".format(a1_name, c2_name), Pair_a1c2)
        pairPrint("aPairT_{}".format(a1_name), Pair_a1)
        pairPrint("aPairT_CP_{}".format(a1_name), Pair_CP_a1)
    elif c1_name == c2_name and (not a1_name == a2_name):
        CP_c1a1 = pair_dist(CP[:,:l_c1,0]); SS_c1a1 = pair_dist(SS[:,:l_c1,0])
        CP_c1a2 = pair_dist(CP[:,:l_c1,1]); SS_c1a2 = pair_dist(SS[:,:l_c1,1])
        Pair_c1a1 = pair_dist(CP[:,:l_c1,0] + SS[:,:l_c1,0])
        Pair_c1a2 = pair_dist(CP[:,:l_c1,1] + SS[:,:l_c1,1])
        Pair_c1 = pair_dist(CP[:,:l_c1,0] + CP[:,:l_c1,1] + SS[:,:l_c1,0] + SS[:,:l_c1,1]); Pair_CP_c1 = pair_dist(CP[:,:l_c1,0] + CP[:,:l_c1,1])
        CP_a1c1 = pair_dist(CP[:,l_c1:l_c1+l_a1,0]); SS_a1c1 = pair_dist(SS[:,l_c1:l_c1+l_a1,0]); Pair_a1 = pair_dist(CP[:,l_c1:l_c1+l_a1,0] + SS[:,l_c1:l_c1+l_a1,0])
        CP_a2c1 = pair_dist(CP[:,l_c1+l_a1:,0]); SS_a2c1 = pair_dist(SS[:,l_c1+l_a1:,0]); Pair_a2 = pair_dist(CP[:,l_c1+l_a1:,0] + SS[:,l_c1+l_a1:,0])

        pairPrint("aPair_{}_{}".format(c1_name, a1_name), CP_c1a1)
        pairPrint("aPair_{}_{}".format(c1_name, a2_name), CP_c1a2)
        pairPrint("aPair_{}_{}".format(a1_name, c1_name), CP_a1c1)
        pairPrint("aPair_{}_{}".format(a2_name, c1_name), CP_a2c1)
        pairPrint("aPair_{}_{}_{}".format(c1_name, a1_name, "OW"), SS_c1a1)
        pairPrint("aPair_{}_{}_{}".format(c1_name, a2_name, "OW"), SS_c1a2)
        pairPrint("aPair_{}_{}_{}".format(a1_name, c1_name, "OW"), SS_a1c1)
        pairPrint("aPair_{}_{}_{}".format(a2_name, c1_name, "OW"), SS_a2c1)
        pairPrint("aPairT_{}_{}".format(c1_name, a1_name), Pair_c1a1)
        pairPrint("aPairT_{}_{}".format(c1_name, a2_name), Pair_c1a2)
        pairPrint("aPairT_{}".format(c1_name), Pair_c1)
        pairPrint("aPairT_CP_{}".format(c1_name), Pair_CP_c1)
        pairPrint("aPairT_{}".format(a1_name), Pair_a1)
        pairPrint("aPairT_{}".format(a2_name), Pair_a2)
    else:
        CP_c1a1 = pair_dist(CP[:,:l_c1,0]); SS_c1a1 = pair_dist(SS[:,:l_c1,0])
        CP_c1a2 = pair_dist(CP[:,:l_c1,1]); SS_c1a2 = pair_dist(SS[:,:l_c1,1])
        Pair_c1a1 = pair_dist(CP[:,:l_c1,0] + SS[:,:l_c1,0])
        Pair_c1a2 = pair_dist(CP[:,:l_c1,1] + SS[:,:l_c1,1])
        Pair_c1 = pair_dist(CP[:,:l_c1,0] + CP[:,:l_c1,1] + SS[:,:l_c1,0] + SS[:,:l_c1,1]); Pair_CP_c1 = pair_dist(CP[:,:l_c1,0] + CP[:,:l_c1,1])

        CP_c2a1 = pair_dist(CP[:,l_c1:l_c1+l_c2,0]); SS_c2a1 = pair_dist(SS[:,l_c1:l_c1+l_c2,0])
        CP_c2a2 = pair_dist(CP[:,l_c1:l_c1+l_c2,1]); SS_c2a2 = pair_dist(SS[:,l_c1:l_c1+l_c2,1])
        Pair_c2a1 = pair_dist(CP[:,l_c1:l_c1+l_c2,0] + SS[:,l_c1:l_c1+l_c2,0])
        Pair_c2a2 = pair_dist(CP[:,l_c1:l_c1+l_c2,1] + SS[:,l_c1:l_c1+l_c2,1])
        Pair_c2 = pair_dist(CP[:,l_c1:l_c1+l_c2,0] + CP[:,l_c1:l_c1+l_c2,1] + SS[:,l_c1:l_c1+l_c2,0] + SS[:,l_c1:l_c1+l_c2,1]); Pair_CP_c2 = pair_dist(CP[:,l_c1:l_c1+l_c2,0] + CP[:,l_c1:l_c1+l_c2,1])

        CP_a1c1 = pair_dist(CP[:,l_c1+l_c2:l_c1+l_c2+l_a1,0]); SS_a1c1 = pair_dist(SS[:,l_c1+l_c2:l_c1+l_c2+l_a1,0])
        CP_a1c2 = pair_dist(CP[:,l_c1+l_c2:l_c1+l_c2+l_a1,1]); SS_a1c2 = pair_dist(SS[:,l_c1+l_c2:l_c1+l_c2+l_a1,1])
        Pair_a1c1 = pair_dist(CP[:,l_c1+l_c2:l_c1+l_c2+l_a1,0] + SS[:,l_c1+l_c2:l_c1+l_c2+l_a1,0])
        Pair_a1c2 = pair_dist(CP[:,l_c1+l_c2:l_c1+l_c2+l_a1,1] + SS[:,l_c1+l_c2:l_c1+l_c2+l_a1,1])
        Pair_a1 = pair_dist(CP[:,l_c1+l_c2:l_c1+l_c2+l_a1,0] + CP[:,l_c1+l_c2:l_c1+l_c2+l_a1,1] + SS[:,l_c1+l_c2:l_c1+l_c2+l_a1,0] + SS[:,l_c1+l_c2:l_c1+l_c2+l_a1,1]);  Pair_CP_a1 = pair_dist(CP[:,l_c1+l_c2:l_c1+l_c2+l_a1,0] + CP[:,l_c1+l_c2:l_c1+l_c2+l_a1,1])

        CP_a2c1 = pair_dist(CP[:,l_c1+l_c2+l_a1:,0]); SS_a2c1 = pair_dist(SS[:,l_c1+l_c2+l_a1:,0])
        CP_a2c2 = pair_dist(CP[:,l_c1+l_c2+l_a1:,1]); SS_a2c2 = pair_dist(SS[:,l_c1+l_c2+l_a1:,1])
        Pair_a2c1 = pair_dist(CP[:,l_c1+l_c2+l_a1:,0] + SS[:,l_c1+l_c2+l_a1:,0])
        Pair_a2c2 = pair_dist(CP[:,l_c1+l_c2+l_a1:,1] + SS[:,l_c1+l_c2+l_a1:,1])
        Pair_a2 = pair_dist(CP[:,l_c1+l_c2+l_a1:,0] + CP[:,l_c1+l_c2+l_a1:,1] + SS[:,l_c1+l_c2+l_a1:,0] + SS[:,l_c1+l_c2+l_a1:,1]); Pair_CP_a2 = pair_dist(CP[:,l_c1+l_c2+l_a1:,0] + CP[:,l_c1+l_c2+l_a1:,1])

        pairPrint("aPair_{}_{}".format(c1_name, a1_name), CP_c1a1)
        pairPrint("aPair_{}_{}".format(c1_name, a2_name), CP_c1a2)
        pairPrint("aPair_{}_{}".format(c2_name, a1_name), CP_c2a1)
        pairPrint("aPair_{}_{}".format(c2_name, a2_name), CP_c2a2)
        pairPrint("aPair_{}_{}".format(a1_name, c1_name), CP_a1c1)
        pairPrint("aPair_{}_{}".format(a1_name, c2_name), CP_a1c2)
        pairPrint("aPair_{}_{}".format(a2_name, c1_name), CP_a2c1)
        pairPrint("aPair_{}_{}".format(a2_name, c2_name), CP_a2c2)
        pairPrint("aPair_{}_{}_{}".format(c1_name, a1_name, "OW"), SS_c1a1)
        pairPrint("aPair_{}_{}_{}".format(c1_name, a2_name, "OW"), SS_c1a2)
        pairPrint("aPair_{}_{}_{}".format(c2_name, a1_name, "OW"), SS_c2a1)
        pairPrint("aPair_{}_{}_{}".format(c2_name, a2_name, "OW"), SS_c2a2)
        pairPrint("aPair_{}_{}_{}".format(a1_name, c1_name, "OW"), SS_a1c1)
        pairPrint("aPair_{}_{}_{}".format(a1_name, c2_name, "OW"), SS_a1c2)
        pairPrint("aPair_{}_{}_{}".format(a2_name, c1_name, "OW"), SS_a2c1)
        pairPrint("aPair_{}_{}_{}".format(a2_name, c2_name, "OW"), SS_a2c2)
        pairPrint("aPairT_{}_{}".format(c1_name, a1_name), Pair_c1a1)
        pairPrint("aPairT_{}_{}".format(c1_name, a2_name), Pair_c1a2)
        pairPrint("aPairT_{}".format(c1_name), Pair_c1)
        pairPrint("aPairT_CP_{}".format(c1_name), Pair_CP_c1)
        pairPrint("aPairT_{}_{}".format(c2_name, a1_name), Pair_c2a1)
        pairPrint("aPairT_{}_{}".format(c2_name, a2_name), Pair_c2a2)
        pairPrint("aPairT_{}".format(c2_name), Pair_c2)
        pairPrint("aPairT_CP_{}".format(c2_name), Pair_CP_c2)
        pairPrint("aPairT_{}_{}".format(a1_name, c1_name), Pair_a1c1)
        pairPrint("aPairT_{}_{}".format(a1_name, c2_name), Pair_a1c2)
        pairPrint("aPairT_{}".format(a1_name), Pair_a1)
        pairPrint("aPairT_CP_{}".format(a1_name), Pair_CP_a1)
        pairPrint("aPairT_{}_{}".format(a2_name, c1_name), Pair_a2c1)
        pairPrint("aPairT_{}_{}".format(a2_name, c2_name), Pair_a2c2)
        pairPrint("aPairT_{}".format(a2_name), Pair_a2)
        pairPrint("aPairT_CP_{}".format(a2_name), Pair_CP_a2)



    # Deletes the temporary h5py file
    os.remove('/tmp/Cluster.hdf5')

if __name__ == "__main__":
    main(trj_file, top_file, c1_name, c2_name, a1_name, a2_name, coord_c1a1, coord_c1a2, coord_c2a1, coord_c2a2, coord_c1W, coord_c2W, coord_a1W, coord_a2W, t_min, t_max, step, nt)
