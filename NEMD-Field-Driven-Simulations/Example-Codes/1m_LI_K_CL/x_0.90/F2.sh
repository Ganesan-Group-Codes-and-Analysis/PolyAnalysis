#!/bin/bash

drc=$1
offset=$2
threads=$3

if [ ! -f "md1.gro" ]; then
    if [ -f "md1.cpt" ]; then
        ibrun -n $threads -o $offset mdrun_mpi -deffnm md1 -cpi md1.cpt
    else
        ibrun -n 1 -o $offset gmx grompp -f ../F1.mdp -c ../../$drc/md.gro -p ../../$drc/topol.top -o md1 -maxwarn 1
        ibrun -n $threads -o $offset mdrun_mpi -deffnm md1
    fi
    rm *#
fi

if [ ! -f "md2.gro" ]; then
    if [ -f "md2.cpt" ]; then
        ibrun -n $threads -o $offset mdrun_mpi -deffnm md2 -cpi md2.cpt
    else
        ibrun -n 1 -o $offset gmx grompp -f ../F2.mdp -c ../../$drc/md.gro -p ../../$drc/topol.top -o md2 -maxwarn 1
        ibrun -n $threads -o $offset mdrun_mpi -deffnm md2
    fi
    rm *#
fi
