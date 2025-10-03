#!/bin/bash

drc=$1
offset=$2
threads=$3

if [ ! -f "md.gro" ]; then
    if [ -f "md.cpt" ]; then
        ibrun -n $threads -o $offset mdrun_mpi -deffnm md -cpi md.cpt
    else
        ibrun -n 1 -o $offset gmx grompp -f ../F.mdp -c ../../$drc/md.gro -p ../../$drc/topol.top -o md -maxwarn 1
        ibrun -n $threads -o $offset mdrun_mpi -deffnm md
    fi
    rm *#
fi
