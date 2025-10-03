#!/bin/bash

c1_name=$1
c2_name=$2
a1_name=$3
a2_name=$4

N1=$5
N2=$6
NSum=$7
nwater=4000

offset=$8
threads=$9



if [[ "$c1_name" == "$c2_name" ]]; then
    n1cations=$NSum
    n2cations=0
else
    n1cations=$N1
    n2cations=$N2
fi

if [[ "$a1_name" == "$a2_name" ]]; then
    n1anions=$NSum
    n2anions=0
else
    n1anions=$N1
    n2anions=$N2
fi



cp ../../../mdp/start.top ./topol.top

check=0
if [[ $n1cations -ne 0 ]]; then
    seed=$(( RANDOM % 10000 ))
    echo $seed
    if [[ $check -eq 0 ]]; then
        ibrun -n 1 -o $offset gmx insert-molecules -f ../../../mdp/${c1_name}.gro -ci ../../../mdp/${c1_name}.gro -nmol $(($n1cations-1)) -box 5.5 5.5 5.5 -try 100 -seed $seed -o sys.gro
        check=1
    else
        ibrun -n 1 -o $offset gmx insert-molecules -f sys.gro -ci ../../../mdp/${c1_name}.gro -nmol $n1cations -try 100 -seed $seed -o sys.gro
    fi
fi

if [[ $n2cations -ne 0 ]]; then
    seed=$(( RANDOM % 10000 ))
    echo $seed
    if [[ $check -eq 0 ]]; then
        ibrun -n 1 -o $offset gmx insert-molecules -f ../../../mdp/${c2_name}.gro -ci ../../../mdp/${c2_name}.gro -nmol $(($n2cations-1)) -box 5.5 5.5 5.5 -try 100 -seed $seed -o sys.gro
        check=1
    else
        ibrun -n 1 -o $offset gmx insert-molecules -f sys.gro -ci ../../../mdp/${c2_name}.gro -nmol $n2cations -try 100 -seed $seed -o sys.gro
    fi
fi

if [[ $n1anions -ne 0 ]]; then
    seed=$(( RANDOM % 10000 ))
    echo $seed
    ibrun -n 1 -o $offset gmx insert-molecules -f sys.gro -ci ../../../mdp/${a1_name}.gro -nmol $n1anions -try 100 -seed $seed -o sys.gro
fi

if [[ $n2anions -ne 0 ]]; then
    seed=$(( RANDOM % 10000 ))
    echo $seed
    ibrun -n 1 -o $offset gmx insert-molecules -f sys.gro -ci ../../../mdp/${a2_name}.gro -nmol $n2anions -try 100 -seed $seed -o sys.gro
fi

if [[ $n1cations -ne 0 ]]; then
    echo "${c1_name}               "$n1cations >> topol.top
fi
if [[ $n2cations -ne 0 ]]; then
    echo "${c2_name}               "$n2cations >> topol.top
fi
if [[ $n1anions -ne 0 ]]; then
    echo "${a1_name}               "$n1anions >> topol.top
fi
if [[ $n2anions -ne 0 ]]; then
    echo "${a2_name}               "$n2anions >> topol.top
fi

ibrun -n 1 -o $offset gmx solvate -cp sys.gro -cs tip4p.gro -o solv.gro -p topol.top -maxsol $nwater
rm *# sys.gro



mkdir equil

ibrun -n 1 -o $offset gmx grompp -f ../../../mdp/em0.mdp -c solv.gro -p topol.top -o em
ibrun -n $threads -o $offset ibrun mdrun_mpi -deffnm em
rm step*

ibrun -n 1 -o $offset gmx grompp -f ../../../mdp/em.mdp -c em.gro -p topol.top -o em
ibrun -n $threads -o $offset mdrun_mpi -deffnm em
rm *# step*
mkdir equil/em
mv em* equil/em
rm equil/em/em.trr

ibrun -n 1 -o $offset gmx grompp -f ../../../mdp/eqx.mdp -c equil/em/em.gro -p topol.top -o eq -maxwarn 1
ibrun -n $threads -o $offset mdrun_mpi -deffnm eq

ibrun -n 1 -o $offset gmx grompp -f ../../../mdp/eq1.mdp -c equil/em/em.gro -p topol.top -o eq
ibrun -n $threads -o $offset mdrun_mpi -deffnm eq

ibrun -n 1 -o $offset gmx grompp -f ../../../mdp/eq3.mdp -c eq.gro -p topol.top -o eq -maxwarn 1
ibrun -n $threads -o $offset mdrun_mpi -deffnm eq
rm *#
mkdir equil/eq
mv eq* equil/eq
rm equil/eq/eq.trr

ibrun -n 1 -o $offset gmx grompp -f ../../../mdp/prd.mdp -c equil/eq/eq.gro -p topol.top -o md -maxwarn 1
ibrun -n $threads -o $offset mdrun_mpi -deffnm md
rm *#

#ibrun -n $threads -o $offset mdrun_mpi -deffnm md -cpi md.cpt
#rm *#
