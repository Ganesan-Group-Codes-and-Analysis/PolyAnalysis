#!/bin/bash

Sys_list='1m_LI_K_CL' # Defines system of interest, here 1 molal LiCl/KCl
Sub_list='0.00 0.10 0.25 0.50 0.75 0.90 1.00' # Defines mole fractions of interest, here defined as mole fraction of salt 1, LiCl
F_list='0.02 0.04 0.06 0.08' # Defines the field strengths of interest based on the corresponding electric field strength, E = 0.02, 0.04, 0.06, 0.08
Sample_list='1 2 3 4 5' # 5 repeat samples at each condition

for i in $Sys_list; do
    drc=$i
    echo $drc
    mkdir -p $drc

    cp equil_runner F_runner F2_runner F_job_runner F2_job_runner equil.sh F.sh F2.sh $drc

    cd $drc

    # Define the number of salt pairs for each species, N1 and N2, and the total number of salt pairs, NSum
    if [[ $i = *'1m_'* ]]; then
        N1=(0 7 18 36 54 65 72)
        N2=(72 65 54 36 18 7 0)
        NSum=(72 72 72 72 72 72 72)
    fi

    # Define the ions names for use in later scripts
    if [[ $i = *'LI_K_CL'* ]]; then
        c1_name='LI'
        c2_name='K'
        a1_name='CL'
        a2_name='CL'
    elif [[ $i = *'K_CL_BR'* ]]; then
        c1_name='K'
        c2_name='K'
        a1_name='CL'
        a2_name='BR'
    elif [[ $i = *'LI_K_BR_CL'* ]]; then
        c1_name='LI'
        c2_name='K'
        a1_name='BR'
        a2_name='CL'

        # For quaternary systems, the code does not work as written for single-salt (i.e., LiCl mole fraction = 0, 1)
        if [[ $i = *'1m_'* ]]; then
            N1=(7 18 36 54 65)
            N2=(65 54 36 18 7)
            NSum=(72 72 72 72 72)
        fi
    else
        break
    fi
    n="${c1_name} ${c2_name} ${a1_name} ${a2_name} SOL"

    ## Create systems, equilibrate, and run production
    #for k in $Sub_list; do
    #    rm equil_runner.*
    #    sbatch equil_runner $c1_name $c2_name $a1_name $a2_name "${N1[*]}" "${N2[*]}" "${NSum[*]}" $k "$Sample_list"
    #done

    ## Check if equil_runner succesfully ran the entire production for all systems
    ## Printed number should be 50,000,000 steps or 100,000 ps
    #for j in $Sub_list; do
    #    for k in $Sample_list; do
    #        drc=x_$j/sample_$k
    #        echo $drc && grep "Writing checkpoint" $drc/md.log | tail -1
    #    done
    #done

    # NEMD Simulations
    count=0
    for j in $Sub_list; do
        drc=x_$j
        mkdir -p $drc
        cp F_runner F2_runner F_job_runner F2_job_runner F.sh F2.sh $drc
        cd $drc
    
        for l in $F_list; do
            drc=F_$l
            mkdir -p $drc
        
            ## Check if F_runner succesfully ran the entire production for all systems
            ## Printed number should be 25,000,000 steps or 50,000 ps
            #for k in $Sample_list; do
            #    drc=F_$l/sample_$k
            #    grep "Writing checkpoint" $drc/mdc.log | tail -1
            #    echo $drc && grep "Writing checkpoint" $drc/mda.log | tail -1
            #done
        done
    
        # Create NEMD GROMACS .mdp files
        # Specifically, fills in the acc-grps and accelerate fields in the .mdp file to apply the artificial fields
        N="${N1[$count]} ${N2[$count]} ${NSum[$count]} 4000"
        python3 ../../mdp.py "$n" "$N" ../../mdp "$F_list"
    
        ## NEMD Simulations
        #if [[ "$a1_name" != "$a2_name" ]] && [[ "$c1_name" != "$c2_name" ]]; then
        #    rm F2_runner.*
        #    sbatch F2_runner "$F_list" "$Sample_list"
        #else
        #    rm F_runner.*
        #    sbatch F_runner "$F_list" "$Sample_list"
        #fi
    
        ### NEMD Post Analsis -> Displacements
        #if [[ "$a1_name" != "$a2_name" ]] && [[ "$c1_name" != "$c2_name" ]]; then
        #    rm F2_job_runner.*
        #    sbatch F2_job_runner $c1_name $c2_name $a1_name $a2_name $j "$F_list" "$Sample_list"
        #else
        #    rm F_job_runner.*
        #    sbatch F_job_runner $c1_name $c2_name $a1_name $a2_name "$F_list" "$Sample_list"
        #fi
    
        cd ..
    
        ((count++))
    done

    cd ..
done
