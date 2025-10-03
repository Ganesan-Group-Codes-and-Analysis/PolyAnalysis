#!/bin/bash

Sys_list='1m_LI_K_CL' # Defines system of interest, here 1 molal LiCl/KCl
Sub_list='0.00 0.10 0.25 0.50 0.75 0.90 1.00' # Defines mole fractions of interest, here defined as mole fraction of salt 1, LiCl
F_list='0.02 0.04 0.06 0.08' # Defines the field strengths of interest based on the corresponding electric field strength, E = 0.02, 0.04, 0.06, 0.08
runs='5' # 5 repeat samples at each condition

# Defines the bounds for measuring the drift velocity from NEMD simulations
Dis_beg='10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000'
Dis_end='40000, 40000, 40000, 40000, 40000, 40000, 40000, 40000, 40000, 40000, 40000, 40000, 40000, 40000, 40000, 40000'

for i in $Sys_list; do
    drc=$i

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
    elif [[ $i = *'LI_CL_BR'* ]]; then
        c1_name='LI'
        c2_name='LI'
        a1_name='CL'
        a2_name='BR'
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

	count=0
	for j in $Sub_list; do
		drc=x_$j

		# Define displacement vs time file names for extracting drift velocities
		if [[ "${c1_name}" == "${c2_name}" ]]; then
			if [[ $j = '0.00' ]]; then
				Dis_filenames="dis_x_${c1_name}.xvg dis_x_${a2_name}.xvg dis_z_${c1_name}.xvg dis_z_${a2_name}.xvg"
				Dis_columns='1 1 1 1'
			elif [[ $j = '1.00' ]]; then
				Dis_filenames="dis_x_${c1_name}.xvg dis_x_${a1_name}.xvg dis_y_${c1_name}.xvg dis_y_${a1_name}.xvg"
				Dis_columns='1 1 1 1'
			else
				Dis_filenames="dis_x_${c1_name}.xvg dis_x_${a1_name}.xvg dis_x_${a2_name}.xvg dis_y_${c1_name}.xvg dis_y_${a1_name}.xvg dis_y_${a2_name}.xvg dis_z_${c1_name}.xvg dis_z_${a1_name}.xvg dis_z_${a2_name}.xvg"
				Dis_columns='1 1 1 1 1 1 1 1 1'
			fi
		elif [[ "${a1_name}" == "${a2_name}" ]]; then
			if [[ $j = '0.00' ]]; then
				Dis_filenames="dis_y_${c2_name}.xvg dis_y_${a1_name}.xvg dis_z_${c2_name}.xvg dis_z_${a1_name}.xvg"
				Dis_columns='1 1 1 1'
			elif [[ $j = '1.00' ]]; then
				Dis_filenames="dis_x_${c1_name}.xvg dis_x_${a1_name}.xvg dis_z_${c1_name}.xvg dis_z_${a1_name}.xvg"
				Dis_columns='1 1 1 1'
			else
				Dis_filenames="dis_x_${c1_name}.xvg dis_x_${c2_name}.xvg dis_x_${a1_name}.xvg dis_y_${c1_name}.xvg dis_y_${c2_name}.xvg dis_y_${a1_name}.xvg dis_z_${c1_name}.xvg dis_z_${c2_name}.xvg dis_z_${a1_name}.xvg"
				Dis_columns='1 1 1 1 1 1 1 1 1'
			fi
		else
			Dis_filenames="md1/dis_x_${c1_name}.xvg md1/dis_x_${c2_name}.xvg md1/dis_x_${a1_name}.xvg md1/dis_x_${a2_name}.xvg md1/dis_y_${c1_name}.xvg md1/dis_y_${c2_name}.xvg md1/dis_y_${a1_name}.xvg md1/dis_y_${a2_name}.xvg md2/dis_x_${c1_name}.xvg md2/dis_x_${c2_name}.xvg md2/dis_x_${a1_name}.xvg md2/dis_x_${a2_name}.xvg md2/dis_y_${c1_name}.xvg md2/dis_y_${c2_name}.xvg md2/dis_y_${a1_name}.xvg md2/dis_y_${a2_name}.xvg"
			Dis_columns='1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1'
		fi
		N="${N1[$count]} ${N2[$count]} ${NSum[$count]} 4000"

		# Measure drift velocities
		echo "Dis"
    	python3 ../dis_manual.py "$Dis_filenames" "$Dis_columns" $runs "$drc" "$F_list" "$Dis_beg" "$Dis_end"

		cd $drc

		# Calculate Onsager coefficients from drift velocities
		echo "Onsager"
		python3 ../../Onsager.py "$n" "$N" $runs

		cd ..

		((count++))
	done

	cd ..
    
done
