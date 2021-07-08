#!/opt/anaconda/envs/DL_CPU/bin/python

#### Authors: Liliana C. Gallegos and Juan V. Alegre-Requena ####
### For any questions, contact: LilianaC.Gallegos@colostate.edu or juanvi89@hotmail.com ###
from __future__ import print_function
import os, sys, glob, shutil, time, math
import re
import numpy as np
import pandas as pd
import json
import argparse
import subprocess
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
# import dbstep.Dbstep as dbstep

possible_atoms = ["", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si",
                 "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
                 "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
                 "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm",
                 "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
                 "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu",
                 "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
                 "Rg", "Uub", "Uut", "Uuq", "Uup", "Uuh", "Uus", "Uuo"]

def detectSP_planar(file,dihedral_cutoff,args):
    discard = False
    sdf_name = file.split('.')[0]+'.sdf'
    # converts xyz into sdf and then into mol
    cmd_obabel = ['obabel', '-ixyz', file, '-osdf', '-O', sdf_name]
    subprocess.run(cmd_obabel)
    mol_load = Chem.SDMolSupplier(sdf_name, removeHs=False)

    try:
        # we use the only mol generated (since there is only 1 conformer in each xyz file)
        mol = mol_load[0]

    	# Finds the M atom and gets the atom types and indexes of all its neighbours
    	# the filter is compatible with molecules that do not contain Ir (always passing)
        metal_idx = None
        neigh_idx = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == possible_atoms.index(args.SP_metal):
                metal_idx = atom.GetIdx()
                for x in atom.GetNeighbors():
                    neigh_idx.append(x.GetIdx())

        # discards complexes with more than 4 neighbours
        if len(neigh_idx) > 4:
            discard = True

    	# I need to get the only 3D conformer generated in that mol object for rdMolTransforms
        mol_conf = mol.GetConformer(0)

        # if the absolute dihedral angle of the 4 atoms bonding the square-planar metal center is greater
        # than the threshold, the compound will be discarded
        try:
            dihedral_angle_list = []
            dihedral_angle_list.append(abs(rdMolTransforms.GetDihedralDeg(mol_conf,neigh_idx[0],neigh_idx[1],neigh_idx[2],neigh_idx[3])))
            dihedral_angle_list.append(abs(rdMolTransforms.GetDihedralDeg(mol_conf,neigh_idx[0],neigh_idx[2],neigh_idx[1],neigh_idx[3])))
            dihedral_angle_list.append(abs(rdMolTransforms.GetDihedralDeg(mol_conf,neigh_idx[0],neigh_idx[3],neigh_idx[1],neigh_idx[2])))

            # use only the bigger value
            dihedral_angle = max(dihedral_angle_list)

            if dihedral_angle > 90:
                dihedral_angle = 180 - dihedral_angle

            if dihedral_angle >= dihedral_cutoff:
                discard = True

        except IndexError:
            error_SP_name = file.split('.')[0]
            error_SP_file = open(f'Error_SP_{error_SP_name}.txt',"w")
            error_SP_file.write(f"Less than 4 neighbours were detected for this squareplanar complex. Please, check the structure manually.") # n of atoms
            error_SP_file.close()

    except AttributeError:
        error_SP_name = file.split('.')[0]
        error_SP_file = open(f'Error_SP_{error_SP_name}.txt',"w")
        error_SP_file.write(f"OpenBabel could not generate a mol object for this squareplanar complex. Please, check the structure manually.") # n of atoms
        error_SP_file.close()

    return discard

## Main code:
def main():
    parser = argparse.ArgumentParser(description='The script analysis ...')
    parser.add_argument('--xtb', default="False", help="Runs XTB and gives XTB parameters: Total Energy (Eh), HOMO-LUMO gap (eV), Atom charge, and Wiberg bond order. \nThe convert option converts log into xyz files. \nOptions: run, convert, analyze, boltz, SP_filter.")
    parser.add_argument('--stacksize', default="1G", help="To specify the stack size available for the xTB calculation.")
    parser.add_argument('--atom', default="False", help="To specify output for selected atom. Option: atom number or element.")
    parser.add_argument('--v_bur', default=False, action="store_true", help="Calculates the buried volume of the atom selected in --atom.")
    parser.add_argument('--csv', default=False, action="store_true", help="Prints out csv file of collected data.")
    parser.add_argument('--gfn_version', default='2', help="To specify version of GFN used in --gfn.")
    parser.add_argument('--chrg', default="auto", help="To specify charge for the xTB calculation. Option: auto (read from the log files) or any number.")
    parser.add_argument('--uhf', default="auto", help="To specify number of unpaired e for the xTB calculation. Option: auto (read from the log files) or any number.")
    parser.add_argument('--iterations', default='250', help="To specify the number of iterations in single-point calculations.")
    parser.add_argument('--opt_precision', default='tight', help="To specify the precision level of the optimization process.")
    parser.add_argument('--acc_xtb', default='0.2', help="To specify the accuracy for single-point calculations.")
    parser.add_argument('--solvent_xtb', default='gas_phase', help="To specify the solvent for all the xTB calculations.")
    parser.add_argument('--solvent_grid', default='tight', help="To specify the grid level for the solvent model in all the xTB calculations.")
    parser.add_argument('--etemp', default=300, help="Electronic temperature used in the xTB calculations.")
    parser.add_argument('--etemp_fod', default=5000, help="Electronic temperature used in the xTB FOD calculations (it should be quite high).")
    parser.add_argument('--input', default="log", help="To specify the type of input files used. Option: log, com or xyz.")
    parser.add_argument('--keep_files', default=False, action="store_true", help="Keep the raw xTB files on a separate folder.")
    parser.add_argument('--discard_freqs', default=False, action="store_true", help="Discard the raw xTB files containing frequency information (large files).")
    parser.add_argument('--discard_opt', default=False, action="store_true", help="Discard the raw xTB files containing optimization information (large files).")
    parser.add_argument('--get_freq', default=False, action="store_true", help="Look for the freq with highest displacement of the selected atom in the --atom option.")
    parser.add_argument('--imag_freq_cutoff', default=50, help="Cut-off to discard optimizations with imaginary frequencies (default: i50 cm-1).")
    parser.add_argument('--suffix', default='_xTB', help="Suffix for the xTB files (include '_' in the suffix, i.e. '_suffix').")
    parser.add_argument('--pyconf_input', default=None, help="Name of the file used initially with pyCONFORT (takes an optional argument in case you want to include multiple solvents in the optimizations (Solvent column from the csv) or do Boltzmann averaging (names of each species in the code_name column)).")
    parser.add_argument('--temp', default=298.15, help="Temperature used in the Boltzmann weighting process.")
    parser.add_argument('--SP_metal', default='', help="Metal of the squareplanar complex to be analyzed for planarity.")


    parser.add_argument('--verbose', default=False, action="store_true", help="Prints out.")

    args = parser.parse_args()

    # saves working directory
    w_dir = os.getcwd()

    # measures execution time
    start_time = time.time()

    # converts the input files into xyz files
    if args.xtb.upper() == "RUN":
        if args.input == 'log' or args.input == 'com':

            files = glob.glob(f'*.{args.input}')
            count_files = len(files)

            for file in files:
                os.chdir(w_dir)
                xyz_filename = file.split('.')[0]
                # convert file into xyz after reading charge and multiplicity
                if args.input == 'com':
                    lines = open(file,"r").readlines()
                    emptylines=[]
                    for i, line in enumerate(lines):
                        if len(line.strip()) == 0:
                            emptylines.append(i)

                    # assigning the charges and multiplicity
                    charge_input = lines[(emptylines[1]+1)].split()[0]
                    multiplicity_input = lines[(emptylines[1]+1)].split()[1]

                    xyzfile = open(f'{xyz_filename}.xyz',"w")
                    xyzfile.write(f'{emptylines[2] - (emptylines[1]+2)}\n') # n of atoms
                    xyzfile.write(f'{xyz_filename}, Charge = {charge_input}, Multiplicity = {multiplicity_input}, Unpaired e = {int(multiplicity_input)-1}\n') # title
                    for i in range((emptylines[1]+2), emptylines[2]): # coordinates
                        xyzfile.write(lines[i])
                    xyzfile.close()

                if args.input == 'log':
                    lines = open(file,"r").readlines()
                    NATOMS,stand_or = 0,0
                    ATOMTYPES, CARTESIANS  = [],[]
                    # forward short loop to detect charge and multiplicity
                    for i in range(0,len(lines)):
                        if lines[i].find("Charge = ") > -1:
                            charge_input = int(lines[i].split()[2])
                            multiplicity_input = int(lines[i].split()[5].rstrip("\n"))
                            break

                    stop_termination = 0
                    stop_get_details_stand_or, stop_get_details_dis_rot = 0,0
                    TERMINATION = ""
                    # reversed loop to detect final geometry (i.e. after molecular optimization)
                    for i in reversed(range(0,len(lines))):
                        if stop_termination == 0:
                            if lines[i].find("Normal termination") > -1:
                                TERMINATION = "normal"
                                stop_termination += 1
                        else:
                            # stops the loop when it finds the last set of coordinates
                            if stop_get_details_stand_or == 1 and stop_get_details_dis_rot == 1:
                                break
                            # Sets where the final coordinates are inside the file
                            elif stop_get_details_dis_rot !=1 and (lines[i].find("Distance matrix") > -1 or lines[i].find("Rotational constants") >-1) :
                                if lines[i-1].find("-------") > -1:
                                    dist_rot_or = i
                                    stop_get_details_dis_rot += 1
                            elif (lines[i].find("Standard orientation") > -1 or lines[i].find("Input orientation") > -1) and stop_get_details_stand_or !=1 :
                                stand_or = i
                                NATOMS = dist_rot_or-i-6
                                stop_get_details_stand_or += 1

                    # short loop to store atom types and coordinates
                    if TERMINATION == "normal":
                        for i in range(stand_or+5,stand_or+5+NATOMS):
                            massno = int(lines[i].split()[1])
                            if massno < len(possible_atoms):
                                atom_symbol = possible_atoms[massno]
                            else:
                                atom_symbol = "XX"
                            ATOMTYPES.append(atom_symbol)
                            CARTESIANS.append([float(lines[i].split()[3]), float(lines[i].split()[4]), float(lines[i].split()[5])])

                        # creates xyz file
                        xyzfile = open(f'{xyz_filename}.xyz',"w")
                        xyzfile.write(f'{NATOMS}\n')
                        xyzfile.write(f'{xyz_filename}, Charge = {charge_input}, Multiplicity = {multiplicity_input}, Unpaired e = {int(multiplicity_input)-1}\n') # title
                        for atom in range(0,NATOMS):
                            xyzfile.write('{0:>2} {1:12.8f} {2:12.8f} {3:12.8f}'.format(ATOMTYPES[atom], CARTESIANS[atom][0],  CARTESIANS[atom][1],  CARTESIANS[atom][2]))
                            xyzfile.write("\n")
                        xyzfile.close()

                    else:
                        print(f'{file} did not finish correctly, check this calculation and try again!')
                        sys.exit()

                # move the initial input files into a folder
                destination = f'{w_dir}/{args.input}_files'
                if not os.path.isdir(destination):
                    os.makedirs(destination)

                source = f'{w_dir}/{file}'
                try:
                    shutil.move(source, destination)
                except (FileExistsError,shutil.Error):
                    os.chdir(destination)
                    os.remove(source.split('/')[-1])
                    shutil.move(source, destination)

                # creates a folder to work with the xyz files
                destination_xyz = f'{w_dir}/initial_xyz_files'
                if not os.path.isdir(destination_xyz):
                    os.makedirs(destination_xyz)

                source_xyz = f'{w_dir}/{xyz_filename}.xyz'
                try:
                    shutil.move(source_xyz, destination_xyz)
                except (FileExistsError,shutil.Error):
                    os.chdir(destination_xyz)
                    os.remove(source_xyz.split('/')[-1])
                    shutil.move(source_xyz, destination_xyz)

        # run xTB from the folder containing the xyz files
        destination_xyz = f'{w_dir}/initial_xyz_files'
        if args.input == 'xyz':
            destination_xyz = w_dir
        if not os.path.isdir(destination_xyz):
            os.makedirs(destination_xyz)
        os.chdir(destination_xyz)

        # loads all the xyz and reads charges/number of unpaired e if the options are set to auto
        files_xyz = glob.glob('*.xyz')
        count_files = len(files_xyz)

        unpaired_e, charge_xtb = None, None
        for file in files_xyz:
            if args.uhf != 'auto':
                unpaired_e = args.uhf
            else:
                lines = open(file,"r").readlines()
                unpaired_e = lines[1].split()[-1]

            if args.chrg != 'auto':
                charge_xtb = args.chrg
            else:
                lines = open(file,"r").readlines()
                charge_xtb = lines[1].split()[3].split(',')[0]

            if unpaired_e is None:
                unpaired_e = 0
            if charge_xtb is None:
                charge_xtb = 0

            # reads the solvent from a CSV file used as the input file for previous pyCONFORT calculations
            solvent = args.solvent_xtb
            valid_solvent = True
            if args.pyconf_input != None:
                os.chdir(w_dir)
                solvent = None
                df_solvent = pd.read_csv(args.pyconf_input)
                for i,molecule in enumerate(df_solvent['code_name']):
                        if file.find(molecule+'_') > -1 or file == molecule:
                            solvent = df_solvent['Solvent'][i]
                            break

                # changes abbreviations to valid solvent names
                if solvent.upper() == 'MECN' or solvent.upper() == 'CH3CN' or solvent.upper() == 'NPRCN' or solvent.upper() == 'BUTYRONITRILE' or solvent.upper() == 'BUTANONITRILE' or solvent.upper() == 'BUTANENITRILE':
                    solvent = 'Acetonitrile'
                if solvent.upper() == 'DCM' or solvent.upper() == 'DICHLOROMETHANE' or solvent.upper() == 'METHYLENE CHLORIDE':
                    solvent = 'CH2Cl2'
                if solvent.upper() == 'CHLOROFORM' or solvent.upper() == 'TRICHLOROMETHANE':
                    solvent = 'CHCl3'
                if solvent.upper() == 'TETRAHYDROFURAN' or solvent.upper() == 'OXOLANE':
                    solvent = 'THF'
                if solvent.upper() == 'METHF': # MeTHF is modeled as THF as an approximation
                    solvent = 'THF'
                if solvent.upper() == 'MEOH' or solvent.upper() == 'METHYL ALCOHOL':
                    solvent = 'Methanol'
                if solvent.upper() == 'ETOH' or solvent.upper() == 'ETHYL ALCOHOL': # EtOH is modeled as MeOH as an approximation
                    solvent = 'Methanol'
                if solvent.upper() == 'MENO2':
                    solvent = 'Nitromethane'
                if solvent.upper() == 'PHME' or solvent.upper() == 'MEPH':
                    solvent = 'Toluene'
                if solvent.upper() == 'H2O':
                    solvent = 'Water'
                if solvent.upper() == 'PH':
                    solvent = 'Benzene'
                if solvent.upper() == 'DIMETHYLSULFOXIDE' or solvent.upper() == 'DIMETHYL SULFOXIDE':
                    solvent = 'DMSO'
                if solvent.upper() == 'DIMETHYLFORMAMIDE' or solvent.upper() == 'DIMETHYL FORMAMIDE':
                    solvent = 'DMF'
                if solvent.upper() == 'ETOAC' or solvent.upper() == 'ACOET' or solvent.upper() == 'ETHYL ACETATE' or solvent.upper() == 'ETHYL ETHANOATE':
                    solvent = 'Ethylacetate'
                if solvent.upper() == 'PROPANONE':
                    solvent = 'Acetone'
                if solvent.upper() == '1,4-DIOXANE':
                    solvent = 'Dioxane'

                # if the solvent is not compatible with the ALPB model, xTB will not optimize the molecule
                if solvent == None:
                    valid_solvent = False

                os.chdir(destination_xyz)

            if valid_solvent:
                # creates a bash script to run xTB on all the xyz files
                # define the xtb path
                xtb_path = '/usr/local/xtb/bin/xtb'

                # write input file to create the external options file for xTB
                xtbfile = open(f'xtb.inp',"w")
                xtbfile.write('$write\n')
                xtbfile.write('json=true\n')
                if args.solvent_xtb != 'gas_phase' or args.pyconf_input is not None:
                    xtbfile.write('gbsa=true\n')
                    xtbfile.write('$gbsa\n')
                    xtbfile.write(f'gbsagrid={args.solvent_grid}')

                xtbfile.close()

                # create an extra file only with solvent properties (without the option to create json files)
                if args.solvent_xtb != 'gas_phase' or args.pyconf_input is not None:
                    xtbfile = open(f'xtb_solvent.inp',"w")
                    xtbfile.write('$write\n')
                    xtbfile.write('gbsa=true\n')
                    xtbfile.write('$gbsa\n')
                    xtbfile.write(f'gbsagrid={args.solvent_grid}')
                    xtbfile.close()

                xtb_file_name = file.split('.')[0]
                sh_file = open(f'run_xtb.sh',"w")
                sh_file.write(f'#!/bin/bash\n')
                sh_file.write(f'export OMP_STACKSIZE={args.stacksize}\n')
                sh_file.write(f'runxtb={xtb_path}\n')
                sh_file.write(f'echo "RUNNING {file} WITH xTB"\n')

                if args.solvent_xtb == 'gas_phase' and args.pyconf_input is None:
                    if int(unpaired_e) > 0:
                        sh_file.write(f'$runxtb {file} --opt {args.opt_precision} --acc {args.acc_xtb} --gfn {args.gfn_version} --iterations {args.iterations} --chrg {charge_xtb} --uhf {unpaired_e} --etemp {args.etemp} > {xtb_file_name}{args.suffix}.opt && $runxtb xtbopt.xyz --pop --wbo --acc {args.acc_xtb} --gfn {args.gfn_version} --chrg {charge_xtb} --uhf {unpaired_e} --etemp {args.etemp} --input xtb.inp > {xtb_file_name}{args.suffix}.out && mv xtbout.json {xtb_file_name}{args.suffix}.json && mv wbo {xtb_file_name}{args.suffix}.wbo\n')
                        # for pop, we use GFN1 by default since gfn2 doesn't print Mulliken and CM5 charges
                        sh_file.write(f'$runxtb xtbopt.xyz --pop --gfn 1 --chrg {charge_xtb} --acc {args.acc_xtb} --uhf {unpaired_e} --etemp {args.etemp} > {xtb_file_name}{args.suffix}.gfn1\n')
                        sh_file.write(f'$runxtb xtbopt.xyz --vomega --gfn {args.gfn_version} --chrg {charge_xtb} --acc {args.acc_xtb} --uhf {unpaired_e} --etemp {args.etemp} > {xtb_file_name}{args.suffix}.omega\n')
                        sh_file.write(f'$runxtb xtbopt.xyz --vfukui --gfn {args.gfn_version} --chrg {charge_xtb} --acc {args.acc_xtb} --uhf {unpaired_e} --etemp {args.etemp} > {xtb_file_name}{args.suffix}.fukui\n')
                        sh_file.write(f'$runxtb xtbopt.xyz --fod --gfn {args.gfn_version} --chrg {charge_xtb} --acc {args.acc_xtb} --uhf {unpaired_e} --etemp {args.etemp_fod} > {xtb_file_name}{args.suffix}.fod\n')
                        sh_file.write(f'$runxtb xtbopt.xyz --hess --gfn {args.gfn_version} --chrg {charge_xtb} --acc {args.acc_xtb} --uhf {unpaired_e} --etemp {args.etemp} > {xtb_file_name}{args.suffix}.thermo && mv g98.out {xtb_file_name}{args.suffix}.freqs\n')
                        # calculates T1 to S0 vertical gap
                        if int(unpaired_e) == 2:
                            sh_file.write(f'$runxtb xtbopt.xyz --gfn {args.gfn_version} --chrg {charge_xtb} --uhf 0 --acc {args.acc_xtb} --etemp {args.etemp} > {xtb_file_name}{args.suffix}.T1toS0gap\n')
                            sh_file.write(f'$runxtb xtbopt.xyz --fod --gfn {args.gfn_version} --chrg {charge_xtb} --uhf 0 --acc {args.acc_xtb} --etemp {args.etemp_fod} > {xtb_file_name}{args.suffix}.fodS0\n')

                    else:
                        sh_file.write(f'$runxtb {file} --opt {args.opt_precision} --acc {args.acc_xtb} --gfn {args.gfn_version} --iterations {args.iterations} --chrg {charge_xtb} --etemp {args.etemp} > {xtb_file_name}{args.suffix}.opt && $runxtb xtbopt.xyz --pop --wbo --acc {args.acc_xtb} --gfn {args.gfn_version} --chrg {charge_xtb} --etemp {args.etemp} --input xtb.inp > {xtb_file_name}{args.suffix}.out && mv xtbout.json {xtb_file_name}{args.suffix}.json && mv wbo {xtb_file_name}{args.suffix}.wbo\n')
                        # for pop, we use GFN1 by default since gfn2 doesn't print Mulliken and CM5 charges
                        sh_file.write(f'$runxtb xtbopt.xyz --pop --gfn 1 --chrg {charge_xtb} --acc {args.acc_xtb} --etemp {args.etemp} > {xtb_file_name}{args.suffix}.gfn1\n')
                        sh_file.write(f'$runxtb xtbopt.xyz --vomega --gfn {args.gfn_version} --chrg {charge_xtb} --acc {args.acc_xtb} --etemp {args.etemp} > {xtb_file_name}{args.suffix}.omega\n')
                        sh_file.write(f'$runxtb xtbopt.xyz --vfukui --gfn {args.gfn_version} --chrg {charge_xtb} --acc {args.acc_xtb} --etemp {args.etemp} > {xtb_file_name}{args.suffix}.fukui\n')
                        sh_file.write(f'$runxtb xtbopt.xyz --fod --gfn {args.gfn_version} --chrg {charge_xtb} --acc {args.acc_xtb} --etemp {args.etemp_fod} > {xtb_file_name}{args.suffix}.fod\n')
                        sh_file.write(f'$runxtb xtbopt.xyz --hess --gfn {args.gfn_version} --chrg {charge_xtb} --acc {args.acc_xtb} --etemp {args.etemp} > {xtb_file_name}{args.suffix}.thermo && mv g98.out {xtb_file_name}{args.suffix}.freqs\n')
                        # calculates S0 to T1 vertical gap
                        sh_file.write(f'$runxtb xtbopt.xyz --gfn {args.gfn_version} --chrg {charge_xtb} --acc {args.acc_xtb} --uhf 2 --etemp {args.etemp} > {xtb_file_name}{args.suffix}.S0toT1gap\n')
                        sh_file.write(f'$runxtb xtbopt.xyz --fod --gfn {args.gfn_version} --chrg {charge_xtb} --acc {args.acc_xtb} --uhf 2 --etemp {args.etemp_fod} > {xtb_file_name}{args.suffix}.fodT1\n')

                else:
                        if int(unpaired_e) > 0:
                            sh_file.write(f'$runxtb {file} --opt {args.opt_precision} --acc {args.acc_xtb} --gfn {args.gfn_version} --iterations {args.iterations} --chrg {charge_xtb} --uhf {unpaired_e} --alpb {solvent} --etemp {args.etemp} --input xtb_solvent.inp > {xtb_file_name}{args.suffix}.opt && $runxtb xtbopt.xyz --pop --wbo --acc {args.acc_xtb} --gfn {args.gfn_version} --chrg {charge_xtb} --uhf {unpaired_e} --alpb {solvent} --etemp {args.etemp} --input xtb.inp > {xtb_file_name}{args.suffix}.out && mv xtbout.json {xtb_file_name}{args.suffix}.json && mv wbo {xtb_file_name}{args.suffix}.wbo\n')
                            # for pop, we use GFN1 by default since gfn2 doesn't print Mulliken and CM5 charges
                            sh_file.write(f'$runxtb xtbopt.xyz --pop --gfn 1 --chrg {charge_xtb} --acc {args.acc_xtb} --uhf {unpaired_e} --alpb {solvent} --etemp {args.etemp} --input xtb_solvent.inp > {xtb_file_name}{args.suffix}.gfn1\n')
                            sh_file.write(f'$runxtb xtbopt.xyz --vomega --gfn {args.gfn_version} --chrg {charge_xtb} --acc {args.acc_xtb} --uhf {unpaired_e} --alpb {solvent} --etemp {args.etemp} --input xtb_solvent.inp > {xtb_file_name}{args.suffix}.omega\n')
                            sh_file.write(f'$runxtb xtbopt.xyz --vfukui --gfn {args.gfn_version} --chrg {charge_xtb} --acc {args.acc_xtb} --uhf {unpaired_e} --alpb {solvent} --etemp {args.etemp} --input xtb_solvent.inp > {xtb_file_name}{args.suffix}.fukui\n')
                            sh_file.write(f'$runxtb xtbopt.xyz --fod --gfn {args.gfn_version} --chrg {charge_xtb} --acc {args.acc_xtb} --uhf {unpaired_e} --alpb {solvent} --etemp {args.etemp_fod} --input xtb_solvent.inp > {xtb_file_name}{args.suffix}.fod\n')
                            sh_file.write(f'$runxtb xtbopt.xyz --hess --gfn {args.gfn_version} --chrg {charge_xtb} --acc {args.acc_xtb} --uhf {unpaired_e} --alpb {solvent} --etemp {args.etemp} --input xtb_solvent.inp > {xtb_file_name}{args.suffix}.thermo && mv g98.out {xtb_file_name}{args.suffix}.freqs\n')
                            # calculates T1 to S0 vertical gap
                            if int(unpaired_e) == 2:
                                sh_file.write(f'$runxtb xtbopt.xyz --gfn {args.gfn_version} --chrg {charge_xtb} --acc {args.acc_xtb} --uhf 0 --alpb {solvent} --etemp {args.etemp} --input xtb_solvent.inp > {xtb_file_name}{args.suffix}.T1toS0gap\n')
                                sh_file.write(f'$runxtb xtbopt.xyz --fod --gfn {args.gfn_version} --chrg {charge_xtb} --acc {args.acc_xtb} --uhf 0 --alpb {solvent} --etemp {args.etemp_fod} --input xtb_solvent.inp > {xtb_file_name}{args.suffix}.fodS0\n')

                        else:
                            sh_file.write(f'$runxtb {file} --opt {args.opt_precision} --acc {args.acc_xtb} --gfn {args.gfn_version} --iterations {args.iterations} --chrg {charge_xtb} --alpb {solvent} --etemp {args.etemp} --input xtb_solvent.inp > {xtb_file_name}{args.suffix}.opt && $runxtb xtbopt.xyz --pop --wbo --acc {args.acc_xtb} --gfn {args.gfn_version} --chrg {charge_xtb} --alpb {solvent} --etemp {args.etemp} --input xtb.inp > {xtb_file_name}{args.suffix}.out && mv xtbout.json {xtb_file_name}{args.suffix}.json && mv wbo {xtb_file_name}{args.suffix}.wbo\n')
                            # for pop, we use GFN1 by default since gfn2 doesn't print Mulliken and CM5 charges
                            sh_file.write(f'$runxtb xtbopt.xyz --pop --gfn 1 --chrg {charge_xtb} --acc {args.acc_xtb} --alpb {solvent} --etemp {args.etemp} --input xtb_solvent.inp > {xtb_file_name}{args.suffix}.gfn1\n')
                            sh_file.write(f'$runxtb xtbopt.xyz --vomega --gfn {args.gfn_version} --chrg {charge_xtb} --acc {args.acc_xtb} --alpb {solvent} --etemp {args.etemp} --input xtb_solvent.inp > {xtb_file_name}{args.suffix}.omega\n')
                            sh_file.write(f'$runxtb xtbopt.xyz --vfukui --gfn {args.gfn_version} --chrg {charge_xtb} --acc {args.acc_xtb} --alpb {solvent} --etemp {args.etemp} --input xtb_solvent.inp > {xtb_file_name}{args.suffix}.fukui\n')
                            sh_file.write(f'$runxtb xtbopt.xyz --fod --gfn {args.gfn_version} --chrg {charge_xtb} --acc {args.acc_xtb} --alpb {solvent} --etemp {args.etemp_fod} --input xtb_solvent.inp > {xtb_file_name}{args.suffix}.fod\n')
                            sh_file.write(f'$runxtb xtbopt.xyz --hess --gfn {args.gfn_version} --chrg {charge_xtb} --acc {args.acc_xtb} --alpb {solvent} --etemp {args.etemp} --input xtb_solvent.inp > {xtb_file_name}{args.suffix}.thermo && mv g98.out {xtb_file_name}{args.suffix}.freqs\n')
                            # calculates S0 to T1 vertical gap
                            sh_file.write(f'$runxtb xtbopt.xyz --gfn {args.gfn_version} --chrg {charge_xtb} --acc {args.acc_xtb} --uhf 2 --alpb {solvent} --etemp {args.etemp} --input xtb_solvent.inp > {xtb_file_name}{args.suffix}.S0toT1gap\n')
                            sh_file.write(f'$runxtb xtbopt.xyz --fod --gfn {args.gfn_version} --chrg {charge_xtb} --acc {args.acc_xtb} --uhf 2 --alpb {solvent} --etemp {args.etemp_fod} --input xtb_solvent.inp > {xtb_file_name}{args.suffix}.fodT1\n')

                sh_file.write(f'mv xtbopt.xyz {xtb_file_name}{args.suffix}.xyz\n')

                sh_file.close()

                # command to give permissions and run the xTB script in Linux
                subprocess.call(f'chmod 777 {destination_xyz}/run_xtb.sh',shell=True)
                xtb_script = [f'{destination_xyz}/run_xtb.sh']
                subprocess.run(xtb_script)

                os.remove('run_xtb.sh')

                # move all the files from xTB to a folder and delete files that are not needed
                destination_raw = f'{w_dir}/xtb_files'
                if not os.path.isdir(destination_raw):
                    os.makedirs(destination_raw)

                files_raw = glob.glob('*.*')
                for extra_file in glob.glob('*'):
                    if extra_file not in files_raw:
                        files_raw.append(extra_file)

                extensions_keep = ['json', 'out', 'wbo', 'fukui', 'omega', 'gfn1', 'fod', 'thermo', 'T1toS0gap', 'S0toT1gap', 'fodS0', 'fodT1']

                if not args.discard_freqs:
                    extensions_keep.append('freqs')
                if not args.discard_opt:
                    extensions_keep.append('opt')

                for file_raw in files_raw:
                    if len(file_raw.split('.')) == 1:
                        try:
                            os.remove(file_raw)
                        except FileNotFoundError:
                            pass
                    elif file_raw.split('.')[1] in extensions_keep or file_raw == f'{xtb_file_name}{args.suffix}.xyz':
                        source_raw = f'{destination_xyz}/{file_raw}'
                        try:
                            shutil.move(source_raw, destination_raw)
                        except (FileExistsError,shutil.Error):
                            os.chdir(destination_raw)
                            os.remove(source_raw.split('/')[-1])
                            shutil.move(source_raw, destination_raw)
                    elif file_raw.split('.')[1] != 'xyz' or file_raw == 'xtbhess.xyz' or file_raw == 'xtblast.xyz':
                        try:
                            os.remove(file_raw)
                        except FileNotFoundError:
                            pass

        os.chdir(w_dir)

        # this is a dirty hack to use xtb_to_jason inside the xtb_files folder
        # TO DO: change the way xtb_to_json finds the files and runs
        source_json = f'{w_dir}/xtb_to_json.py'
        destination_xtb = f'{w_dir}/xtb_files'
        shutil.move(source_json, destination_xtb)

        os.chdir(destination_xtb)
        if args.get_freq:
            json_script = [f'python', f'{destination_xtb}/xtb_to_json.py','--get_freq']
        else:
            json_script = [f'python', f'{destination_xtb}/xtb_to_json.py']
        subprocess.run(json_script)

        # moves the xtb_to_json script back
        source_json_final = f'{destination_xtb}/xtb_to_json.py'
        shutil.move(source_json_final, w_dir)

        if not args.keep_files:
            # remove all the files that are not json files
            files_raw = glob.glob('*.*')+glob.glob('*')

            for file_raw in files_raw:
                if file_raw.split('.')[1] != 'json':
                    try:
                        os.remove(file_raw)
                    except FileNotFoundError:
                        pass

        os.chdir(w_dir)

    if args.xtb.upper() == "ANALYZE":
        os.chdir(w_dir)
        destination_xtb = f'{w_dir}/xtb_files'
        if not os.path.isdir(destination_xtb):
            print('The processed xTB files were not found, please run the code with "--xtb RUN" first!')
        os.chdir(destination_xtb)

        # list of molecular properties from xTB
        E_name, xtb_energy, ho_lu_gap, homos, lumos, homo_occs, lumo_occs, dipoles = [], [], [], [], [], [], [], []
        trans_dip, Fermi_lvl, tot_chrg, unp_e, free_E, ZPE, T1_S0_gap, S0_T1_gap = [], [], [], [], [], [], [], []
        electron_affinities, ionization_potentials, global_electrophilicities, smiles_strings = [], [], [], []
        ho_lu_gap_S0_SP, homos_S0_SP, lumos_S0_SP, homos_occ_S0_SP, lumos_occ_S0_SP, dipoles_S0_SP, Fermi_lvl_S0_SP = [], [], [], [], [], [], []
        ho_lu_gap_T1_SP, homos_T1_SP, lumos_T1_SP, homos_occ_T1_SP, lumos_occ_T1_SP, dipoles_T1_SP, Fermi_lvl_T1_SP = [], [], [], [], [], [], []
        total_C6AA, total_C8AA, total_alpha, total_fods, total_SASA = [], [], [], [], []
        total_C6AA_S0_SP, total_C8AA_S0_SP, total_alpha_S0_SP, total_fods_S0_SP = [], [], [], []
        total_C6AA_T1_SP, total_C8AA_T1_SP, total_alpha_T1_SP, total_fods_T1_SP = [], [], [], []

        # list of atomic properties from xTB
        xtb_fup, xtb_fun, xtb_fne, xtb_chrg, mulliken_chrg, cm5_chrg = [], [], [], [], [], []
        xtb_s_prop, xtb_p_prop, xtb_d_prop = [], [], []
        xtb_covCN, xtb_C6AA, xtb_alpha = [], [], []
        xtb_chrg_S0_SP, xtb_covCN_S0_SP, xtb_C6AA_S0_SP, xtb_alpha_S0_SP = [], [], [], []
        xtb_chrg_T1_SP, xtb_covCN_T1_SP, xtb_C6AA_T1_SP, xtb_alpha_T1_SP = [], [], [], []
        freq_cm, freq_disp_mod, freq_redmass, freq_IRintens = [], [], [], []
        freq_second_cm, freq_second_disp_mod, freq_second_redmass, freq_second_IRintens, termination, imag_freqs = [], [], [], [], [], []
        fod, s_prop_fods, p_prop_fods, d_prop_fods = [], [], [], []
        born_rad, SASA, h_bond = [], [], []
        fod_S0_SP, s_prop_fods_S0_SP, p_prop_fods_S0_SP, d_prop_fods_S0_SP = [], [], [], []
        fod_T1_SP, s_prop_fods_T1_SP, p_prop_fods_T1_SP, d_prop_fods_T1_SP = [], [], [], []

        # buried volume from dbstep
        v_bur = []

        json_files = glob.glob('*.json')
        count_json_files = len(json_files)
        for file in json_files:
            if len(json_files) == 0: print('\n   Error: No json file(s) found \n')
            else:
                file_name = os.path.basename(file)
                [name, format] = os.path.splitext(file_name)
                if args.verbose: print('\no ',name)

                # Collect XTB parameters
                f = open(file, 'r') # Opening JSON file
                data = json.loads(f.read()) # read file

                termination_ind = 'Normal'
                # these tests avoid optimizations with error terminations or invalid frequencies (requires optimizations with --get_freq)
                if args.get_freq:
                    if 'Termination' in data: termination_ind = data['Termination']
                    if 'Imag_freqs' in data: imag_freqs_ind = data['Imag_freqs']
                    # filter for imaginary freqs
                    try:
                        if imag_freqs_ind != 'NaN':
                            for imag_freq in imag_freqs_ind:
                                if abs(float(imag_freq)) > abs(float(args.imag_freq_cutoff)):
                                    termination_ind = 'Error_freqs'
                        else:
                            termination_ind = 'Error_freqs'
                    except UnboundLocalError:
                        print('No frequency information was found in the json file! Please, run disco-xtb.py (or xtb_to_json.py only) with --get_freq')

                if termination_ind == 'Normal':
                    # parse molecular properties from JSON
                    energy, homo_lumo_gap, homo, lumo, homo_occ, lumo_occ, electron_affinity, ionization_potential, global_electrophilicity = 0, 0, 0, 0, 0, 0, 0, 0, 0
                    transition_dipole, Fermi_level, total_charge, unpaired_e, free_energy, zero_point_E, T1toS0gap, S0toT1gap = 0,0,0,0,0,0,0,0
                    homo_lumo_S0_SP, homo_S0_SP, lumo_S0_SP, homo_occ_S0_SP, lumo_occ_S0_SP, dipole_module_S0_SP, Fermi_level_S0_SP = 0,0,0,0,0,0,0
                    homo_lumo_T1_SP, homo_T1_SP, lumo_T1_SP, homo_occ_T1_SP, lumo_occ_T1_SP, dipole_module_T1_SP, Fermi_level_T1_SP = 0,0,0,0,0,0,0
                    total_C6AA_ind, total_C8AA_ind, total_alpha_ind, total_fods_ind, total_SASA_ind = 0,0,0,0,0
                    total_C6AA_ind_S0_SP, total_C8AA_ind_S0_SP, total_alpha_ind_S0_SP, total_fods_ind_S0_SP = 0,0,0,0
                    total_C6AA_ind_T1_SP, total_C8AA_ind_T1_SP, total_alpha_ind_T1_SP, total_fods_ind_T1_SP = 0,0,0,0
                    dipole, wiberg_matrix, smiles = 0, [], ''

                    if "total energy" in data: energy = data['total energy']
                    if "HOMO-LUMO gap/eV" in data: homo_lumo_gap = data['HOMO-LUMO gap/eV']
                    if "HOMO" in data: homo = data['HOMO']
                    if "LUMO" in data: lumo = data['LUMO']
                    if "HOMO occupancy" in data: homo_occ = data['HOMO occupancy']
                    if "LUMO occupancy" in data: lumo_occ = data['LUMO occupancy']
                    if "Dipole module/D" in data: dipole = data['Dipole module/D']
                    if "Transition dipole module/D" in data: transition_dipole = data['Transition dipole module/D']
                    if "Fermi-level/eV" in data: Fermi_level = data['Fermi-level/eV']
                    if "Total free energy" in data: free_energy = data['Total free energy']
                    if "Zero point energy" in data: zero_point_E = data['Zero point energy']
                    if "T1-to-S0 vertical relaxation E" in data: T1toS0gap = data['T1-to-S0 vertical relaxation E']
                    if "S0-to-T1 vertical excitation E" in data: S0toT1gap = data['S0-to-T1 vertical excitation E']
                    if 'HOMO-LUMO gap at S0/eV' in data: homo_lumo_S0_SP = data['HOMO-LUMO gap at S0/eV']
                    if 'HOMO at S0' in data: homo_S0_SP = data['HOMO at S0']
                    if 'LUMO at S0' in data: lumo_S0_SP = data['LUMO at S0']
                    if 'HOMO occupancy at S0' in data: homo_occ_S0_SP = data['HOMO occupancy at S0']
                    if 'LUMO occupancy at S0' in data: lumo_occ_S0_SP = data['LUMO occupancy at S0']
                    if 'Dipole module at S0/D' in data: dipole_module_S0_SP = data['Dipole module at S0/D']
                    if 'Fermi-level at S0/eV' in data: Fermi_level_S0_SP = data['Fermi-level at S0/eV']
                    if 'HOMO-LUMO gap at T1/eV' in data: homo_lumo_T1_SP = data['HOMO-LUMO gap at T1/eV']
                    if 'HOMO at T1' in data: homo_T1_SP = data['HOMO at T1']
                    if 'LUMO at T1' in data: lumo_T1_SP = data['LUMO at T1']
                    if 'HOMO occupancy at T1' in data: homo_occ_T1_SP = data['HOMO occupancy at T1']
                    if 'LUMO occupancy at T1' in data: lumo_occ_T1_SP = data['LUMO occupancy at T1']
                    if 'Dipole module at T1/D' in data: dipole_module_T1_SP = data['Dipole module at T1/D']
                    if 'Fermi-level at T1/eV' in data: Fermi_level_T1_SP = data['Fermi-level at T1/eV']
                    if "Total charge" in data: total_charge = data['Total charge']
                    if "number of unpaired electrons" in data: unpaired_e = data['number of unpaired electrons']
                    if "electron affinity" in data: electron_affinity = data['electron affinity']
                    if "ionization potential" in data: ionization_potential = data['ionization potential']
                    if "global electrophilicity" in data: global_electrophilicity = data['global electrophilicity']
                    if "Total dispersion C6" in data: total_C6AA_ind = data['Total dispersion C6']
                    if "Total dispersion C8" in data: total_C8AA_ind = data['Total dispersion C8']
                    if "Total polarizability alpha" in data: total_alpha_ind = data['Total polarizability alpha']
                    if "Total FOD" in data: total_fods_ind = data['Total FOD']
                    if "Total SASA" in data: total_SASA_ind = data['Total SASA']
                    if "Total dispersion C6 at S0" in data: total_C6AA_ind_S0_SP = data['Total dispersion C6 at S0']
                    if "Total dispersion C8 at S0" in data: total_C8AA_ind_S0_SP = data['Total dispersion C8 at S0']
                    if "Total polarizability alpha at S0" in data: total_alpha_ind_S0_SP = data['Total polarizability alpha at S0']
                    if "Total FOD at S0" in data: total_fods_ind_S0_SP = data['Total FOD at S0']
                    if "Total dispersion C6 at T1" in data: total_C6AA_ind_T1_SP = data['Total dispersion C6 at T1']
                    if "Total dispersion C8 at T1" in data: total_C8AA_ind_T1_SP = data['Total dispersion C8 at T1']
                    if "Total polarizability alpha at T1" in data: total_alpha_ind_T1_SP = data['Total polarizability alpha at T1']
                    if "Total FOD at T1" in data: total_fods_ind_T1_SP = data['Total FOD at T1']
                    if "Wiberg matrix" in data: wiberg_matrix = data['Wiberg matrix']
                    if "SMILES" in data: smiles = data['SMILES']

                    E_name.append(name); xtb_energy.append(energy); ho_lu_gap.append(homo_lumo_gap); homos.append(homo); lumos.append(lumo); homo_occs.append(homo_occ); lumo_occs.append(lumo_occ); dipoles.append(dipole)
                    trans_dip.append(transition_dipole), Fermi_lvl.append(Fermi_level), tot_chrg.append(total_charge), unp_e.append(unpaired_e), free_E.append(free_energy), ZPE.append(zero_point_E), T1_S0_gap.append(T1toS0gap), S0_T1_gap.append(S0toT1gap)
                    electron_affinities.append(electron_affinity); ionization_potentials.append(ionization_potential); global_electrophilicities.append(global_electrophilicity)
                    total_C6AA.append(total_C6AA_ind); total_C8AA.append(total_C8AA_ind); total_alpha.append(total_alpha_ind); total_fods.append(total_fods_ind); total_SASA.append(total_SASA_ind)

                    ho_lu_gap_S0_SP.append(homo_lumo_S0_SP); homos_S0_SP.append(homo_S0_SP); lumos_S0_SP.append(lumo_S0_SP);
                    homos_occ_S0_SP.append(homo_occ_S0_SP); lumos_occ_S0_SP.append(lumo_occ_S0_SP); dipoles_S0_SP.append(dipole_module_S0_SP); Fermi_lvl_S0_SP.append(Fermi_level_S0_SP)
                    total_C6AA_S0_SP.append(total_C6AA_ind_S0_SP); total_C8AA_S0_SP.append(total_C8AA_ind_S0_SP); total_alpha_S0_SP.append(total_alpha_ind_S0_SP); total_fods_S0_SP.append(total_fods_ind_S0_SP)

                    ho_lu_gap_T1_SP.append(homo_lumo_T1_SP); homos_T1_SP.append(homo_T1_SP); lumos_T1_SP.append(lumo_T1_SP);
                    homos_occ_T1_SP.append(homo_occ_T1_SP); lumos_occ_T1_SP.append(lumo_occ_T1_SP); dipoles_T1_SP.append(dipole_module_T1_SP); Fermi_lvl_T1_SP.append(Fermi_level_T1_SP)
                    total_C6AA_T1_SP.append(total_C6AA_ind_T1_SP); total_C8AA_T1_SP.append(total_C8AA_ind_T1_SP); total_alpha_T1_SP.append(total_alpha_ind_T1_SP); total_fods_T1_SP.append(total_fods_ind_T1_SP)

                    T1S0_include, S0T1_include = False, False
                    for value in T1_S0_gap:
                        if value != 0:
                            T1S0_include = True
                            break
                    for value in S0_T1_gap:
                        if value != 0:
                            S0T1_include = True
                            break

                    smiles_strings.append(smiles)

                    # parse atomic properties from JSON
                    atoms, xtb_charges, mulliken_charges, cm5_charges, fukui_plus, fukui_minus, fukui_rad = [], [], [], [], [], [], []
                    s_prop, p_prop, d_prop = [], [], []
                    covCN, C6AA, alpha = [], [], []
                    charges_S0_SP, covCN_S0_SP, C6AA_S0_SP, alpha_S0_SP = [], [], [], []
                    xtb_charges_T1_SP, covCN_T1_SP, C6AA_T1_SP, alpha_T1_SP = [], [], [], []
                    freq_cm_ind, freq_disp_mod_ind, freq_redmass_ind, freq_IRintens_ind, = [], [], [], []
                    fod_ind, s_prop_fods_ind, p_prop_fods_ind, d_prop_fods_ind = [], [], [], []
                    born_rad_ind, SASA_ind, h_bond_ind = [], [], []
                    fod_ind_S0_SP, s_prop_fods_ind_S0_SP, p_prop_fods_ind_S0_SP, d_prop_fods_ind_S0_SP = [], [], [], []
                    fod_ind_T1_SP, s_prop_fods_ind_T1_SP, p_prop_fods_ind_T1_SP, d_prop_fods_ind_T1_SP = [], [], [], []

                    if "atoms" in data: atoms = data['atoms']
                    if "partial charges" in data: xtb_charges = data['partial charges']
                    if "mulliken charges" in data: mulliken_charges = data['mulliken charges']
                    if "cm5 charges" in data: cm5_charges = data['cm5 charges']
                    if "FUKUI+" in data: fukui_plus = data['FUKUI+']
                    if "FUKUI-" in data: fukui_minus = data['FUKUI-']
                    if "FUKUIrad" in data: fukui_rad = data['FUKUIrad']
                    if "s proportion" in data: s_prop = data['s proportion']
                    if "p proportion" in data: p_prop = data['p proportion']
                    if "d proportion" in data: d_prop = data['d proportion']
                    if "Coordination numbers" in data: covCN = data['Coordination numbers']
                    if "Dispersion coefficient C6" in data: C6AA = data['Dispersion coefficient C6']
                    if "Polarizability alpha" in data: alpha = data['Polarizability alpha']
                    if 'partial charges at S0' in data: xtb_charges_S0_SP = data['partial charges at S0']
                    if 'Coordination numbers at S0' in data: covCN_S0_SP = data['Coordination numbers at S0']
                    if 'Dispersion coefficient C6 at S0' in data: C6AA_S0_SP = data['Dispersion coefficient C6 at S0']
                    if 'Polarizability alpha at S0' in data: alpha_S0_SP = data['Polarizability alpha at S0']
                    if 'partial charges at T1' in data: xtb_charges_T1_SP = data['partial charges at T1']
                    if 'Coordination numbers at T1' in data: covCN_T1_SP = data['Coordination numbers at T1']
                    if 'Dispersion coefficient C6 at T1' in data: C6AA_T1_SP = data['Dispersion coefficient C6 at T1']
                    if 'Polarizability alpha at T1' in data: alpha_T1_SP = data['Polarizability alpha at T1']
                    if 'Frequencies' in data: freq_cm_ind = data['Frequencies']
                    if 'Frequency displacement modules' in data: freq_disp_mod_ind = data['Frequency displacement modules']
                    if 'Frequency reduced masses' in data: freq_redmass_ind = data['Frequency reduced masses']
                    if 'Frequency IR intensity' in data: freq_IRintens_ind = data['Frequency IR intensity']
                    if 'FOD' in data: fod_ind = data['FOD']
                    if 'FOD s proportion' in data: s_prop_fods_ind = data['FOD s proportion']
                    if 'FOD p proportion' in data: p_prop_fods_ind = data['FOD p proportion']
                    if 'FOD d proportion' in data: d_prop_fods_ind = data['FOD d proportion']
                    if 'Born radii' in data: born_rad_ind = data['Born radii']
                    if 'Atomic SASAs' in data: SASA_ind = data['Atomic SASAs']
                    if 'Solvent H bonds' in data: h_bond_ind = data['Solvent H bonds']
                    if 'FOD at S0' in data: fod_ind_S0_SP = data['FOD at S0']
                    if 'FOD s proportion at S0' in data: s_prop_fods_ind_S0_SP = data['FOD s proportion at S0']
                    if 'FOD p proportion at S0' in data: p_prop_fods_ind_S0_SP = data['FOD p proportion at S0']
                    if 'FOD d proportion at S0' in data: d_prop_fods_ind_S0_SP = data['FOD d proportion at S0']
                    if 'FOD at T1' in data: fod_ind_T1_SP = data['FOD at T1']
                    if 'FOD s proportion at T1' in data: s_prop_fods_ind_T1_SP = data['FOD s proportion at T1']
                    if 'FOD p proportion at T1' in data: p_prop_fods_ind_T1_SP = data['FOD p proportion at T1']
                    if 'FOD d proportion at T1' in data: d_prop_fods_ind_T1_SP = data['FOD d proportion at T1']

                    if args.verbose != False:
                        print(F'  XTB energy: {energy:.5f}')
                        print(F'  XTB HOMO_LUMO gap: {homo_lumo_gap:.4f}')

                    if args.atom != "False":
                        atom_of_interest = None
                        # thresholds to detects single and double bonds
                        single_bond_threshold = 0.35
                        double_bond_threshold = 1.3
                        for request in args.atom.split(','):
                            if request.find('-') > -1 or request.find('=') > -1:
                                [at1, at2] = re.split('-|=', request)
                                for a, atom_a in enumerate(atoms):
                                    for b, atom_b in enumerate(atoms):
                                        if at2 != '*':
                                            if atom_a == at1 and atom_b == at2:
                                                if request.find('-') > -1: bo_cutoff = single_bond_threshold
                                                if request.find('=') > -1: bo_cutoff = double_bond_threshold
                                                if wiberg_matrix[a][b] > bo_cutoff : # a & b are bonded & match the atom types at1 & at2
                                                    atom_of_interest = a
                                                    break

                                        if at2 == '*':
                                            if atom_a == at1:
                                                if request.find('-') > -1: bo_cutoff = single_bond_threshold
                                                if request.find('=') > -1: bo_cutoff = double_bond_threshold
                                                if wiberg_matrix[a][b] > bo_cutoff : # a & b are bonded & match the atom types at1 & at2
                                                    atom_of_interest = a
                                                    break

                                        if at2 == 'Hal':
                                            if atom_b in ['F', 'Cl', 'Br', 'I']:
                                                if atom_a == at1:
                                                    if request.find('-') > -1: bo_cutoff = single_bond_threshold
                                                    if wiberg_matrix[a][b] > bo_cutoff : # a & b are bonded & match the atom types at1 & at2
                                                        atom_of_interest = a
                                                        break

                        if atom_of_interest != None:
                            xtb_fup.append(fukui_plus[atom_of_interest])
                            xtb_fun.append(fukui_minus[atom_of_interest])
                            xtb_fne.append(fukui_rad[atom_of_interest])
                            xtb_s_prop.append(s_prop[atom_of_interest])
                            xtb_p_prop.append(p_prop[atom_of_interest])
                            xtb_d_prop.append(d_prop[atom_of_interest])
                            xtb_chrg.append(xtb_charges[atom_of_interest])
                            mulliken_chrg.append(mulliken_charges[atom_of_interest])
                            cm5_chrg.append(cm5_charges[atom_of_interest])
                            xtb_covCN.append(covCN[atom_of_interest])
                            xtb_C6AA.append(C6AA[atom_of_interest])
                            xtb_alpha.append(alpha[atom_of_interest])
                            fod.append(fod_ind[atom_of_interest])
                            s_prop_fods.append(s_prop_fods_ind[atom_of_interest])
                            p_prop_fods.append(p_prop_fods_ind[atom_of_interest])
                            d_prop_fods.append(d_prop_fods_ind[atom_of_interest])
                            # I need to fix these 3 try statements in a better way, in theory when calcs
                            # are in gas phase this part shouldn't be read
                            try:
                                born_rad.append(born_rad_ind[atom_of_interest])
                            except IndexError:
                                born_rad.append(0)
                            try:
                                SASA.append(SASA_ind[atom_of_interest])
                            except IndexError:
                                SASA.append(0)
                            try:
                                h_bond.append(h_bond_ind[atom_of_interest])
                            except IndexError:
                                h_bond.append(0)

                            if T1S0_include:
                                xtb_chrg_S0_SP.append(xtb_charges_S0_SP[atom_of_interest])
                                xtb_covCN_S0_SP.append(covCN_S0_SP[atom_of_interest])
                                xtb_C6AA_S0_SP.append(C6AA_S0_SP[atom_of_interest])
                                xtb_alpha_S0_SP .append(alpha_S0_SP[atom_of_interest])
                                fod_S0_SP.append(fod_ind_S0_SP[atom_of_interest])
                                s_prop_fods_S0_SP.append(s_prop_fods_ind_S0_SP[atom_of_interest])
                                p_prop_fods_S0_SP.append(p_prop_fods_ind_S0_SP[atom_of_interest])
                                d_prop_fods_S0_SP.append(d_prop_fods_ind_S0_SP[atom_of_interest])

                            if S0T1_include:
                                xtb_chrg_T1_SP.append(xtb_charges_T1_SP[atom_of_interest])
                                xtb_covCN_T1_SP.append(covCN_T1_SP[atom_of_interest])
                                xtb_C6AA_T1_SP.append(C6AA_T1_SP[atom_of_interest])
                                xtb_alpha_T1_SP .append(alpha_T1_SP[atom_of_interest])
                                fod_T1_SP.append(fod_ind_T1_SP[atom_of_interest])
                                s_prop_fods_T1_SP.append(s_prop_fods_ind_T1_SP[atom_of_interest])
                                p_prop_fods_T1_SP.append(p_prop_fods_ind_T1_SP[atom_of_interest])
                                d_prop_fods_T1_SP.append(d_prop_fods_ind_T1_SP[atom_of_interest])

                            if args.v_bur:
                                try:
                                    dbmol = dbstep.dbstep(name+'.xyz',atom1=str(atom_of_interest+1),verbose=False,commandline=True,volume=True)
                                    v_bur.append(dbmol.bur_vol)
                                except:
                                    pass

                            # this part detects and stores the freq with larger displacement for the atom of interest
                            if args.get_freq:
                                larger_freq, larger_disp, larger_redmass, larger_IRintens = 0,0,0,0
                                second_larger_freq, second_larger_disp, second_larger_redmass, second_larger_IRintens = 0,0,0,0
                                for i,freq_ind in enumerate(freq_cm_ind):
                                    if freq_disp_mod_ind[i][atom_of_interest] > larger_disp:
                                        # detect largest and second largest displacements on the specific atom
                                        second_larger_freq = larger_freq
                                        second_larger_disp = larger_disp
                                        second_larger_redmass = larger_redmass
                                        second_larger_IRintens = larger_IRintens

                                        larger_freq = freq_ind
                                        larger_disp = freq_disp_mod_ind[i][atom_of_interest]
                                        larger_redmass = freq_redmass_ind[i]
                                        larger_IRintens = freq_IRintens_ind[i]

                                    elif freq_disp_mod_ind[i][atom_of_interest] > second_larger_disp:
                                        second_larger_freq = freq_ind
                                        second_larger_disp = freq_disp_mod_ind[i][atom_of_interest]
                                        second_larger_redmass = freq_redmass_ind[i]
                                        second_larger_IRintens = freq_IRintens_ind[i]

                                # append largest and second largest frequencies on the specific atom
                                freq_cm.append(larger_freq)
                                freq_disp_mod.append(larger_disp)
                                freq_redmass.append(larger_redmass)
                                freq_IRintens.append(larger_IRintens)

                                freq_second_cm.append(second_larger_freq)
                                freq_second_disp_mod.append(second_larger_disp)
                                freq_second_redmass.append(second_larger_redmass)
                                freq_second_IRintens.append(second_larger_IRintens)

                        else:
                            xtb_fup.append('')
                            xtb_fun.append('')
                            xtb_fne.append('')
                            xtb_s_prop.append('')
                            xtb_p_prop.append('')
                            xtb_d_prop.append('')
                            xtb_chrg.append('')
                            mulliken_chrg.append('')
                            cm5_chrg.append('')
                            v_bur.append('')
                            xtb_covCN.append('')
                            xtb_C6AA.append('')
                            xtb_alpha.append('')
                            xtb_chrg_S0_SP.append('')
                            xtb_covCN_S0_SP.append('')
                            xtb_C6AA_S0_SP.append('')
                            xtb_alpha_S0_SP.append('')
                            xtb_chrg_T1_SP.append('')
                            xtb_covCN_T1_SP.append('')
                            xtb_C6AA_T1_SP.append('')
                            xtb_alpha_T1_SP.append('')
                            xtb_alpha_T1_SP.append('')
                            freq_cm.append('')
                            freq_disp_mod.append('')
                            freq_redmass.append('')
                            freq_IRintens.append('')
                            freq_second_cm.append('')
                            freq_second_disp_mod.append('')
                            freq_second_redmass.append('')
                            freq_second_IRintens.append('')
                            fod.append('')
                            s_prop_fods.append('')
                            p_prop_fods.append('')
                            d_prop_fods.append('')
                            born_rad.append('')
                            SASA.append('')
                            h_bond.append('')
                            fod_S0_SP.append('')
                            s_prop_fods_S0_SP.append('')
                            p_prop_fods_S0_SP.append('')
                            d_prop_fods_S0_SP.append('')
                            fod_T1_SP.append('')
                            s_prop_fods_T1_SP.append('')
                            p_prop_fods_T1_SP.append('')
                            d_prop_fods_T1_SP.append('')

        if len(E_name) > 0:
            # XTB molecule data
            xtb_mol_prop = {'code_name': E_name, 'SMILES': smiles_strings, 'Energy': xtb_energy, 'Free_energy': free_E,
                            'ZPE': ZPE, 'Total_charge': tot_chrg, 'Unpaired_e': unp_e, 'HOMO_LUMO_gap': ho_lu_gap,
                            'HOMO': homos, 'LUMO':lumos, 'HOMO_occupancy': homo_occs, 'LUMO_occupancy':lumo_occs,
                            'Fermi_level': Fermi_lvl, 'Dipole': dipoles, 'Transition_dipole': trans_dip,
                            'Elec_affin': electron_affinities, 'Ioniz_pot': ionization_potentials, 'Global_electroph': global_electrophilicities,
                            'Total_C6_dispersion': total_C6AA, 'Total_C8_dispersion': total_C8AA, 'Total_polariz_alpha':total_alpha,
                            'Total_FOD': total_fods, 'Total_SASA': total_SASA}

            if T1S0_include:
                xtb_mol_prop['T1_to_S0_gap'] = T1_S0_gap
                xtb_mol_prop['HOMO_LUMO_gap_at_S0'] = ho_lu_gap_S0_SP
                xtb_mol_prop['HOMO_at_S0'] = homos_S0_SP
                xtb_mol_prop['LUMO_at_S0'] = lumos_S0_SP
                xtb_mol_prop['HOMO_occupancy_at_S0'] = homos_occ_S0_SP
                xtb_mol_prop['LUMO_occupancy_at_S0'] = lumos_occ_S0_SP
                xtb_mol_prop['Dipole_at_S0'] = dipoles_S0_SP
                xtb_mol_prop['Fermi_level_at_S0'] = Fermi_lvl_S0_SP
                xtb_mol_prop['Total_C6_dispersion_at_S0'] = total_C6AA_S0_SP
                xtb_mol_prop['Total_C8_dispersion_at_S0'] = total_C8AA_S0_SP
                xtb_mol_prop['Total_polariz_alpha_at_S0'] = total_alpha_S0_SP
                xtb_mol_prop['Total_FOD_at_S0'] = total_fods_S0_SP

            if S0T1_include:
                xtb_mol_prop['S0_to_T1_gap'] = S0_T1_gap
                xtb_mol_prop['HOMO_LUMO_gap_at_T1'] = ho_lu_gap_T1_SP
                xtb_mol_prop['HOMO_at_T1'] = homos_T1_SP
                xtb_mol_prop['LUMO_at_T1'] = lumos_T1_SP
                xtb_mol_prop['HOMO_occupancy_at_T1'] = homos_occ_T1_SP
                xtb_mol_prop['LUMO_occupancy_at_T1'] = lumos_occ_T1_SP
                xtb_mol_prop['Dipole_at_T1'] = dipoles_T1_SP
                xtb_mol_prop['Fermi_level_at_T1'] = Fermi_lvl_T1_SP
                xtb_mol_prop['Total_C6_dispersion_at_T1'] = total_C6AA_T1_SP
                xtb_mol_prop['Total_C8_dispersion_at_T1'] = total_C8AA_T1_SP
                xtb_mol_prop['Total_polariz_alpha_at_T1'] = total_alpha_T1_SP
                xtb_mol_prop['Total_FOD_at_T1'] = total_fods_T1_SP

            xtb_mol_data = pd.DataFrame(xtb_mol_prop)

            # optional XTB atomic data (appended only if atom selected)
            if args.atom != "False":
                xtb_at_prop = {f'{at1}_xTB_charge': xtb_chrg, f'{at1}_Mulliken_charge': mulliken_chrg, f'{at1}_CM5_charge': cm5_chrg,
                               f'{at1}_Fukui+': xtb_fup, f'{at1}_Fukui-': xtb_fun, f'{at1}_Fukui_rad': xtb_fne,
                               f'{at1}_s_proportion': xtb_s_prop, f'{at1}_p_proportion': xtb_p_prop, f'{at1}_d_proportion': xtb_d_prop,
                               f'{at1}_Coord_number': xtb_covCN, f'{at1}_C6_dispersion': xtb_C6AA, f'{at1}_Polariz_alpha': xtb_alpha,
                               f'{at1}_FOD': fod, f'{at1}_FOD_s_proportion': s_prop_fods, f'{at1}_FOD_p_proportion': p_prop_fods,
                               f'{at1}_FOD_d_proportion': d_prop_fods, f'{at1}_Born_radius': born_rad, f'{at1}_SASA': SASA, f'{at1}_solvent_H_bond': h_bond}
                if args.get_freq:
                    xtb_at_prop = {f'{at1}_xTB_charge': xtb_chrg, f'{at1}_Mulliken_charge': mulliken_chrg, f'{at1}_CM5_charge': cm5_chrg,
                                   f'{at1}_Fukui+': xtb_fup, f'{at1}_Fukui-': xtb_fun, f'{at1}_Fukui_rad': xtb_fne,
                                   f'{at1}_s_proportion': xtb_s_prop, f'{at1}_p_proportion': xtb_p_prop, f'{at1}_d_proportion': xtb_d_prop,
                                   f'{at1}_Coord_number': xtb_covCN, f'{at1}_C6_dispersion': xtb_C6AA, f'{at1}_Polariz_alpha': xtb_alpha,
                                   f'{at1}_FOD': fod, f'{at1}_FOD_s_proportion': s_prop_fods, f'{at1}_FOD_p_proportion': p_prop_fods,
                                   f'{at1}_FOD_d_proportion': d_prop_fods, f'{at1}_Born_radius': born_rad, f'{at1}_SASA': SASA, f'{at1}_solvent_H_bond': h_bond,
                                   f'{at1}_Freq_1st': freq_cm, f'{at1}_Freq_1st_displac': freq_disp_mod,
                                   f'{at1}_Freq_1st_red_mass': freq_redmass, f'{at1}_Freq_1st_IR': freq_IRintens,
                                   f'{at1}_Freq_2nd': freq_second_cm, f'{at1}_Freq_2nd_displac': freq_second_disp_mod,
                                   f'{at1}_Freq_2nd_red_mass': freq_second_redmass, f'{at1}_Freq_2nd_IR': freq_second_IRintens}

                if T1S0_include:
                    xtb_at_prop = {f'{at1}_xTB_charge': xtb_chrg, f'{at1}_Mulliken_charge': mulliken_chrg, f'{at1}_CM5_charge': cm5_chrg,
                                   f'{at1}_Fukui+': xtb_fup, f'{at1}_Fukui-': xtb_fun, f'{at1}_Fukui_rad': xtb_fne,
                                   f'{at1}_s_proportion': xtb_s_prop, f'{at1}_p_proportion': xtb_p_prop, f'{at1}_d_proportion': xtb_d_prop,
                                   f'{at1}_Coord_number': xtb_covCN, f'{at1}_C6_dispersion': xtb_C6AA, f'{at1}_Polariz_alpha': xtb_alpha,
                                   f'{at1}_FOD': fod, f'{at1}_FOD_s_proportion': s_prop_fods, f'{at1}_FOD_p_proportion': p_prop_fods,
                                   f'{at1}_FOD_d_proportion': d_prop_fods, f'{at1}_Born_radius': born_rad, f'{at1}_SASA': SASA, f'{at1}_solvent_H_bond': h_bond,
                                   f'{at1}_xTB_charge_at_S0': xtb_chrg_S0_SP, f'{at1}_Coord_number_at_S0': xtb_covCN_S0_SP,
                                   f'{at1}_C6_dispersion_at_S0': xtb_C6AA_S0_SP, f'{at1}_Polariz_alpha_at_S0': xtb_alpha_S0_SP,
                                   f'{at1}_FOD_at_S0': fod_S0_SP, f'{at1}_FOD_s_proportion_at_S0': s_prop_fods_S0_SP,
                                   f'{at1}_FOD_p_proportion_at_S0': p_prop_fods_S0_SP, f'{at1}_FOD_d_proportion_at_S0': d_prop_fods_S0_SP}
                    if args.get_freq:
                        xtb_at_prop = {f'{at1}_xTB_charge': xtb_chrg, f'{at1}_Mulliken_charge': mulliken_chrg, f'{at1}_CM5_charge': cm5_chrg,
                                       f'{at1}_Fukui+': xtb_fup, f'{at1}_Fukui-': xtb_fun, f'{at1}_Fukui_rad': xtb_fne,
                                       f'{at1}_s_proportion': xtb_s_prop, f'{at1}_p_proportion': xtb_p_prop, f'{at1}_d_proportion': xtb_d_prop,
                                       f'{at1}_Coord_number': xtb_covCN, f'{at1}_C6_dispersion': xtb_C6AA, f'{at1}_Polariz_alpha': xtb_alpha,
                                       f'{at1}_FOD': fod, f'{at1}_FOD_s_proportion': s_prop_fods, f'{at1}_FOD_p_proportion': p_prop_fods,
                                       f'{at1}_FOD_d_proportion': d_prop_fods, f'{at1}_Born_radius': born_rad, f'{at1}_SASA': SASA, f'{at1}_solvent_H_bond': h_bond,
                                       f'{at1}_xTB_charge_at_S0': xtb_chrg_S0_SP, f'{at1}_Coord_number_at_S0': xtb_covCN_S0_SP,
                                       f'{at1}_C6_dispersion_at_S0': xtb_C6AA_S0_SP, f'{at1}_Polariz_alpha_at_S0': xtb_alpha_S0_SP,
                                       f'{at1}_FOD_at_S0': fod_S0_SP, f'{at1}_FOD_s_proportion_at_S0': s_prop_fods_S0_SP,
                                       f'{at1}_FOD_p_proportion_at_S0': p_prop_fods_S0_SP, f'{at1}_FOD_d_proportion_at_S0': d_prop_fods_S0_SP,
                                       f'{at1}_Freq_1st': freq_cm, f'{at1}_Freq_1st_displac': freq_disp_mod,
                                       f'{at1}_Freq_1st_red_mass': freq_redmass, f'{at1}_Freq_1st_IR': freq_IRintens,
                                       f'{at1}_Freq_2nd': freq_second_cm, f'{at1}_Freq_2nd_displac': freq_second_disp_mod,
                                       f'{at1}_Freq_2nd_red_mass': freq_second_redmass, f'{at1}_Freq_2nd_IR': freq_second_IRintens}

                if S0T1_include:
                    xtb_at_prop = {f'{at1}_xTB_charge': xtb_chrg, f'{at1}_Mulliken_charge': mulliken_chrg, f'{at1}_CM5_charge': cm5_chrg,
                                   f'{at1}_Fukui+': xtb_fup, f'{at1}_Fukui-': xtb_fun, f'{at1}_Fukui_rad': xtb_fne,
                                   f'{at1}_s_proportion': xtb_s_prop, f'{at1}_p_proportion': xtb_p_prop, f'{at1}_d_proportion': xtb_d_prop,
                                   f'{at1}_Coord_number': xtb_covCN, f'{at1}_C6_dispersion': xtb_C6AA, f'{at1}_Polariz_alpha': xtb_alpha,
                                   f'{at1}_FOD': fod, f'{at1}_FOD_s_proportion': s_prop_fods, f'{at1}_FOD_p_proportion': p_prop_fods,
                                   f'{at1}_FOD_d_proportion': d_prop_fods, f'{at1}_Born_radius': born_rad, f'{at1}_SASA': SASA, f'{at1}_solvent_H_bond': h_bond,
                                   f'{at1}_xTB_charge_at_T1': xtb_chrg_T1_SP, f'{at1}_Coord_number_at_T1': xtb_covCN_T1_SP,
                                   f'{at1}_C6_dispersion_at_T1': xtb_C6AA_T1_SP, f'{at1}_Polariz_alpha_at_T1': xtb_alpha_T1_SP,
                                   f'{at1}_FOD_at_T1': fod_T1_SP, f'{at1}_FOD_s_proportion_at_T1': s_prop_fods_T1_SP,
                                   f'{at1}_FOD_p_proportion_at_T1': p_prop_fods_T1_SP, f'{at1}_FOD_d_proportion_at_T1': d_prop_fods_T1_SP}
                    if args.get_freq:
                        xtb_at_prop = {f'{at1}_xTB_charge': xtb_chrg, f'{at1}_Mulliken_charge': mulliken_chrg, f'{at1}_CM5_charge': cm5_chrg,
                                       f'{at1}_Fukui+': xtb_fup, f'{at1}_Fukui-': xtb_fun, f'{at1}_Fukui_rad': xtb_fne,
                                       f'{at1}_s_proportion': xtb_s_prop, f'{at1}_p_proportion': xtb_p_prop, f'{at1}_d_proportion': xtb_d_prop,
                                       f'{at1}_Coord_number': xtb_covCN, f'{at1}_C6_dispersion': xtb_C6AA, f'{at1}_Polariz_alpha': xtb_alpha,
                                       f'{at1}_FOD': fod, f'{at1}_FOD_s_proportion': s_prop_fods, f'{at1}_FOD_p_proportion': p_prop_fods,
                                       f'{at1}_FOD_d_proportion': d_prop_fods, f'{at1}_Born_radius': born_rad, f'{at1}_SASA': SASA, f'{at1}_solvent_H_bond': h_bond,
                                       f'{at1}_xTB_charge_at_T1': xtb_chrg_T1_SP, f'{at1}_Coord_number_at_T1': xtb_covCN_T1_SP,
                                       f'{at1}_C6_dispersion_at_T1': xtb_C6AA_T1_SP, f'{at1}_Polariz_alpha_at_T1': xtb_alpha_T1_SP,
                                       f'{at1}_FOD_at_T1': fod_T1_SP, f'{at1}_FOD_s_proportion_at_T1': s_prop_fods_T1_SP,
                                       f'{at1}_FOD_p_proportion_at_T1': p_prop_fods_T1_SP, f'{at1}_FOD_d_proportion_at_T1': d_prop_fods_T1_SP,
                                       f'{at1}_Freq_1st': freq_cm, f'{at1}_Freq_1st_displac': freq_disp_mod,
                                       f'{at1}_Freq_1st_red_mass': freq_redmass, f'{at1}_Freq_1st_IR': freq_IRintens,
                                       f'{at1}_Freq_2nd': freq_second_cm, f'{at1}_Freq_2nd_displac': freq_second_disp_mod,
                                       f'{at1}_Freq_2nd_red_mass': freq_second_redmass, f'{at1}_Freq_2nd_IR': freq_second_IRintens}

                xtb_at_data = pd.DataFrame(xtb_at_prop)
                xtb_data = xtb_mol_data.join(xtb_at_data)

            else:
                xtb_data = xtb_mol_data

            if len(v_bur) > 0:
                bur_vol = {'Vbur': v_bur}
                bur_vol_data = pd.DataFrame(bur_vol)
                xtb_data = xtb_data.join(bur_vol_data)

            if len(xtb_data) != 0:
                print('\n  o DISCO XTB DATA SUMMARY \n', xtb_data)

            os.chdir(w_dir)
            if args.csv: # write to csv
                xtb_data = xtb_data.sort_values('code_name')
                xtb_data.to_csv('DISCO_xtb-data.csv', index=False)
        else:
            print('\nThere are no valid calculations to be analyzed!')

    if args.xtb.upper() == "RUN":
        time_file = open(f'Exec_time_run.txt',"w")
        time_file.write(f"--- Execution time: {round(time.time() - start_time,0)} seconds for {count_files} files ---") # n of atoms
        time_file.close()

        print(f"--- Execution time: {round(time.time() - start_time,0)} seconds for {count_files} files ---")

    if args.xtb.upper() == "ANALYZE":
        time_file = open(f'Exec_time_analyze.txt',"w")
        time_file.write(f"--- Execution time: {round(time.time() - start_time,0)} seconds for {count_json_files} files ---") # n of atoms
        time_file.close()

        print(f"--- Execution time: {round(time.time() - start_time,0)} seconds for {count_json_files} files ---")

    if args.xtb.upper() == "BOLTZ":
        os.chdir(w_dir)
        xtb_data = pd.read_csv('DISCO_xtb-data.csv')

        if args.pyconf_input != None:
            try:
                pyconf_data = pd.read_csv(args.pyconf_input)
            except:
                print('The specified input file of pyCONFORT could not be found (make sure the file is in the same folder and it is a CSV file!)')
        else:
            print('You need to specify the input file used by pyCONFORT! (--pyconf_input FILENAME)')

        # this part cleans up the descriptors with NaN or inf values (i.e. EA, IP and GE work poorly
        # with Ir radicals)

        xtb_data = xtb_data.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
        xtb_data.to_csv('DISCO_xtb-data_BOLTZ.csv', index=False)

        # adds a dummy row at the end of the dataframe (otherwise the last point is omitted)
        new_row = pd.DataFrame(columns=xtb_data.columns)
        new_row.loc[0, :] = np.zeros(len(xtb_data.columns))
        new_row['code_name'].iloc[0] = 'Dummy'
        xtb_data = xtb_data.append(new_row, ignore_index=True)

        # some units used in the Boltzmann averaging process
        au_to_kcal, gas_constant = 627.5, 0.001987
        pyconf_boltz_data = pd.DataFrame(columns=xtb_data.columns)
        boltz_avg_props = pd.DataFrame(columns=xtb_data.columns)
        for i,molecule in enumerate(pyconf_data['code_name']):
            start_compound = 0
            g_min, boltz_sum = 10000000, 0
            boltz_fac_list, boltz_avg_array, boltz_prob = np.array([]), np.array([]), []
            # simple filter based on E (less than 0.05 kcal/mol of E and 0.005 eV of
            # HOMO_LUMO difference is considered a duplicate)
            dup_E, dup_homo_lumo, filter_E, filter_homo_lumo, track_mol = [], [], 0.05, 0.005, -1
            individ_props = pd.DataFrame(columns=xtb_data.columns)
            for j,xtb_molecule in enumerate(xtb_data['code_name']):
                if xtb_molecule.find(molecule+'_') > -1 or xtb_molecule == molecule:
                    dup_mol = False
                    for k,dup_E_ind in enumerate(dup_E):
                        if k != track_mol:
                            if abs((xtb_data['Energy'][j]*au_to_kcal)-dup_E_ind) < filter_E:
                                if abs(xtb_data['HOMO_LUMO_gap'][j]-dup_homo_lumo[k]) < filter_homo_lumo:
                                    dup_mol = True
                                    break
                    if not dup_mol:
                        individ_props.loc[j, :] = xtb_data.loc[j, :].tolist()
                        dup_E.append(xtb_data['Energy'][j]*au_to_kcal)
                        dup_homo_lumo.append(xtb_data['HOMO_LUMO_gap'][j])
                        track_mol += 1
                        if xtb_data['Free_energy'][j]*au_to_kcal <= g_min:
                            g_min = xtb_data['Free_energy'][j]*au_to_kcal
                            smiles_min = xtb_data['SMILES'][j]
                        start_compound += 1

                elif start_compound > 0:
                    # remove the name and SMILES and Boltz average the rest
                    individ_props = individ_props.reset_index()
                    individ_props = individ_props.drop(labels=['index','code_name','SMILES'],axis=1)

                    for k,free_E in enumerate(individ_props['Free_energy']):
                        g_rel = free_E*au_to_kcal - g_min
                        boltz_fac = math.exp(-g_rel / gas_constant / args.temp)
                        boltz_fac_list = np.append(boltz_fac_list, boltz_fac)
                        boltz_sum += boltz_fac

                    boltz_value = np.zeros(len(individ_props.loc[0, :].to_numpy()))
                    for k,_ in enumerate(individ_props['Free_energy']):
                        boltz_prob = boltz_fac_list[k] / boltz_sum
                        # avoid the two first columns

                        indiv_array = individ_props.loc[k, :].to_numpy()

                        boltz_value = boltz_value + (indiv_array * boltz_prob)

                    # put names, SMILES and averaged values together
                    name_boltz = np.array(pyconf_data['code_name'][i], dtype=object)
                    smiles_boltz = np.array(smiles_min, dtype=object)

                    boltz_avg_array = np.append(boltz_avg_array,name_boltz)
                    boltz_avg_array = np.append(boltz_avg_array,smiles_boltz)
                    boltz_avg_array = np.append(boltz_avg_array,boltz_value)

                    boltz_avg_array = list(boltz_avg_array)

                    boltz_avg_props.loc[i, :] = boltz_avg_array
                    boltz_avg_props.append(boltz_avg_array, ignore_index=True)

                    break

        boltz_avg_props.to_csv('DISCO_xtb-data_BOLTZ.csv', index=False)

    if args.xtb.upper() == "SP_FILTER":
        os.chdir(w_dir)
        destination_xtb = f'{w_dir}/xtb_files'
        if not os.path.isdir(destination_xtb):
            print('The processed xTB files were not found, please run the code with "--xtb RUN" first!')
        os.chdir(destination_xtb)

        # cut-off angle to discard non-planar square-planar complexes
        dihedral_cutoff = 30

        files_geom = glob.glob('*.xyz')
        for file in files_geom:
            # calculates dihedral angle around the metal center
            discard = detectSP_planar(file,dihedral_cutoff,args)
            # remove temporary sdf files
            try:
                os.remove(file.split('.')[0]+'.sdf')
            except FileNotFoundError:
                pass

            if discard:
                files_discard = glob.glob(file.split('.')[0]+'.*')
                # move the initial input files into a folder
                destination_discard = f'{destination_xtb}/nonplanar_SP_complexes'
                if not os.path.isdir(destination_discard):
                    os.makedirs(destination_discard)
                for file_discard in files_discard:
                    source_discard = f'{destination_xtb}/{file_discard}'
                    try:
                        shutil.move(source_discard, destination_discard)
                    except (FileExistsError,shutil.Error):
                        os.chdir(destination_discard)
                        os.remove(source_discard.split('/')[-1])
                        shutil.move(source_discard, destination_discard)

if __name__ == "__main__":
    main()
