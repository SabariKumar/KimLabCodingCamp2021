#!/opt/anaconda/envs/DL_CPU/bin/python

#### Authors: Liliana C. Gallegos and Juan V. Alegre-Requena ####
### For any questions, contact: LilianaC.Gallegos@colostate.edu or juanvi89@hotmail.com ###
from __future__ import print_function
import os, sys, glob
import re
import pandas as pd
import numpy as np
import json
import argparse
import subprocess
try: from openbabel import openbabel
except: print("! Openbabel is required for SMILES conversion")

parser = argparse.ArgumentParser(description='The script analysis ...')
parser.add_argument('--get_freq', default=False, action="store_true", help="Look for the freq with highest displacement of the selected atom in the --atom option.")
args = parser.parse_args()

def read_fukui(file):
    """
    Read fukui output file created from XTB option. Return data.
    """
    f = open(file, 'r')
    data = f.readlines()
    f.close()

    f_pos, f_negs, f_neutrals = [], [], []
    for i in range(0,len(data)):
        if data[i].find("f(+)") > -1:
            start = i+1
            break
    for j in range(start,len(data)):
        if data[j].find("      -------------") > -1:
            end = j
            break

    fukui_data = data[start:end]

    for line in fukui_data:
        item = line.split()
        f_po = float(item[-3])
        f_neg = float(item[-2])
        f_neutral = float(item[-1])
        f_pos.append(f_po)
        f_negs.append(f_neg)
        f_neutrals.append(f_neutral)

    return f_pos, f_negs, f_neutrals

def read_gfn1(file):
    """
    Read fukui output file created from XTB option. Return data.
    """
    if file.find(".gfn1") > -1:
        f = open(file, 'r')
        data = f.readlines()
        f.close()

        for i in range(0,len(data)):
            if data[i].find("Mulliken/CM5 charges") > -1:
                start = i+1
                break
        for j in range(start,len(data)):
            if data[j].find("Wiberg/Mayer (AO) data") > -1 or data[j].find('generalized Born model') > -1:
                end = j-1
                break

        pop_data = data[start:end]
        mulliken, cm5, s_prop, p_prop, d_prop = [], [], [], [], []
        for line in pop_data:
            item = line.split()
            q_mull = float(item[-5])
            q_cm5 = float(item[-4])
            s_prop_ind = float(item[-3])
            p_prop_ind = float(item[-2])
            d_prop_ind = float(item[-1])
            mulliken.append(q_mull)
            cm5.append(q_cm5)
            s_prop.append(s_prop_ind)
            p_prop.append(p_prop_ind)
            d_prop.append(d_prop_ind)

        return mulliken, cm5, s_prop, p_prop, d_prop

def read_wbo(file):
    """
    Read wbo output file created from XTB option. Return data.
    """
    if file.find(".wbo") > -1:
        f = open(file, 'r')
        data = f.readlines()
        f.close()

        bonds, wbos = [], []
        for line in data:
            item = line.split()
            bond = [int(item[0]), int(item[1])]
            wbo = float(item[2])
            bonds.append(bond)
            wbos.append(wbo)
        return bonds, wbos

def read_omega(file):
    """
    Read xtb.out file. Return data.
    """
    if file.find(".omega") > -1:
        f = open(file, 'r')
        data = f.readlines()
        f.close()

        ionization_potential, electron_affinity, global_electrophilicity = np.nan, np.nan, np.nan

        for i in range(0,len(data)):
            try:
                if data[i].find("delta SCC IP (eV)") > -1:
                    ionization_potential = float(data[i].replace(':', ' ').split()[4])
                if data[i].find("delta SCC EA (eV)") > -1:
                    electron_affinity = float(data[i].replace(':', ' ').split()[4])
                if data[i].find("Global electrophilicity index") > -1:
                    global_electrophilicity = float(data[i].split()[4])
            except: pass
        return ionization_potential, electron_affinity, global_electrophilicity
    else: pass

def read_xtb(file):
    """
    Read xtb.out file. Return data.
    """
    f = open(file, 'r')
    data = f.readlines()
    f.close()

    energy, homo_lumo, homo, lumo, atoms, numbers, chrgs, wbos = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    dipole_module, Fermi_level, transition_dipole_moment = np.nan, np.nan, np.nan
    total_charge, total_SASA = np.nan, np.nan
    total_C6AA, total_C8AA, total_alpha = np.nan, np.nan, np.nan

    for i in range(0,len(data)):
        if data[i].find("SUMMARY") > -1:
            energy = float(data[i+2].split()[3])
        if data[i].find("charge                     :") > -1:
            total_charge = int(data[i].split()[-1])
        if data[i].find("(HOMO)") > -1:
            if data[i].split()[3] != "(HOMO)":
                homo = float(data[i].split()[3])
                homo_occ = float(data[i].split()[1])
            else:
                homo = float(data[i].split()[2])
                homo_occ = 0
        if data[i].find("(LUMO)") > -1:
            if data[i].split()[3] != "(LUMO)":
                lumo = float(data[i].split()[3])
                lumo_occ = float(data[i].split()[1])
            else:
                lumo = float(data[i].split()[2])
                lumo_occ = 0
        homo_lumo = float(lumo-homo)
        if data[i].find("molecular dipole:") > -1:
            dipole_module = float(data[i+3].split()[-1])
        if data[i].find("transition dipole moment") > -1:
            transition_dipole_moment = float(data[i+2].split()[-1])
        if data[i].find("Fermi-level") > -1:
            Fermi_level = float(data[i].split()[-2])

    # get atomic properties related to charges, dispersion, etc
    start,end = 0,0
    for j in range(0,len(data)):
        if data[j].find("#   Z          covCN") > -1:
            start = j+1
            break
    for k in range(start,len(data)):
        if data[k].find("Mol. ") > -1:
            end = k-1
            total_C6AA = float(data[k].split()[-1])
            total_C8AA = float(data[k+1].split()[-1])
            total_alpha = float(data[k+2].split()[-1])
            break

    chrg_data = data[start:end]
    atoms, numbers, chrgs = [], [], []
    covCN, C6AA, alpha = [], [], []
    for line in chrg_data:
        item = line.split()
        numbers.append(int(item[0]))
        atoms.append(item[2])
        covCN.append(float(item[3]))
        chrgs.append(float(item[4]))
        C6AA.append(float(item[5]))
        alpha.append(float(item[6]))

    # get atomic properties related to solvent
    start_solv,end_solv = 0,0
    for j in range(0,len(data)):
        if data[j].find("#   Z     Born rad") > -1:
            start_solv = j+1
            break
    for k in range(start_solv,len(data)):
        if data[k].find("total SASA ") > -1:
            end_solv = k-1
            total_SASA = float(data[k].split()[-1])
            break

    solv_data = data[start_solv:end_solv]
    born_rad, SASA, h_bond = [], [], []
    for line in solv_data:
        item = line.split()
        born_rad.append(float(item[3]))
        SASA.append(float(item[4]))
        # in apolar solvents such as CH2Cl2, xTB doesn't return any H bond parameters
        try:
            h_bond.append(float(item[5]))
        except IndexError:
            h_bond.append(float(0))

    return energy, total_charge, homo_lumo, homo, lumo, atoms, numbers, chrgs, dipole_module, Fermi_level, transition_dipole_moment, covCN, C6AA, alpha, homo_occ, lumo_occ, born_rad, SASA, h_bond, total_SASA, total_C6AA, total_C8AA, total_alpha

def read_json(file):
    """
    Takes json file and parses data into pandas table. Returns data.
    """
    if file.find(".json") > -1:
        f = open(file, 'r') # Opening JSON file
        data = json.loads(f.read()) # read file
        f.close()
        return data
    else: pass

def read_thermo(file):
    """
    Read xtb.thermo file. Return thermodata.
    """
    f = open(file, 'r')
    data = f.readlines()
    f.close()

    free_energy, ZPE = np.nan, np.nan

    for i in range(0,len(data)):
        if data[i].find(":: total free energy") > -1:
            free_energy = float(data[i].split()[4])
        if data[i].find(":: zero point energy") > -1:
            ZPE = float(data[i].split()[4])

    return free_energy, ZPE

def freq_disp_detect(data,initial_line,freq_column):
    """
    Grabs freq displacement modules. It requires the lines to read, initial line and freq_column
    (freq column in the file, there are up to three freq columns in the same freq section).
    """
    start_disp = initial_line+7
    # detects where each freq section ends
    for i in range(start_disp,len(data)):
        if len(data[i].split()) <= 3 or i == len(data) - 1:
            stop_disp = i
            # this ensures that the final line is read
            if i == len(data) - 1:
                stop_disp = i + 1
            break

    disp_module = []
    for i in range(start_disp,stop_disp):
        if freq_column == 1:
            x = float(data[i].split()[2])
            y = float(data[i].split()[3])
            z = float(data[i].split()[4])
        elif freq_column == 2:
            x = float(data[i].split()[5])
            y = float(data[i].split()[6])
            z = float(data[i].split()[7])
        elif freq_column == 3:
            x = float(data[i].split()[8])
            y = float(data[i].split()[9])
            z = float(data[i].split()[10])
        module_atom = np.sqrt(x**2 + y**2 + z**2)
        disp_module.append(module_atom)

    return disp_module

def read_freq(file):
    """
    Read xtb.freqs file. Return two lists, one with the freq cm-1 and the other with the module
    of the freq displacement.
    """
    f = open(file, 'r')
    data = f.readlines()
    f.close()

    freq_cm_ind, freq_disp_ind, freq_redmass_ind, freq_IRintens_ind, imag_freqs_ind  = [],[],[],[],[]

    for i in range(0,len(data)):
        if data[i].find("Frequencies --") > -1:
            # append freq cm-1 values and displacement modules
            try:
                n_of_freqs = len(data[i-1].split())
                if n_of_freqs >= 1:
                    freq_cm_ind.append(float(data[i].split()[2]))
                    if float(data[i].split()[2]) < 0:
                        imag_freqs_ind.append(float(data[i].split()[2]))
                    freq_redmass_ind.append(float(data[i+1].split()[3]))
                    freq_IRintens_ind.append(float(data[i+3].split()[3]))
                    disp_module_list = freq_disp_detect(data,i,1)
                    freq_disp_ind.append(disp_module_list)
                if n_of_freqs >= 2:
                    freq_cm_ind.append(float(data[i].split()[3]))
                    if float(data[i].split()[3]) < 0:
                        imag_freqs_ind.append(float(data[i].split()[3]))
                    freq_redmass_ind.append(float(data[i+1].split()[4]))
                    freq_IRintens_ind.append(float(data[i+3].split()[4]))
                    disp_module_list = freq_disp_detect(data,i,2)
                    freq_disp_ind.append(disp_module_list)
                if n_of_freqs == 3:
                    freq_cm_ind.append(float(data[i].split()[4]))
                    if float(data[i].split()[4]) < 0:
                        imag_freqs_ind.append(float(data[i].split()[4]))
                    freq_redmass_ind.append(float(data[i+1].split()[5]))
                    freq_IRintens_ind.append(float(data[i+3].split()[5]))
                    disp_module_list = freq_disp_detect(data,i,3)
                    freq_disp_ind.append(disp_module_list)
                termination_ind = 'Normal'

            except:
                freq_cm_ind, freq_disp_ind, freq_redmass_ind, freq_IRintens_ind, imag_freqs_ind = 'NaN','NaN','NaN','NaN','NaN'
                termination_ind = 'Error_freqs'
                break

    return freq_cm_ind, freq_disp_ind, freq_redmass_ind, freq_IRintens_ind, termination_ind, imag_freqs_ind

def read_fod(file):
    """
    Read xtb.fod files. Return FOD-related properties.
    """
    f = open(file, 'r')
    data = f.readlines()
    f.close()

    # get fractional occupation density (FOD)
    for j in range(0,len(data)):
        if data[j].find("Loewdin FODpop") > -1:
            start_fod = j+1
            total_fod = float(data[j-2].split()[-1])
            break
    for k in range(start_fod,len(data)):
        if data[k].find("Wiberg/Mayer") > -1:
            end_fod = k-1
            break

    fod_data = data[start_fod:end_fod]
    fod, s_prop_fod, p_prop_fod, d_prop_fod = [], [], [], []
    for line in fod_data:
        item = line.split()
        fod.append(float(item[1]))
        s_prop_fod.append(float(item[2]))
        p_prop_fod.append(float(item[3]))
        d_prop_fod.append(float(item[4]))

    return total_fod, fod, s_prop_fod, p_prop_fod, d_prop_fod

def read_gap(file, energy):
    """
    Read xtb.S0toT1gap and xtb.T1toS0gap files. Return energy of vertical excitation/relaxation.
    """
    f = open(file, 'r')
    data = f.readlines()
    f.close()

    for i in range(0,len(data)):
        if data[i].find("SUMMARY") > -1:
            energy_after = float(data[i+2].split()[3])

    gap = energy_after - energy

    return gap

def xyz_to_smiles(file):
    """
    Takes xyz and uses openbabel to convert to smiles
    """
    smi = ""
    try:
        if file.find(".xyz") > -1:
            obConversion = openbabel.OBConversion()
            obConversion.SetInAndOutFormats("xyz", "smi")
            mol = openbabel.OBMol()
            obConversion.ReadFile(mol, file)
            smi = obConversion.WriteString(mol)
        else: pass
    except: pass
    return smi.split('\t')[0]

XTB_OUT = glob.glob('*.json')
for file in XTB_OUT:
    invalid_opt = False
    rootname = os.path.splitext(file)[0]
    energy, total_charge, homo_lumo, homo, lumo, atoms, numbers, chrgs, dipole_module, Fermi_level, transition_dipole_moment, covCN, C6AA, alpha, homo_occ, lumo_occ, born_rad, SASA, h_bond, total_SASA, total_C6AA, total_C8AA, total_alpha = read_xtb(rootname+'.out')
    FPLUS, FMINUS, FRAD = read_fukui(rootname+'.fukui')
    IP, EA, GE = read_omega(rootname+'.omega')
    MULLIKEN, CM5, s_prop, p_prop, d_prop = read_gfn1(rootname+'.gfn1')
    free_energy, ZPE = read_thermo(rootname+'.thermo')
    total_fod, fod, s_prop_fod, p_prop_fod, d_prop_fod = read_fod(rootname+'.fod')
    if args.get_freq:
        try:
            freq_cm, freq_disp, freq_redmass, freq_IRintens, termination, imag_freqs = read_freq(rootname+'.freqs')
        except FileNotFoundError:
            freq_cm, freq_disp, freq_redmass, freq_IRintens, imag_freqs = 'NaN','NaN','NaN','NaN','NaN'
            termination = 'Error_freqs'
    T1toS0gap, S0toT1gap = None, None
    if rootname+'.T1toS0gap' in glob.glob('*.*'):
        T1toS0gap = read_gap(rootname+'.T1toS0gap',energy)
        _, _, homo_lumo_S0_SP, homo_S0_SP, lumo_S0_SP, _, _, chrgs_S0_SP, dipole_module_S0_SP, Fermi_level_S0_SP, _, covCN_S0_SP, C6AA_S0_SP, alpha_S0_SP, homo_occ_S0_SP, lumo_occ_S0_SP, _, _, _, _, total_C6AA_S0_SP, total_C8AA_S0_SP, total_alpha_S0_SP = read_xtb(rootname+'.T1toS0gap')
        total_fod_S0_SP, fod_S0_SP, s_prop_fod_S0_SP, p_prop_fod_S0_SP, d_prop_fod_S0_SP = read_fod(rootname+'.fodS0')
    elif rootname+'.S0toT1gap' in glob.glob('*.*'):
        S0toT1gap = read_gap(rootname+'.S0toT1gap',energy)
        _, _, homo_lumo_T1_SP, homo_T1_SP, lumo_T1_SP, _, _, chrgs_T1_SP, dipole_module_T1_SP, Fermi_level_T1_SP, _, covCN_T1_SP, C6AA_T1_SP, alpha_T1_SP, homo_occ_T1_SP, lumo_occ_T1_SP, _, _, _, _, total_C6AA_T1_SP, total_C8AA_T1_SP, total_alpha_T1_SP = read_xtb(rootname+'.S0toT1gap')
        total_fod_T1_SP, fod_T1_SP, s_prop_fod_T1_SP, p_prop_fod_T1_SP, d_prop_fod_T1_SP = read_fod(rootname+'.fodT1')
    smiles = xyz_to_smiles(rootname+'.xyz')
    bonds, wbos = read_wbo(rootname+'.wbo')

    #create matrix of Wiberg bond-orders
    nat = len(atoms)
    wbo_matrix = np.zeros((nat, nat))
    for i, bond in enumerate(bonds):
        wbo_matrix[(bond[0]-1)][(bond[1]-1)] = wbos[i]
        wbo_matrix[(bond[1]-1)][(bond[0]-1)] = wbos[i]

    """
    Now add disparate XTB outputs to existing json files.
    """
    json_data = read_json(rootname+'.json')
    json_data['Dipole module/D'] = dipole_module
    json_data['Total charge'] = total_charge
    json_data['Transition dipole module/D'] = transition_dipole_moment
    json_data['HOMO'] = homo
    json_data['LUMO'] = lumo
    json_data['HOMO occupancy'] = homo_occ
    json_data['LUMO occupancy'] = lumo_occ
    json_data['electron affinity'] = EA
    json_data['ionization potential'] = IP
    json_data['global electrophilicity'] = GE
    json_data['mulliken charges'] = MULLIKEN
    json_data['cm5 charges'] = CM5
    json_data['FUKUI+'] = FPLUS
    json_data['FUKUI-'] = FMINUS
    json_data['FUKUIrad'] = FRAD
    json_data['s proportion'] = s_prop
    json_data['p proportion'] = p_prop
    json_data['d proportion'] = d_prop
    json_data['Fermi-level/eV'] = Fermi_level
    json_data['Coordination numbers'] = covCN
    json_data['Dispersion coefficient C6'] = C6AA
    json_data['Total dispersion C6'] = total_C6AA
    json_data['Total dispersion C8'] = total_C8AA
    json_data['Polarizability alpha'] = alpha
    json_data['Total polarizability alpha'] = total_alpha
    json_data['Wiberg matrix'] = wbo_matrix.tolist()
    json_data['Total free energy'] = free_energy
    json_data['Zero point energy'] = ZPE
    json_data['Born radii'] = born_rad
    json_data['Atomic SASAs'] = SASA
    json_data['Solvent H bonds'] = h_bond
    json_data['Total SASA'] = total_SASA
    json_data['Total FOD'] = total_fod
    json_data['FOD'] = fod
    json_data['FOD s proportion'] = s_prop_fod
    json_data['FOD p proportion'] = p_prop_fod
    json_data['FOD d proportion'] = d_prop_fod
    if args.get_freq:
        json_data['Frequencies'] = freq_cm
        json_data['Frequency displacement modules'] = freq_disp
        json_data['Frequency reduced masses'] = freq_redmass
        json_data['Frequency IR intensity'] = freq_IRintens
        json_data['Imag_freqs'] = imag_freqs
        json_data['Termination'] = termination
    if T1toS0gap is not None:
        json_data['T1-to-S0 vertical relaxation E'] = T1toS0gap
        json_data['HOMO-LUMO gap at S0/eV'] = homo_lumo_S0_SP
        json_data['HOMO at S0'] = homo_S0_SP
        json_data['LUMO at S0'] = lumo_S0_SP
        json_data['HOMO occupancy at S0'] = homo_occ_S0_SP
        json_data['LUMO occupancy at S0'] = lumo_occ_S0_SP
        json_data['partial charges at S0'] = chrgs_S0_SP
        json_data['Dipole module at S0/D'] = dipole_module_S0_SP
        json_data['Fermi-level at S0/eV'] = Fermi_level_S0_SP
        json_data['Coordination numbers at S0'] = covCN_S0_SP
        json_data['Dispersion coefficient C6 at S0'] = C6AA_S0_SP
        json_data['Total dispersion C6 at S0'] = total_C6AA_S0_SP
        json_data['Total dispersion C8 at S0'] = total_C8AA_S0_SP
        json_data['Polarizability alpha at S0'] = alpha_S0_SP
        json_data['Total polarizability alpha at S0'] = total_alpha_S0_SP
        json_data['Total FOD at S0'] = total_fod_S0_SP
        json_data['FOD at S0'] = fod_S0_SP
        json_data['FOD s proportion at S0'] = s_prop_fod_S0_SP
        json_data['FOD p proportion at S0'] = p_prop_fod_S0_SP
        json_data['FOD d proportion at S0'] = d_prop_fod_S0_SP
    if S0toT1gap is not None:
        json_data['S0-to-T1 vertical excitation E'] = S0toT1gap
        json_data['HOMO-LUMO gap at T1/eV'] = homo_lumo_T1_SP
        json_data['HOMO at T1'] = homo_T1_SP
        json_data['LUMO at T1'] = lumo_T1_SP
        json_data['HOMO occupancy at T1'] = homo_occ_T1_SP
        json_data['LUMO occupancy at T1'] = lumo_occ_T1_SP
        json_data['partial charges at T1'] = chrgs_T1_SP
        json_data['Dipole module at T1/D'] = dipole_module_T1_SP
        json_data['Fermi-level at T1/eV'] = Fermi_level_T1_SP
        json_data['Coordination numbers at T1'] = covCN_T1_SP
        json_data['Dispersion coefficient C6 at T1'] = C6AA_T1_SP
        json_data['Total dispersion C6 at T1'] = total_C6AA_T1_SP
        json_data['Total dispersion C8 at T1'] = total_C8AA_T1_SP
        json_data['Polarizability alpha at T1'] = alpha_T1_SP
        json_data['Total polarizability alpha at T1'] = total_alpha_T1_SP
        json_data['Total FOD at T1'] = total_fod_T1_SP
        json_data['FOD at T1'] = fod_T1_SP
        json_data['FOD s proportion at T1'] = s_prop_fod_T1_SP
        json_data['FOD p proportion at T1'] = p_prop_fod_T1_SP
        json_data['FOD d proportion at T1'] = d_prop_fod_T1_SP
    json_data['SMILES'] = smiles
    json_data['atoms'] = atoms

    with open(rootname+'.json', 'w') as outfile:
        json.dump(json_data, outfile)
