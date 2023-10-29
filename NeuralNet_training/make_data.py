import torch
import torchani
import os
import math
import tqdm
import pandas as pd
import numpy as np
from torch import nn
from torch.nn.functional import pad
import torch.nn.functional as F
import torchani.utils
import gc
from collections import Counter
# helper function to convert energy unit from Hartree to kcal/mol
from torchani.units import hartree2kcalmol
from main_functions import *

def make_data(label_col, subtractE = 0, csv_file = '631Gd_ani1x_master'):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_ANI1x = torchani.models.ANI1x(periodic_table_index=True).to(device)
    model_ANI1x_0 = model_ANI1x[0]
    aev_computer = model_ANI1x.aev_computer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data = pd.read_csv(f'{csv_file}.csv')
    molecules = list(data['molecule'].unique())
    atoms_per_mol = data.groupby(['mol_id', 'molecule']).agg({'atom':'count'}).reset_index().drop('mol_id', axis=1).drop_duplicates()
    atoms_dict = atoms_per_mol.set_index('molecule').to_dict()['atom']
    anum2symbol = {1 : 'H', 6 : 'C', 7 : 'N', 8 : 'O'}

    all_coordinates = []
    all_species = []
    all_cspecies = []
    all_elecpots = []
    all_energies = []
    mol_names = []
    molecules = [mol for mol in data['molecule'].unique()]
    for mol in molecules:
        mol_df = data[(data['molecule']==mol)]
        mol_ids = mol_df['mol_id'].unique()
        n_steps = mol_ids.shape[0]
        mol_nums = mol_df[(mol_df['mol_id']==mol_ids[0])]['atomic_num'].to_numpy()
        nqm_atoms = len(mol_nums)
        sub_energy = 0
        for an in mol_nums:
            sub_energy += model_ANI1x.sae_dict[anum2symbol[an]]
        species = torch.tensor(mol_nums)
        species = pad_mol(species, mol, -1, nqm_atoms, max(atoms_dict.values())).expand(n_steps, -1)
        cspecies = torch.tensor(convert_species(mol_nums))
        cspecies = pad_mol(cspecies, mol, -1, nqm_atoms, max(atoms_dict.values())).expand(n_steps, -1)
        coords, energies, elecpot, mol_ids = pd2arr(data, mol, device, label_col, n_steps, nqm_atoms)
        coords = pad_mol(coords, mol, 0, nqm_atoms, max(atoms_dict.values()))
        elecpot = pad_mol(elecpot, mol, 0, nqm_atoms, max(atoms_dict.values()))
        if subtractE == 1:
            new_energies = energies-sub_energy
        else:
            new_energies = energies
        all_coordinates.append(coords)
        all_species.append(species)
        all_cspecies.append(cspecies)
        all_elecpots.append(elecpot)
        all_energies.append(new_energies)
        mol_names.extend(mol_ids)

    coordinates = torch.cat(all_coordinates).to(device, dtype=torch.float32)
    cspecies = torch.cat(all_cspecies).to(device)
    species = torch.cat(all_species).to(device, dtype=torch.float32)
    elecpots = torch.cat(all_elecpots).to(device, dtype=torch.float32)
    energies = torch.cat(all_energies).reshape(-1).to(device)
    aevs = aev_computer((cspecies, coordinates))[1]
    aevs_elecpots = torch.cat([aevs, elecpots], axis=2)



    # get last activation layer
    A6_energies = []
    A4_outputs = []
    atoms_idx_dict = {1: 0, 6: 1, 7: 2, 8: 3}
    mnum2symbol = {0 : 'H', 1 : 'C', 2 : 'N', 3 : 'O'}
    for idx in range(0, species.shape[0]):
        species_idx = species[idx]
        species_inp = species_idx[species_idx>=0].type(torch.int64).to(device)
        species_inp = torch.tensor([atoms_idx_dict[i.item()] for i in species_inp]).reshape(1, -1).to(device)
        respecies_inp = torch.tensor(reconvert_species(species_inp[0].cpu().numpy())).reshape(1, -1).to(device)
        coords_idx = coordinates[idx]
        coords_inp = coords_idx[species_idx>=0].reshape(1, -1, 3).to(device)
        aevs_inp = model_ANI1x.aev_computer((species_inp, coords_inp))[1].to(device)
        sub_energy = 0
        for an in species_inp[0]:
            an_item = an.item()
            sub_energy += model_ANI1x.sae_dict[mnum2symbol[an_item]]
        # energies predicted by forward prop
        a6_energy, a4_output = forward_prop(species_inp, aevs_inp, model_ANI1x_0)
        a4_output = a4_output.reshape(1, a4_output.shape[0], a4_output.shape[1])
        a4_output = pad_mol(a4_output, a4_output.shape[1], 0, a4_output.shape[1], max(atoms_dict.values()))
        A6_energies.append(a6_energy)
        A4_outputs.append(a4_output)
        
    A6_energies = torch.tensor(A6_energies).to(device)
    A4_outputs = torch.cat(A4_outputs, 0).to(device)
    A4_elecpots = torch.cat((A4_outputs, elecpots), -1)

    return(species, cspecies, aevs_elecpots, energies, mol_names)

