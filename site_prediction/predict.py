#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 08:34:51 2020

@author: roshan
"""

from train import Unetmodel,parser_args
from openbabel import pybel
import numpy as np
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing
from argparse import ArgumentParser
import torch
from prepare_data import Featurizer


parser = ArgumentParser()
parser.add_argument("--pdb_file", type=str, default="/home/roshan/site_prediction/protein.mol2",
                        help="path where dataset pdb file is stored")
filename = parser.parse_args()
hparams = parser_args()
model = Unetmodel(hparams)
new_model = model.load_from_checkpoint('trained_model.ckpt')
mol = next(pybel.readfile('mol2',filename.pdb_file))

def make_grid(coords, features, grid_resolution=2, max_dist=35):

    coords = np.asarray(coords, dtype=np.float)
    c_shape = coords.shape
    N = len(coords)
    features = np.asarray(features, dtype=np.float)
    f_shape = features.shape
    num_features = f_shape[1]
    max_dist = float(max_dist)
    grid_resolution = float(grid_resolution)

    box_size = int(np.ceil(2 * max_dist / grid_resolution + 1))


    grid_coords = (coords + max_dist) / grid_resolution
    grid_coords = grid_coords.round().astype(int)


    in_box = ((grid_coords >= 0) & (grid_coords < box_size)).all(axis=1)
    grid = np.zeros((1, box_size, box_size, box_size, num_features),dtype = np.float32)
    for (x, y, z), f in zip(grid_coords[in_box], features[in_box]):
        grid[0, x, y, z] += f

    return grid

def pocket_density_from_mol(mol):

    prot_coords, prot_features = Featurizer().protein_featurizer(mol)
    centroid = prot_coords.mean(axis=0)
    prot_coords -= centroid
    x = make_grid(prot_coords, prot_features,
                             max_dist=hparams.max_dist,
                             grid_resolution=hparams.grid_resolution)
    x = x.transpose(0,4,3,2,1)
    grid = torch.tensor(x)
    density = new_model(grid)
    density = density.detach().numpy()
    density = density.transpose(0,4,3,2,1)
    origin = (centroid - hparams.max_dist)
    step = np.array([hparams.grid_resolution] * 3)
    return density, origin, step


def get_pockets_segmentation(density, threshold=0.5, min_size=50):


        voxel_size = (2) ** 3
        bw = closing((density[0] > threshold).any(axis=-1))

        cleared = clear_border(bw)

        label_image, num_labels = label(cleared, return_num=True)
        for i in range(1, num_labels + 1):
            pocket_idx = (label_image == i)
            pocket_size = pocket_idx.sum() * voxel_size
            if pocket_size < min_size:
                label_image[np.where(pocket_idx)] = 0
        return label_image
    
def predict_pocket_atoms(mol, dist_cutoff=5, expand_residue=True):
    from scipy.spatial.distance import cdist

    coords = np.array([a.coords for a in mol.atoms])
    atom2residue = np.array([a.residue.idx for a in mol.atoms])
    residue2atom = np.array([[a.idx - 1 for a in r.atoms]
                             for r in mol.residues])

    # predcit pockets
    density, origin, step = pocket_density_from_mol(mol)
    pockets = get_pockets_segmentation(density)

    # find atoms close to pockets
    pocket_atoms = []
    for pocket_label in range(1, pockets.max() + 1):
        indices = np.argwhere(pockets == pocket_label).astype('float32')
        indices *= step
        indices += origin
        distance = cdist(coords, indices)
        close_atoms = np.where((distance < dist_cutoff).any(axis=1))[0]
        if len(close_atoms) == 0:
            continue
        if expand_residue:
            residue_ids = np.unique(atom2residue[close_atoms])
            close_atoms = np.concatenate(residue2atom[residue_ids])
        pocket_atoms.append([int(idx) for idx in close_atoms])


    pocket_mols = []

    for pocket in pocket_atoms:
        pocket_mol = mol.clone
        atoms_to_del = (set(range(len(pocket_mol.atoms)))
                        - set(pocket))
        pocket_mol.OBMol.BeginModify()
        for aidx in sorted(atoms_to_del, reverse=True):
            atom = pocket_mol.OBMol.GetAtom(aidx + 1)
            pocket_mol.OBMol.DeleteAtom(atom)
        pocket_mol.OBMol.EndModify()
        pocket_mols.append(pocket_mol)

    return pocket_mols

pocket_mols = predict_pocket_atoms(mol)

for i, pocket in enumerate(pocket_mols):
    pocket.write('pdb', 'pocket%i.pdb' % i, overwrite=True)
