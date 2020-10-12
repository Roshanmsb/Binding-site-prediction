#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 18:19:52 2020

@author: roshan
"""
import h5py
import numpy as np
import re
import os
from openbabel import pybel


class Featurizer():
    def __init__(self, smarts=True):
        self.smarts = smarts
    
    def pocket_featurizer(self,mol):
        coords = []

        for a in mol.atoms:
            coords.append(a.coords)
        coords = np.array(coords)
        features = np.ones((len(coords), 1))
        return coords, features

    def encode_num(self,atomic_num):
        ATOM_CODES = {}
        metals = ([3, 4, 11, 12, 13] + list(range(19, 32))
                          + list(range(37, 51)) + list(range(55, 84))
                          + list(range(87, 104)))
        atom_classes = [
                    (5, 'B'),
                    (6, 'C'),
                    (7, 'N'),
                    (8, 'O'),
                    (15, 'P'),
                    (16, 'S'),
                    (34, 'Se'),
                    ([9, 17, 35, 53], 'halogen'),
                    (metals, 'metal')
                ]
        for code, (atom, name) in enumerate(atom_classes):
                    if type(atom) is list:
                        for a in atom:
                            ATOM_CODES[a] = code
                    else:
                        ATOM_CODES[atom] = code
        NUM_ATOM_CLASSES = len(atom_classes)
        encoding = np.zeros(NUM_ATOM_CLASSES)
        try:
            encoding[ATOM_CODES[atomic_num]] = 1.0
        except:
            pass
        return encoding

    def named_prop(self,atom):
        NAMED_PROPS = ['hyb', 'heavydegree', 'heterodegree','partialcharge']
        prop = [atom.__getattribute__(prop) for prop in NAMED_PROPS]
        return prop

    def smart_feats(self,molecule):
        __PATTERNS = []
        SMARTS = [
                    '[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]',
                    '[a]',
                    '[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]',
                    '[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]',
                    '[r]'
                ]
        smarts_labels = ['hydrophobic', 'aromatic', 'acceptor', 'donor',
                                 'ring']
        for smarts in SMARTS:
            __PATTERNS.append(pybel.Smarts(smarts))
        features = np.zeros((len(molecule.atoms), len(__PATTERNS)))

        for (pattern_id, pattern) in enumerate(__PATTERNS):
            atoms_with_prop = np.array(list(*zip(*pattern.findall(molecule))),
                                           dtype=int) - 1
            features[atoms_with_prop, pattern_id] = 1.0
        return features

    def protein_featurizer(self,mol):
        coords = []
        features = []
        heavy_atoms = []
        for i,atom in enumerate(mol.atoms):
            if atom.atomicnum > 1:
                heavy_atoms.append(i)
                coords.append(atom.coords)
                features.append(np.concatenate((
                            self.encode_num(atom.atomicnum),
                            self.named_prop(atom),
                        )))
        coords = np.array(coords, dtype=np.float32)
        features = np.array(features, dtype=np.float32)
        if self.smarts:
            features = np.hstack([features,self.smart_feats(mol)[heavy_atoms]])

        return coords,features
        
        
class Prepare_data():
    def __init__(self, data_dir,hdf_path):
        self.data_dir = data_dir
        self.hdf_path = hdf_path
        self.featurizer = Featurizer()
        
    def prepare_dataset(self,hdf_mode='w'):

        get_id = lambda structure_id: re.sub('_[0-9]+$', '', structure_id)
        
        data_path = self.data_dir

        ids = os.listdir(data_path)

        multiple_pockets = {}

        with h5py.File(self.hdf_path, mode=hdf_mode) as f:
            for structure_id in iter(ids):
                protein = next(pybel.readfile('mol2',os.path.join(data_path,structure_id,'protein.mol2')))
                pocket = next(pybel.readfile('mol2',os.path.join(data_path,structure_id,'cavity6.mol2')))

                pocket_coords, pocket_features = self.featurizer.pocket_featurizer(pocket)
                prot_coords, prot_features = self.featurizer.protein_featurizer(protein)

                centroid = prot_coords.mean(axis=0)
                pocket_coords -= centroid
                prot_coords -= centroid

                group_id = get_id(structure_id)
                if group_id in f:
                    group = f[group_id]
                    if not np.allclose(centroid, group['centroid'][:], atol=0.5):
                        print('Structures for %s are not aligned, ignoring pocket %s' % (group_id, structure_id))
                        continue
                        
                    multiple_pockets[group_id] = multiple_pockets.get(group_id, 1) + 1

                    for key, data in (('pocket_coords', pocket_coords),
                                      ('pocket_features', pocket_features)):
                        data = np.concatenate((group[key][:], data))
                        del group[key]
                        group.create_dataset(key, data=data, shape=data.shape, dtype='float32', compression='lzf')
                else:
                    group = f.create_group(group_id)
                    for key, data in (('coords', prot_coords),
                                      ('features', prot_features),
                                      ('pocket_coords', pocket_coords),
                                      ('pocket_features', pocket_features),
                                      ('centroid', centroid)):
                        group.create_dataset(key, data=data, shape=data.shape, dtype='float32', compression='lzf')


          
    
    
