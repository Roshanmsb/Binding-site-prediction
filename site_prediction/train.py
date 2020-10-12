#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 16:28:50 2020

@author: roshan
"""

from torch.optim import Adam
from pytorch_lightning.core.lightning import LightningModule
from dataset import Binding_pocket_dataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from model import UNet, Dice_loss, Ovl
import h5py
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser


class Unetmodel(LightningModule):

    def __init__(self,hparams):
        super(Unetmodel,self).__init__()
        
        self.hparams = hparams
        self.hdf_path = self.hparams.hdf_path
        self.max_dist = self.hparams.max_dist
        self.grid_resolution = self.hparams.grid_resolution
        self.augment = self.hparams.augment
        self.net = UNet()
        self.metric = Ovl('Volume_overlap')
        self.loss = Dice_loss()
        self.lr = self.hparams.lr
        self.batch_size = self.hparams.batch_size
        
    def forward(self, x):
        return self.net(x)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y, y_hat)
        metric = self.metric(y,y_hat)
        tensorboard_logs = {'train_loss': loss,'train_metric':metric}
        return {'loss': loss,'log':tensorboard_logs,'progress_bar':tensorboard_logs}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        return {'train_loss': avg_loss}
                
    def configure_optimizers(self):
        optimizer = Adam(self.net.parameters(), lr=self.lr)
       	return [optimizer]
    
           
    def train_dataloader(self):
        self.data_handle = h5py.File(self.hdf_path, mode='r')
        self.ids = list(self.data_handle.keys())
        self.data_handle.close()
        self.train_dataset = Binding_pocket_dataset(self.hdf_path,self.max_dist, 
                                        self.grid_resolution,self.ids,self.augment)
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)
        return loader
    
 


def parser_args():
    parser = ArgumentParser()
    parser.add_argument("--hdf_path", type=str, default="/home/roshan/site_prediction/dataset.hdf",
                        help="path where dataset is stored")
    parser.add_argument("--lr", type=float, default=0.0003, help="adam: learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--max_dist", type=int, default=35, help="maximum distance for atoms from the center of grid")
    parser.add_argument("--grid_resolution", type=int, default=2, help="resolution for the grid")
    parser.add_argument("--augment", type=bool, default=True, help="perform augmentation of dataset")
    hparams = parser.parse_args()
    return hparams

if __name__ == '__main__':
    trainer = Trainer(gpus=1,max_epochs = 100)
    hparams = parser_args()
    model = Unetmodel(hparams)
    print(model.hparams)
    trainer.fit(model)
    trainer.save_checkpoint("trained_model.ckpt")
