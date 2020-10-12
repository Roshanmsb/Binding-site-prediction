#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 18:30:53 2020

@author: roshan
"""
import torch
from pytorch_lightning.metrics import TensorMetric



class UNet_down_block(torch.nn.Module):
    def __init__(self, input_channel, output_channel, pool_size,down_size):
        super(UNet_down_block, self).__init__()
        self.conv1 = torch.nn.Conv3d(input_channel, output_channel, 3,padding=1)
        self.conv2 = torch.nn.Conv3d(output_channel, output_channel, 3,padding=1)
        self.max_pool = torch.nn.MaxPool3d(pool_size)
        self.relu = torch.nn.ReLU()
        self.down_size = down_size

    def forward(self, x):
        if self.down_size:
            x = self.max_pool(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x
    
class UNet_up_block(torch.nn.Module):
    def __init__(self, prev_channel, input_channel, output_channel,size):
        super(UNet_up_block, self).__init__()
        self.up_sampling = torch.nn.Upsample(scale_factor=size)
        self.conv1 = torch.nn.Conv3d(prev_channel + input_channel, output_channel, 3, padding=1)
        self.conv2 = torch.nn.Conv3d(output_channel, output_channel, 3, padding=1)
        self.relu = torch.nn.ReLU()
        
    def forward(self, prev_feature_map, x):
        x = self.up_sampling(x)
        x = torch.cat((x, prev_feature_map), dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.down_block1 = UNet_down_block(18, 32,2,False)
        self.down_block2 = UNet_down_block(32, 64,2,True)
        self.down_block3 = UNet_down_block(64, 128,2,True)
        self.down_block4 = UNet_down_block(128, 256,3,True)
        self.max_pool = torch.nn.MaxPool3d(3)
        
        

        self.mid_conv1 = torch.nn.Conv3d(256, 512, 3, padding=1)
        self.mid_conv2 = torch.nn.Conv3d(512, 512, 3, padding=1)

        self.up_block1 = UNet_up_block(256, 512, 256,3)
        self.up_block2 = UNet_up_block(128, 256, 128,3)
        self.up_block3 = UNet_up_block(64, 128, 64,2)
        self.up_block4 = UNet_up_block(32, 64, 32,2)
        

        self.last_conv1 = torch.nn.Conv3d(32, 1, 1, padding=0)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        self.x5 = self.max_pool(self.x4)
        self.x6 = self.relu(self.mid_conv1(self.x5))
        self.x6 = self.relu(self.mid_conv2(self.x6))
        x = self.up_block1(self.x4, self.x6)
        x = self.up_block2(self.x3, x)
        x = self.up_block3(self.x2, x)
        x = self.up_block4(self.x1, x)
        x = self.sigmoid(self.last_conv1(x))
        return x
    
class Ovl(TensorMetric):
    def forward(self, y_true, y_pred,smooth = 0.01):
        concat = torch.cat((y_true,y_pred),dim=1)
        return ((concat.min(dim=1)[0].sum() + smooth)
                / (concat.max(dim=1)[0].sum() + smooth))
    
class Dice_loss(torch.nn.Module):
    def __init__(self):
        super(Dice_loss,self).__init__()
        
    def forward(self,x,y,smooth=0.01):
        iflat = torch.flatten(x)
        tflat = torch.flatten(y)
        intersection = (iflat * tflat).sum()

        return 1 - ((2. * intersection + smooth) /
                  (iflat.sum() + tflat.sum() + smooth))