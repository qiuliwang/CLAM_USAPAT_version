from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
import random
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils, models
import torch.nn.functional as F
import cv2
from PIL import Image, ImageStat
import h5py
import scipy
from random import randrange
import cv2

def eval_transforms(pretrained=False):
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    else:
        mean = (0.5,0.5,0.5)
        std = (0.5,0.5,0.5)

    trnsfrms_val = transforms.Compose(
                    [
                     transforms.ToTensor(),
                     transforms.Normalize(mean = mean, std = std)
                    ]
                )

    return trnsfrms_val

class Whole_Slide_Bag(Dataset):
    def __init__(self,
        file_path,
        pretrained=False,
        custom_transforms=None,
        target_patch_size=-1,
        ):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
        """
        self.pretrained=pretrained
        if target_patch_size > 0:
            self.target_patch_size = (target_patch_size, target_patch_size)
        else:
            self.target_patch_size = None

        if not custom_transforms:
            self.roi_transforms = eval_transforms(pretrained=pretrained)
        else:
            self.roi_transforms = custom_transforms

        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = f['imgs']
            self.length = len(dset)

        self.summary()
            
    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file['imgs']
        for name, value in dset.attrs.items():
            print(name, value)

        print('pretrained:', self.pretrained)
        print('transformations:', self.roi_transforms)
        if self.target_patch_size is not None:
            print('target_size: ', self.target_patch_size)

    def __getitem__(self, idx):
        with h5py.File(self.file_path,'r') as hdf5_file:
            img = hdf5_file['imgs'][idx]
            coord = hdf5_file['coords'][idx]
        
        img = Image.fromarray(img)
        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)
        img = self.roi_transforms(img).unsqueeze(0)
        return img, coord

class Whole_Slide_Bag_FP_(Dataset):
    def __init__(self,
        file_path,
        wsi,
        pretrained=False,
        custom_transforms=None,
        custom_downsample=1,
        target_patch_size=-1
        ):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
            custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
            target_patch_size (int): Custom defined image size before embedding
        """
        self.pretrained=pretrained
        self.wsi = wsi
        if not custom_transforms:
            self.roi_transforms = eval_transforms(pretrained=pretrained)
        else:
            self.roi_transforms = custom_transforms

        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = f['coords']
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
            self.length = len(dset)
            if target_patch_size > 0:
                self.target_patch_size = (target_patch_size, ) * 2
            elif custom_downsample > 1:
                self.target_patch_size = (self.patch_size // custom_downsample, ) * 2
            else:
                self.target_patch_size = None
        self.summary()
            
    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file['coords']
        for name, value in dset.attrs.items():
            print(name, value)

        print('\nfeature extraction settings')
        print('target patch size: ', self.target_patch_size)
        print('pretrained: ', self.pretrained)
        print('transformations: ', self.roi_transforms)
        print('dset: ', len(dset))

    def __getitem__(self, idx):
        with h5py.File(self.file_path,'r') as hdf5_file:
            coord = hdf5_file['coords'][idx]
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
        print('Image size:', img.size)
        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)
        img = self.roi_transforms(img).unsqueeze(0)
        return img, coord

class Whole_Slide_Bag_FP(Dataset):
    def __init__(self,
        file_path,
        wsi,
        slide_id,
        pretrained=False,
        custom_transforms=None,
        custom_downsample=1,
        target_patch_size=-1,
        feat_dir = ''
        ):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
            custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
            target_patch_size (int): Custom defined image size before embedding
        """
        self.pretrained=pretrained
        self.wsi = wsi
        self.slide_id = slide_id
        if not custom_transforms:
            self.roi_transforms = eval_transforms(pretrained=pretrained)
        else:
            self.roi_transforms = custom_transforms

        self.file_path = file_path
        self.feat_dir = feat_dir
        with h5py.File(self.file_path, "r") as f:

            dset = f['coords']
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
            self.length = len(dset)
            if target_patch_size > 0:
                self.target_patch_size = (target_patch_size, ) * 2
            elif custom_downsample > 1:
                self.target_patch_size = (self.patch_size // custom_downsample, ) * 2
            else:
                self.target_patch_size = None
        self.summary()
              
    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file['coords']
        print(dset.attrs.items())
        for name, value in dset.attrs.items():
            print(name, value)

        print('\nfeature extraction settings')
        print('target patch size: ', self.target_patch_size)
        print('pretrained: ', self.pretrained)
        print('transformations: ', self.roi_transforms)
        print('dset: ', len(dset))

    def __getitem__(self, idx):
        with h5py.File(self.file_path,'r') as hdf5_file:
            coord = hdf5_file['coords'][idx]
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
        status = ImageStat.Stat(img).var
        # if status[0] > 100 and status[0] < 6000 and status[1] > 100 and status[1] < 6000 and status[2] > 100 and status[2] < 6000:
        #     if random.randint(1,10) % 5 == 0:
        img.save(os.path.join(self.feat_dir, self.slide_id + '_' + str(coord[0]) + '_' + str(coord[1]) + '.jpeg'))
        # + '_' + str(ImageStat.Stat(img).var)
        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)
        img = self.roi_transforms(img).unsqueeze(0)
        return img, coord

class Dataset_All_Bags(Dataset):

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df['slide_id'][idx]





class Whole_Slide_Bag_FP_mask(Dataset):
    def __init__(self,
        file_path,
        wsi,
        slide_id,
        mask, 
        pretrained=False,
        custom_transforms=None,
        custom_downsample=1,
        target_patch_size=-1
        ):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
            custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
            target_patch_size (int): Custom defined image size before embedding
        """
        self.pretrained=pretrained
        self.wsi = wsi
        self.slide_id = slide_id
        self.mask = mask
        if not custom_transforms:
            self.roi_transforms = eval_transforms(pretrained=pretrained)
        else:
            self.roi_transforms = custom_transforms

        self.file_path = file_path
        with h5py.File(self.file_path, "r") as f:

            dset = f['coords']
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
            self.length = len(dset)
            if target_patch_size > 0:
                self.target_patch_size = (target_patch_size, ) * 2
            elif custom_downsample > 1:
                self.target_patch_size = (self.patch_size // custom_downsample, ) * 2
            else:
                self.target_patch_size = None
        self.summary()
              
    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file['coords']
        print(dset.attrs.items())
        for name, value in dset.attrs.items():
            print(name, value)

        print('\nfeature extraction settings')
        print('target patch size: ', self.target_patch_size)
        print('pretrained: ', self.pretrained)
        print('transformations: ', self.roi_transforms)
        print('dset: ', len(dset))

    def __getitem__(self, idx):
        with h5py.File(self.file_path,'r') as hdf5_file:
            coord = hdf5_file['coords'][idx]

        # if not os.path.exists('TCGA_DataResult_padding/' + self.slide_id + '_' + str(coord[0]) + '_' + str(coord[1]) + '_mask.png'):
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
        status = ImageStat.Stat(img).var
        # print(img.size)

        # print(coord)

        shape = self.mask.size

        # inner_mask = self.mask.crop((coord[0], coord[1], coord[0] + self.patch_size, coord[1] + self.patch_size)).convert('RGB')
        inner_mask = self.mask.crop((coord[0] / 4, coord[1] / 4, coord[0] / 4 + self.patch_size, coord[1] / 4 + self.patch_size)).convert('RGB')
        inner_mask_npy = np.array(inner_mask)
        inner_mask_npy = np.where(inner_mask_npy > 127, 255, 0)

        # if np.array(inner_mask).max() > 0:
        # print(os.path.join('Glioma_Extract_Patch', self.slide_id + '_' + str(coord[0]) + '_' + str(coord[1]) + '_mask.npy'))
        feat_dir = 'Glioma_Extract_Patch'
        # print(feat_dir)
        inner_mask.save(os.path.join(feat_dir, self.slide_id + '_' + str(coord[0]) + '_' + str(coord[1]) + '_mask.png'))
        inner_mask_npy = np.array(inner_mask)
        np.save(os.path.join(feat_dir, self.slide_id + '_' + str(coord[0]) + '_' + str(coord[1]) + '_mask.npy'), inner_mask_npy)
        img.save(os.path.join(feat_dir,  self.slide_id + '_' + str(coord[0]) + '_' + str(coord[1]) + '.png'))

        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)
        img = self.roi_transforms(img).unsqueeze(0)
        
        return img, coord