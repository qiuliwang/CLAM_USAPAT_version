import os
import numpy as np
import math
import re
import pdb
import pickle
import cv2
from PIL import Image, ImageStat
import h5py
import scipy
import csv 
import PIL

image_dir = 'PRIVATE_Extracted_Patch' 
data_slide_dir  = 'E:\\Data\\MALI_20240603\\SVS_Slide\\HE'
patch_size = 1024

svs_files = []
temp = os.listdir(data_slide_dir)
for one in temp:
    if '.svs' in one:
        svs_files.append(one)

# print(svs_files)

def readCSV(filename):
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines

svs_info = readCSV('mask_info.csv')

ids = []
for one in svs_info:
    ids.append(one[0])

# print((ids))
'''
get all patches and classify them with id
'''
all_patches = os.listdir('PRIVATE_Extracted_Patch')
patch_bag = {}
for one_id in ids:
    for one_patch in all_patches:
        if one_id in one_patch:
            if one_id not in patch_bag.keys():
                patch_bag[one_id] = []
                patch_bag[one_id].append(one_patch)
            else:
                patch_bag[one_id].append(one_patch)

'''
join all patch according id
'''
import tqdm

for one_id in tqdm.tqdm(ids):
    if one_id not in patch_bag.keys():
        print(one_id)
    else:
        bag = patch_bag[one_id]
        info = []
        for one in svs_info:
            if one_id == one[0]:
                info = one
                break

        # print(info)
        width = int(info[1])
        hight = int(info[2])
        shape = [width, hight, 3]
        sketch = np.zeros(shape)
        print(sketch.shape)
        patch_size = 1024
        # 16635, 18007,
        for one_patch in tqdm.tqdm(bag):
            # print(one_patch)
            if 'mask' not in one_patch:
                temp = one_patch.split('.png')[0]
                # print(temp)
                location = temp.split('_')
                x = int(location[1]) // 4
                y = int(location[2]) // 4
                img = cv2.resize(cv2.imread(os.path.join('PRIVATE_Extracted_Patch', one_patch)), (patch_size, patch_size))
                sketch[y : y + patch_size, x : x + patch_size, :] = img
        cv2.imwrite(one[0] + '_ddd.png', sketch)