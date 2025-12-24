'''
Getting vessels from slices.
Each patch has a vessel in its center.
'''

import json
import os
import openslide
import numpy as np 
from PIL import ImageDraw, Image
from scipy.ndimage.morphology import binary_fill_holes
import cv2
import matplotlib.pyplot as plt 
import tqdm
import threading

cases = os.listdir('/home/wangqiuli/Data/Liver_TLS/WSI')
count = 0
sign = 0
print(len(cases))

training_list = cases
for one_case in cases:
    if one_case in training_list:
        if not os.path.exists(one_case.split('.')[0] + '.png'):

            print(one_case)
            wsi_path = '/home/wangqiuli/Data/Liver_TLS/WSI/' + one_case

            slide = openslide.OpenSlide(wsi_path)
            level_index = 2

            print('Image dimension: ', slide.level_dimensions[level_index])
            print('Image downsamples: ', slide.level_downsamples[level_index])
            slide_pixels = slide.read_region((0, 0), level_index, slide.level_dimensions[level_index])
            slide_pixels = slide_pixels.convert('RGB')
            mask_shape = slide.level_dimensions[level_index]
            print('Shape: ', slide.level_dimensions[level_index])

            slide_pixels.save(one_case.split('.')[0] + '.png')

# print(training_list)