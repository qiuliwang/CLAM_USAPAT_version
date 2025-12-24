#!/usr/bin/env python
# coding: utf-8

# In[9]:


import openslide 
import os
import numpy as np


# In[10]:


data_path = 'DataSource/'

files_ori = os.listdir(data_path)
files = []
for onefile in files_ori:
    if 'DS_Store' not in onefile:
        files.append(onefile)
print(files)


# In[11]:


for onefile in files:
    temp_file = os.path.join(data_path, onefile)
    print(temp_file)
    slide = openslide.OpenSlide(temp_file)

    print('Level Count: ', slide.level_count)
    
    dimensions = slide.level_dimensions
    print('Dimensions: ', dimensions)

    downsamples = slide.level_downsamples
    print('Downsamples: ', downsamples)


# In[34]:


slide = openslide.OpenSlide(os.path.join(data_path, files[2]))
print('Level Count: ', slide.level_count)
dimensions = slide.level_dimensions
print('Dimensions: ', dimensions)
downsamples = slide.level_downsamples
print('Downsamples: ', downsamples)


# In[36]:


# read_region(location, level, size)，Return an RGBA Image containing the contents of the specified region.
# Parameters:  
# location (tuple) tuple giving the top left pixel in the level 0 reference frame
# level (int)  the level number
# size (tuple) (width, height) tuple giving the region size
# read_region()返回一个数组的rgba数值数组
slide = openslide.OpenSlide(os.path.join(data_path, files[2]))
read_slide = slide.read_region((0,0),2,slide.level_dimensions[2])
slide = np.array(read_slide)
print(slide.shape)


# In[32]:


for one_dimension in dimensions:
    print(one_dimension)
    print(one_dimension[0] / one_dimension[1])


# In[13]:


import imageio
imageio.imwrite('test.png', slide)


# In[37]:


from PIL import Image
slide = openslide.OpenSlide(os.path.join(data_path, files[2]))
dimensions = slide.level_count
for one_dimension in range(dimensions):
    if one_dimension == 2:
        print(one_dimension)
        read_slide = slide.read_region((0,0), one_dimension, slide.level_dimensions[one_dimension])
        print(read_slide.mode)
        read_slide = read_slide.convert('RGB')
        print(read_slide.mode)

    read_slide.save('read_slide_' + str(one_dimension) + '.png')
    #     slide = np.array(read_slide)
#     print(slide.shape)


# In[ ]:




