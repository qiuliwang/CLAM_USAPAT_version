import openslide
import os

data_path = 'DataSource/'

files_ori = os.listdir(data_path)
files = []
for onefile in files_ori:
    if 'DS_Store' not in onefile:
        files.append(onefile)


for onefile in files:
    temp_file = os.path.join(data_path, onefile)
    print(temp_file)
    slide = openslide.OpenSlide(temp_file)

    print('Level Count: ', slide.level_count)
    
    dimensions = slide.level_dimensions
    print('Dimensions: ', dimensions)

    downsamples = slide.level_downsamples
    print('Downsamples: ', downsamples)

    # read_region(location, level, size)，Return an RGBA Image containing the contents of the specified region.
    # Parameters:  
    # location (tuple) tuple giving the top left pixel in the level 0 reference frame
    # level (int)  the level number
    # size (tuple) (width, height) tuple giving the region size
    # read_region()返回一个数组的rgba数值数组
    slide.read_region((0,0),8,slide.level_dimensions[8])