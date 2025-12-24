import openslide
from wsi_core.WholeSlideImage import WholeSlideImage
import os

dataPath = 'DataSource'
seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False, 'keep_ids': [], 'exclude_ids': []}
filter_params = {'a_t':10.0, 'a_h': 16.0, 'max_n_holes':8}
patch_params = {'use_padding': True, 'contour_fn': 'basic', 'patch_level': 0, 'patch_size': 256, 'step_size': 256, 'save_path': 'DataResult/patches'}

def patching(WSI_object, **kwargs):
	print('Patching Function.')
	# Patch
	file_path = WSI_object.process_contours(**kwargs)
	return file_path

def segment(WSI_object, seg_params, filter_params):
    ### Start Seg Timer
    # Segment
    WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

    ### Stop Seg Timers
    return WSI_object

if __name__ == '__main__':
    temp_files = os.listdir(dataPath)
    files = []
    for onefile in temp_files:
        if 'DS_S' not in onefile:
            files.append(onefile)

    file_name = os.path.join(dataPath, files[1])
    print('File name: ', file_name)
    WSI_object = WholeSlideImage(file_name)
    w, h = WSI_object.level_dim[-1] 

    if seg_params['seg_level'] < 0:
        if len(WSI_object.level_dim) == 1:
            seg_params['seg_level'] = 0
        
        else:
            wsi = WSI_object.getOpenSlide()
            best_level = wsi.get_best_level_for_downsample(64)
        seg_params['seg_level'] = best_level
    print('Seg_params in segment: ', seg_params)
    print('filter_params in segment: ', filter_params)


    WSI_object = segment(WSI_object, seg_params, filter_params) 
    # current_vis_params:  {'vis_level': 2, 'line_thickness': 250}

    vis_params = {'vis_level': seg_params['seg_level'], 'line_thickness': 250}
    print('current_vis_params: ', vis_params)
    mask = WSI_object.visWSI(**vis_params)
    mask_path = os.path.join('/Users/wangql/Local/WorkingOn/CLAM/DataResult/masks', files[0][:len(files[0]) - 4]+'.jpg')
    mask.save(mask_path)
        
    print("====================")
    print("Here Patching.")
    print("====================")
    print('current_patch_params: ', patch_params)
    file_path = patching(WSI_object = WSI_object,  **patch_params,)

'''
a_t: area filter threshold for tissue (positive integer, the minimum size of detected foreground contours to consider, relative to a reference patch size of 512 x 512 at level 0, e.g. a value 10 means only detected foreground contours of size greater than 10 512 x 512 sized patches at level 0 will be processed, default: 100)
a_h: area filter threshold for holes (positive integer, the minimum size of detected holes/cavities in foreground contours to avoid, once again relative to 512 x 512 sized patches at level 0, default: 16)
max_n_holes: maximum of holes to consider per detected foreground contours (positive integer, default: 10, higher maximum leads to more accurate patching but increases computational cost)


progress: 0.00, 0/1
processing TCGA-HT-7608-01A-01-BS1.0b72cd2c-0eaa-4b6e-b164-3ec6a570dd42.svs
vis_level:  2
seg_level:  2
====================
Here Segmentation.
current_seg_params {'seg_level': 2, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False, 'keep_ids': [], 'exclude_ids': []}
current_filter_params {'seg_level': 2, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False, 'keep_ids': [], 'exclude_ids': []}
====================
Image size:  (3213, 2763)
Seg_level:  2
Image size:  (2763, 3213, 4)
Scale:  (16.0, 16.002533478103512)
a_t:  104652900.0
a_h:  (16.0, 16.002533478103512)
'''



'''
{'seg_level': 2, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False, 'keep_ids': [], 'exclude_ids': []}
current_filter_params {'seg_level': 2, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False, 'keep_ids': [], 'exclude_ids': []}

{'seg_level': 2, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False, 'keep_ids': 'none', 'exclude_ids': 'none'}
filter_params:  {'a_t': 100.0, 'a_h': 16.0, 'max_n_holes': 8}

'''


'''
keep_ids:  []
contour_ids:  {0}

keep_ids:  []
contour_ids:  set()
'''


'''
current_patch_params:  {'use_padding': True, 'contour_fn': 'four_pt', 'patch_level': 0, 'patch_size': 256, 'step_size': 256, 'save_path': 'DataResult/patches'}
'''


