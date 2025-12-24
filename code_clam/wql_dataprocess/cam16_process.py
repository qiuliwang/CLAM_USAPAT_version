import csvTools
import os

ids = ['case_id', 'slide_id', 'label']

'''
CAML data structure:
    patient_0,slide_0,tumor_tissue
    patient_0,slide_1,normal_tissue
'''

'''
CAMELYON17 data structure:
    patient_000_node_0.tif,negative
    patient_000_node_1.tif,negative
    patient_000_node_2.tif,negative
    patient_000_node_3.tif,negative
    patient_000_node_4.tif,negative
'''
'''
CAMELYON16 data structure:
    normal/normal_084.tif
    tumor/tumor_010.tif
'''
camelyon17_data = csvTools.readCSV('/home1/qiuliwang/Data/CAMELYON17/training/stage_labels.csv')
print('number of lines: ', len(camelyon17_data))

def split_label(temp):
    # print(temp[0][11])
    # return 0

    patient_id = temp[0][:11]
    slide_id = patient_id + '_' + temp[0][12:len(temp[0]) - 4]

    if temp[1] == 'negative':
        label = 'subtype_1'
    else:
        label = 'subtype_2'
    return [patient_id, slide_id, label]

def split_16_label(temp):
    patient_id = 'NAN'
    slide_id = temp[:len(temp) - 4]
    if 'normal' in temp:
        label = 'subtype_1'
    else:
        label = 'subtype_2'
    return [patient_id, slide_id, label]

process_label = []
for oneline in camelyon17_data[1:]:
    temp = oneline
    filename = temp[0]
    if '.zip' not in filename:
        temp_label = split_label(temp)
        process_label.append(temp_label)

data_dir = '/home1/qiuliwang/Data/CAMELYON17/training/center_0/'
files = os.listdir(data_dir)
for one_file in files:
    if '_node_' not in one_file:
        temp_label = split_16_label(one_file)
        process_label.append(temp_label)
import random
random.shuffle(process_label)
csvTools.writeCSV('processed_label.csv', process_label)