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

camelyon_data = csvTools.readCSV('/home1/qiuliwang/Data/CAMELYON17/training/stage_labels.csv')
print('number of lines: ', len(camelyon_data))

def split_label(temp):
    # print(temp[0][11])
    # return 0

    patient_id = temp[0][:11]
    slide_id = patient_id + '_' + temp[0][12:len(temp[0]) - 4]
    
    if temp[1] == 'itc':
        label = 'subtype_1'
    elif temp[1] == 'macro':
        label = 'subtype_2'
    else:
        label = 'subtype_3'
    return [patient_id, slide_id, label]

process_label = []
for oneline in camelyon_data[1:]:
    temp = oneline
    filename = temp[0]
    if '.zip' not in filename:
        if temp[1] != 'negative':
            temp_label = split_label(temp)
            process_label.append(temp_label)

csvTools.writeCSV('processed_label.csv', process_label)