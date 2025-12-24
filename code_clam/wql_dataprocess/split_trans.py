import numpy as np
import pandas as pd

csv_path = '/home1/qiuliwang/Code/CLAM_wql_version/splits/task_1_tumor_vs_normal_75/splits_0.csv'

slide_data = pd.read_csv(csv_path)
train_id = slide_data['train']
# print(train_id)

label_path = '/home1/qiuliwang/Code/CLAM_wql_version/dataset_csv/normal_v_tumor.csv'
labels = pd.read_csv(label_path)


# def locate(_list):
#     t = 0
#     for one_location in _list:
#         if one_location == True:
#             return t
#         t += 1

'''
case_id,slide_id,label
'''


print('train_id[1]: ', train_id[1])
info = labels[labels['slide_id'] == train_id[1]].index.tolist()
print(labels.loc[info])

res = []
for one_train_id in train_id:
    info = labels[labels['slide_id'] == one_train_id].index.tolist()
    info = labels.loc[info]
    slide_id = info['slide_id'].values[0]
    level = info['label'].values[0]
    res.append([slide_id, level])

df = pd.DataFrame(res)
df.to_csv('test.csv', index = False)