import pandas as pd 
import csvTools
csv_path = '/home1/qiuliwang/Code/CLAM_wql_version/splits/splits_0.csv'

df = pd.read_csv(csv_path)
train = df['train'].values.tolist()
val = df['val'].values.tolist()
test = df['test'].values.tolist()

print(len(train))
print(len(val))

slide_ids = train + val + test

selected = []
for one_id in slide_ids:
    try:
        if 'patient' in one_id:
            one_id = one_id
            selected.append(one_id)
    except:
        a = 1
print(len(selected))

csvTools.writeCSV_('split_0_slideids.csv', selected)