import csvTools

autogen_path = '/home1/qiuliwang/Code/CLAM_wql_version/DataResult/process_list_autogen.csv'
autogen_info = csvTools.readCSV(autogen_path)

split0 = 'split_0_slideids.csv'
split1 = 'split_1_slideids.csv'
split2 = 'split_2_slideids.csv'
split_info = csvTools.readCSV(split0) # + csvTools.readCSV(split1) + csvTools.readCSV(split2)

selected_split_info = []
for one_info in split_info:
    if one_info not in selected_split_info:
        selected_split_info.append(one_info)

print(len(selected_split_info))

selected_autogen = []
for one_auto in autogen_info:
    for one_selected_split in selected_split_info:
        # print(one_selected_split)
        if one_selected_split[0] in one_auto[0]:
            # print(True)
            selected_autogen.append(one_auto)
            break

print(len(selected_autogen))
print(selected_autogen[0])
csvTools.writeCSV('test.csv', selected_autogen)