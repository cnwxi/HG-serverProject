"""
从原数据集中取出部分组成数据集
1、取出所需数据
"""
import json
import os
import shutil
from tqdm import tqdm

need = [168, 122, 316, 150, 305, 260, 346, 100, 330, 67]

data_path = './data/kinetics_raw/kinetics_val/'
label_path = './data/kinetics_raw/kinetics_val_label.json'
save_dir = f'./mydata/val'
new_label_josn = dict()
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
with open(label_path, 'r') as f:
    label_info = json.load(f)
    f.close()
for file_name in tqdm(label_info):
    file_path = data_path + file_name + '.json'
    if label_info[file_name]['has_skeleton']:
        label_index = label_info[file_name]['label_index']
        if label_index in need:
            shutil.copy2(file_path, save_dir)
            new_file_path = save_dir + '/' + file_name + '.json'
            # new_label_josn[file_name] = label_info[file_name]
            print(need.index(label_index))

# with open('./mydata/val_label.json', 'w') as f:
#     json.dump(new_label_josn, f)
#     f.close()
