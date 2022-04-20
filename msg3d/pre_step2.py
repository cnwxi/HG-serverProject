"""
从原数据集中取出部分组成数据集
2、修改标签索引
"""

import json
import os
from tqdm import tqdm

need = [168, 122, 316, 150, 305, 260, 346, 100, 330, 67]

part = ['train', 'val']

for i in part:
    print(i)
    data_file_path = '../data/{}/'.format(i)
    file_list = os.listdir(data_file_path)
    for file_name in tqdm(file_list):
        file_path = data_file_path + file_name
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            f.close()
        label_index = data['label_index']
        if label_index in need:
            index = need.index(label_index)
            data['label_index'] = index
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
                f.close()

    label_json_path = '../data/{}_label.json'.format(i)
    with open(label_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        f.close()
    for i in tqdm(data):
        data[i]['label_index'] = need.index(data[i]['label_index'])
    with open(label_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)
        f.close()
