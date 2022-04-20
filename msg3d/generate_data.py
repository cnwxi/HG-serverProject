import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import argparse


def to_local(pose, o1):
    dis_x = o1[0]
    dix_y = o1[1]
    for i in range(0, len(pose), 2):
        if pose[i] == 0 or pose[i + 1] == 0:
            pose[i] = 0
            pose[i + 1] = 0
        else:
            pose[i] -= dis_x
            pose[i + 1] -= dix_y
    return pose


num_joint = 18
max_frame = 300
num_person_out = 2
num_person_in = 5


class Feeder(Dataset):
    def __init__(self,
                 data_path,
                 label_path,
                 ignore_empty_sample=True,
                 window_size=-1,
                 num_person_in=5,
                 num_person_out=2):
        self.data_path = data_path
        self.label_path = label_path
        self.window_size = window_size
        self.num_person_in = num_person_in
        self.num_person_out = num_person_out
        self.ignore_empty_sample = ignore_empty_sample

        self.load_data()

    def load_data(self):
        # load file list
        self.sample_name = os.listdir(self.data_path)

        # load label
        label_path = self.label_path
        with open(label_path) as f:
            label_info = json.load(f)

        sample_id = [name.split('.')[0] for name in self.sample_name]
        self.label = np.array([label_info[id]['label_index'] for id in sample_id])
        has_skeleton = np.array([label_info[id]['has_skeleton'] for id in sample_id])

        # ignore the samples which does not has skeleton sequence
        if self.ignore_empty_sample:
            self.sample_name = [s for h, s in zip(has_skeleton, self.sample_name) if h]
            self.label = self.label[has_skeleton]

        # output data shape (N, C, T, V, M)
        self.N = len(self.sample_name)  # sample
        self.C = 3  # channel
        self.T = max_frame  # frame
        self.V = num_joint  # joint
        self.M = self.num_person_out  # person

    def __len__(self):
        return len(self.sample_name)

    def __iter__(self):
        return self

    def __getitem__(self, index):

        # output shape (C, T, V, M)
        # get data
        sample_name = self.sample_name[index]
        sample_path = os.path.join(self.data_path, sample_name)
        with open(sample_path, 'r') as f:
            video_info = json.load(f)

        # fill data_numpy
        data_numpy = np.zeros((self.C, self.T, self.V, self.num_person_in))
        for frame_info in video_info['data']:
            frame_index = frame_info['frame_index']
            for m, skeleton_info in enumerate(frame_info["skeleton"]):
                if m >= self.num_person_in:
                    break

                pose = skeleton_info['pose']
                if m == 0:
                    o1 = pose[2:4]
                    if o1[0] == 0 and o1[1] == 0:
                        continue
                pose = to_local(pose=pose, o1=o1)
                score = skeleton_info['score']
                data_numpy[0, frame_index, :, m] = pose[0::2]
                data_numpy[1, frame_index, :, m] = pose[1::2]
                data_numpy[2, frame_index, :, m] = score

        #
        # data_numpy[0:2] = data_numpy[0:2] - 0.5
        data_numpy[1:2] = -data_numpy[1:2]
        data_numpy[0][data_numpy[2] == 0] = 0
        data_numpy[1][data_numpy[2] == 0] = 0

        # get & check label index
        label = video_info['label_index']
        assert (self.label[index] == label)

        # sort by score
        sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
        for t, s in enumerate(sort_index):
            data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2,
                                                                       0))
        data_numpy = data_numpy[:, :, :, 0:self.num_person_out]

        return data_numpy, label


def gendata(data_path, label_path,
            data_out_path, label_out_path,
            num_person_in=num_person_in,  # observe the first 5 persons
            num_person_out=num_person_out,  # then choose 2 persons with the highest score
            max_frame=max_frame):
    feeder = Feeder(
        data_path=data_path,
        label_path=label_path,
        num_person_in=num_person_in,
        num_person_out=num_person_out,
        window_size=max_frame)
    sample_name = feeder.sample_name
    sample_label = []
    fp = np.zeros((len(sample_name), 3, max_frame, num_joint, num_person_out), dtype=np.float32)

    for i, s in enumerate(tqdm(sample_name)):
        data, label = feeder[i]
        fp[i, :, 0:data.shape[1], :, :] = data
        sample_label.append(label)


    with open(label_out_path, 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    np.save(data_out_path, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Kinetics-skeleton Data Converter.')
    parser.add_argument(
        '--data_path', default='../data')
    parser.add_argument(
        '--out_folder', default='../data')
    arg = parser.parse_args()

    part = ['train', 'val']
    for p in part:
        print(p)
        if not os.path.exists(arg.out_folder):
            os.makedirs(arg.out_folder)
        data_path = '{}/{}'.format(arg.data_path, p)
        label_path = '{}/{}_label.json'.format(arg.data_path, p)
        data_out_path = '{}/gen/local_{}_data_joint.npy'.format(arg.out_folder, p)
        label_out_path = '{}/gen/local_{}_label.pkl'.format(arg.out_folder, p)
        gendata(data_path, label_path, data_out_path, label_out_path)
