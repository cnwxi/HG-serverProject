import os
import argparse
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

bone_pairs = {
    'kinetics': (
        (0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6), (8, 2), (9, 8), (10, 9),
        (11, 5), (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17, 15)
    )
}

benchmarks = {
    'kinetics': ('kinetics',)
}

parts = {'train', 'val'}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bone data generation for NTU60/NTU120/Kinetics')
    parser.add_argument('--dataset', default='kinetics')
    args = parser.parse_args()

    for benchmark in benchmarks[args.dataset]:
        for part in parts:
            print(benchmark, part)
            try:
                data = np.load('../data/{}_data_joint.npy'.format(part), mmap_mode='r')
                N, C, T, V, M = data.shape
                fp_sp = open_memmap(
                    '../data/{}_data_bone.npy'.format(part),
                    dtype='float32',
                    mode='w+',
                    shape=(N, 3, T, V, M))
                fp_sp[:, :C, :, :, :] = data
                for v1, v2 in tqdm(bone_pairs[benchmark]):
                    fp_sp[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]
            except Exception as e:
                print(f'Run into error: {e}')
                print(f'Skipping ({benchmark} {part})')
