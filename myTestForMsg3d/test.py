# 读取文件
import json
import os
import pickle
import random

import numpy as np
import torch

from model.msg3d import Model

num_joint = 18
max_frame = 300
num_person_out = 2
num_person_in = 5

myTestNum = 4

dataPath = '../data/my_val'
labelPath = '../data/my_val_label.json'
saveJoint = '../data/testJoint.npy'
saveBone = '../data/testBone.npy'
saveLabel = '../data/testLabel.pkl'
modelJointPath = '../msg3dModels/kinetics-joint.pt'
modelBonePath = '../msg3dModels/kinetics-bone.pt'

# 读取label
with open(labelPath) as f:
    label_info = json.load(f)
    f.close()
# 读取骨架json文件列表
fileList = os.listdir(dataPath)
fileList.sort(key=lambda x: str(x[:-5]))
fileList = fileList[:myTestNum]
print(fileList)
sampleId = [name.split('.')[0] for name in fileList]

# 获取对应标签
labelList = np.array([label_info[id]['label_index'] for id in sampleId])
# 是否有骨架数据的标识
has_skeleton = np.array([label_info[id]['has_skeleton'] for id in sampleId])
# 忽略没有骨架数据的json
fileList = [s for h, s in zip(has_skeleton, fileList) if h]
labelList = labelList[has_skeleton]
with open(saveLabel, 'wb') as f:
    pickle.dump((sampleId, list(labelList)), f)
N = len(fileList)
C = 3
T = max_frame
V = num_joint
M = num_person_out


def getItem(index):
    sampleName = fileList[index]
    samplePath = os.path.join(dataPath, sampleName)
    with open(samplePath, 'r') as f:
        videoInfo = json.load(f)
    dataNumpy = np.zeros((C, T, V, num_person_in))
    for frameInfo in videoInfo['data']:
        frameIndex = frameInfo['frame_index']
        for m, skeletonInfo in enumerate(frameInfo['skeleton']):
            if m > num_person_in:
                break
            pose = skeletonInfo['pose']
            score = skeletonInfo['score']
            dataNumpy[0, frameIndex, :, m] = pose[0::2]
            dataNumpy[1, frameIndex, :, m] = pose[1::2]
            dataNumpy[2, frameIndex, :, m] = score
    dataNumpy[0:2] = dataNumpy[0:2] - 0.5
    dataNumpy[1:2] = -dataNumpy[1:2]
    dataNumpy[0][dataNumpy[2] == 0] = 0
    dataNumpy[1][dataNumpy[2] == 0] = 0

    label = videoInfo['label_index']
    assert (labelList[index] == label)
    # 选出分数最高的两人
    sortIndex = (-dataNumpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
    for t, s in enumerate(sortIndex):
        dataNumpy[:, t, :, :] = dataNumpy[:, t, :, s].transpose((1, 2, 0))
    dataNumpy = dataNumpy[:, :, :, 0:num_person_out]
    return dataNumpy, label


fp = np.zeros((len(fileList), C, T, V, M), dtype=np.float32)
# 遍历json文件,得到关节信息,存储在fp
for i, s in enumerate(fileList):
    data, label = getItem(i)
    fp[i, :, 0:data.shape[1], :, :] = data
np.save(saveJoint, fp)
# 得到骨骼信息,存储在fp_sp
bone_pairs = (
    (0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6), (8, 2), (9, 8), (10, 9),
    (11, 5), (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17, 15)
)
N, C, T, V, M = fp.shape
print(fp.shape)
fp_sp = np.zeros((N, C, T, V, M))
fp_sp[:, :C, :, :, :] = fp
for v1, v2 in bone_pairs:
    fp_sp[:, :, :, v1, :] = fp[:, :, :, v1, :] - fp[:, :, :, v2, :]
np.save(saveBone, fp_sp)

seed = random.randrange(200)

# 预测


num_class = 400
num_point = 18
num_person = 2
num_gcn_scales = 8
num_g3d_scales = 8
graph = 'graph.kinetics.AdjMatrixGraph'
device = 0

# weights = modelJointPath
# model = Model(num_class, num_point, num_person, num_gcn_scales, num_g3d_scales, graph).cuda(device)
# model.load_state_dict(torch.load(weights))
# model.eval()
# torch.no_grad()
#
# for i in fp:
#     data = torch.tensor(np.array([i]))
#     data = data.float().cuda(device)
#     output = model(data)
#     if isinstance(output, tuple):
#         output, _ = output
#     _, predict_label = torch.max(output.data, 1)
#     # print(output)
#     print(predict_label.tolist()[0])
#
# weights = modelBonePath
# model = Model(num_class, num_point, num_person, num_gcn_scales, num_g3d_scales, graph).cuda(device)
# model.load_state_dict(torch.load(weights))
# model.eval()
# torch.no_grad()
#
# for i in fp_sp:
#     data = torch.tensor(np.array([i]))
#     data = data.float().cuda(device)
#     output = model(data)
#     output = output[0]
#     _, predict_label = torch.topk(output.data, 5, 1)
#     # print(output)
#     print(predict_label.tolist()[0])


jointModel = Model(num_class, num_point, num_person, num_gcn_scales, num_g3d_scales, graph).cuda(device)
boneModel = Model(num_class, num_point, num_person, num_gcn_scales, num_g3d_scales, graph).cuda(device)
jointModel.load_state_dict(torch.load(modelJointPath))
boneModel.load_state_dict(torch.load(modelBonePath))
jointModel.eval()
boneModel.eval()
torch.no_grad()
for index in range(len(fp)):
    # print(i.shape)
    jointData = torch.tensor(np.array([fp[index]]))
    jointData = jointData.float().cuda(device)
    jointOutput = jointModel(jointData)
    if isinstance(jointOutput, tuple):
        jointOutput, _ = jointOutput
    _, predict_label = torch.max(jointOutput.data, 1)
    print(predict_label.tolist())

    boneData = torch.tensor(np.array([fp_sp[index]]))
    boneData = boneData.float().cuda(device)
    boneOutput = boneModel(boneData)
    if isinstance(boneOutput, tuple):
        boneOutput, _ = boneOutput
    _, predict_label = torch.max(boneOutput.data, 1)
    print(predict_label.tolist())

    output = jointOutput.data + boneOutput.data
    # print(output)
    _, predict_label = torch.topk(output, 5, 1)
    print(predict_label.tolist())

"""
        ture_label
        
        "label_index": 166

        "label_index": 129

        "label_index": 244

        "label_index": 314
"""
