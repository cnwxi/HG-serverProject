import json
import time

import cv2
from openpose import pyopenpose as op
from model.msg3d import Model
import numpy as np
import torch


# 图像预处理
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape
    x_new = 720  # x height y width
    y_new = 1280
    # 判断图片的长宽比率
    if width / height >= y_new / x_new:
        img_new = cv2.resize(img, (y_new, int(height * y_new / width)))
    else:
        img_new = cv2.resize(img, (int(width * x_new / height), x_new))
    return img_new


def get_skeleton(img):
    datum = op.Datum()
    datum.cvInputData = img
    wrapper.emplaceAndPop(op.VectorDatum([datum]))
    cv2.namedWindow("OpenPose", 0);
    cv2.resizeWindow("OpenPose", 640, 360);
    cv2.imshow("OpenPose", cv2.cvtColor(datum.cvOutputData, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)
    # print(datum.poseKeypoints)
    return datum.poseKeypoints


def getItem(data):
    dataNumpy = np.zeros((3, 300, 18, 5))
    for frameInfo in data['data']:
        frameIndex = frameInfo['frame_index']
        for m, skeletonInfo in enumerate(frameInfo['skeleton']):
            if m > 5:
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
    sortIndex = (-dataNumpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
    for t, s in enumerate(sortIndex):
        dataNumpy[:, t, :, :] = dataNumpy[:, t, :, s].transpose((1, 2, 0))
    dataNumpy = dataNumpy[:, :, :, 0:2]
    return dataNumpy


def recognize(skeleton_data):
    modelJointPath = './msg3dModels/kinetics-joint.pt'
    modelBonePath = './msg3dModels/kinetics-bone.pt'
    data = getItem(skeleton_data)
    jointData = np.zeros((1, 3, 300, 18, 2))
    # 得到关节信息
    jointData[0, :, 0:data.shape[1], :, :] = data
    bone_pairs = (
        (0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6), (8, 2), (9, 8), (10, 9),
        (11, 5), (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17, 15)
    )
    # 得到骨骼信息
    boneData = np.zeros((1, 3, 300, 18, 2))
    jointData[:, :3, :, :, :] = jointData
    for v1, v2 in bone_pairs:
        boneData[:, :, :, v1, :] = jointData[:, :, :, v1, :] - jointData[:, :, :, v2, :]
    num_class = 400
    num_point = 18
    num_person = 2
    num_gcn_scales = 8
    num_g3d_scales = 8
    graph = 'graph.kinetics.AdjMatrixGraph'
    device = 0
    torch.no_grad()

    jointModel = Model(num_class, num_point, num_person, num_gcn_scales, num_g3d_scales, graph).cuda(device)
    jointModel.load_state_dict(torch.load(modelJointPath))
    jointModel.eval()
    jointData = torch.tensor(np.array(jointData))
    jointData = jointData.float().cuda(device)
    jointOutput = jointModel(jointData)
    if isinstance(jointOutput, tuple):
        jointOutput, _ = jointOutput
    _, predict_lable = torch.topk(jointOutput.data, 5, 1)
    print(f'关节预测：{predict_lable.tolist()}')

    '''
    性能限制只能跑一个
    '''

    # boneModel = Model(num_class, num_point, num_person, num_gcn_scales, num_g3d_scales, graph).cuda(device)
    # boneModel.load_state_dict(torch.load(modelBonePath))
    # boneModel.eval()
    # boneData = torch.tensor(np.array(boneData))
    # boneData = boneData.float().cuda(device)
    # boneOutput = boneModel(boneData)
    # if isinstance(boneOutput, tuple):
    #     jointOutput, _ = boneOutput
    # _, predict_lable = torch.topk(boneOutput.data, 5, 1)
    # print(f'骨骼预测：{predict_lable.tolist()}')

    # output = jointOutput.data + boneOutput.data
    # _, predict_label = torch.topk(output, 5, 1)
    # print(f'双流预测：{predict_label.tolist()}')
    return


def push(img, msg):
    return


def get_info(narray, x, y):
    skeleton_data = []
    for i in narray:
        pose = i[:, :2]
        pose = pose / [x, y]
        pose = [round(i, 3) for item in pose for i in item]
        score = i[:, 2].tolist()
        score = [round(i, 3) for i in score]
        skeleton_data.append({'pose': pose, 'score': score})
    return skeleton_data


def print_log(msg):
    localtime = time.asctime(time.localtime(time.time()))
    msg = f'[{localtime}]{msg}'
    print(msg)


if __name__ == '__main__':
    # 从拉流中读取帧(暂时以本地视频替代
    cap = cv2.VideoCapture('./data/video.mp4')
    # 从摄像头读取
    # cap = cv2.VideoCapture(0)

    # 启动openpose
    params = dict()
    params['model_folder'] = './OpenPoseModels/'
    params['model_pose'] = 'COCO'
    params['net_resolution'] = '320x176'
    wrapper = op.WrapperPython()
    wrapper.configure(params)
    wrapper.start()

    count = 0
    has_skeleton = 0
    step = 1
    # 存储骨架信息
    data = {'data': []}
    # 关键帧
    shot = []
    while cap.isOpened():
        success, frame = cap.read()
        # 读取成功
        if success:
            count += 1
            frame = preprocess(frame)
            if count % step == 0:
                ret = get_skeleton(frame)
                x, y, z = frame.shape

                frame_index = int(count / step) - 1
                print_log(f'第{frame_index}帧')

                if ret is not None:  # has_skeleton
                    skeleton_data = get_info(narray=ret, x=x, y=y)
                    has_skeleton += 1
                    # 取中间帧作为记录帧
                else:
                    skeleton_data = []
                data['data'].append({'frame_index': frame_index, 'skeleton': skeleton_data})
                if (count / step) % 30 == 0:
                    shot.append(frame)
                # 测试
                # if (count / step) % 30 == 0:
                #     with open('data.json', 'w') as f:
                #         json.dump(data, f)
                #     break
            # 每处理300帧骨骼信息进行一次动作识别
            if count / step == 30:
                if has_skeleton > 25:
                    pre = recognize(data)
                    msg = f''
                else:  # 空白帧过多，跳过这段视频
                    msg = f'无事发生'
                push(shot, msg)
                count = 0
                has_skeleton = 0
                shot.clear()
                data['data'].clear()
        if cv2.waitKey(1) == ord('1'):
            cv2.destroyAllWindows()
            break
    cap.release()
