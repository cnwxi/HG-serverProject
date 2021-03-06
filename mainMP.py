import json
import time
import cv2
import numpy as np
import torch
import requests
from mymediapipe.mediapipe import PoseDetector, toOpenPosePoint
from model.msg3d import Model
from utils import imageToBase64


# 图像预处理
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape
    x_new = 720
    y_new = 1280
    # 判断图片的长宽比率
    if width / height >= y_new / x_new:
        img_new = cv2.resize(img, (y_new, int(height * y_new / width)))
    else:
        img_new = cv2.resize(img, (int(width * x_new / height), x_new))
    return img_new


def get_skeleton(img, detector):
    img = detector.find_pose(img, draw=True)
    points = toOpenPosePoint(detector.get_positions())
    cv2.namedWindow("MediaPipe", 0)
    cv2.resizeWindow("MediaPipe", 640, 360)
    cv2.imshow("MediaPipe", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return points


def getItem(data):
    dataNumpy = np.zeros((3, 300, 18, 2))
    for frameInfo in data['data']:
        frameIndex = frameInfo['frame_index']
        for m, skeletonInfo in enumerate(frameInfo['skeleton']):
            pose = skeletonInfo['pose']
            dataNumpy[0, frameIndex, :, m] = pose[0::2]
            dataNumpy[1, frameIndex, :, m] = pose[1::2]
    return dataNumpy


def recognize(skeleton_data):
    modelJointPath = './msg3dModels/myModel_joint.pt'
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
    _, predict_lable = torch.topk(jointOutput.data, 3, 1)
    print_log(f'关节预测：{predict_lable.tolist()[0]}')

    boneModel = Model(num_class, num_point, num_person, num_gcn_scales, num_g3d_scales, graph).cuda(device)
    boneModel.load_state_dict(torch.load(modelBonePath))
    boneModel.eval()
    boneData = torch.tensor(np.array(boneData))
    boneData = boneData.float().cuda(device)
    boneOutput = boneModel(boneData)
    if isinstance(boneOutput, tuple):
        jointOutput, _ = boneOutput
    _, predict_lable = torch.topk(boneOutput.data, 3, 1)
    print_log(f'骨骼预测：{predict_lable.tolist()[0]}')

    output = jointOutput.data + boneOutput.data
    _, predict_label = torch.topk(output, 3, 1)
    print_log(f'双流预测：{predict_label.tolist()[0]}')

    del boneOutput
    del jointOutput
    return predict_lable.tolist()[0]


def push(imgList, label):
    url = 'http://127.0.0.1:8000/post_case'
    data = {
        'data': [],
        'label': label
    }
    for i in imgList:
        data['data'].append(imageToBase64(i))
    try:
        r = requests.post(url=url, data=json.dumps(data))
        if r.ok and r.json().get('success'):
            print_log(f'向Django推送成功 记录帧数{len(imgList)}')
        else:
            print_log('向Django推送失败')
    except EOFError as e:
        print_log(e)


def get_info(narray):
    skeleton_data = []
    skeleton_data.append({'pose': narray})
    return skeleton_data


def print_log(msg):
    localtime = time.asctime(time.localtime(time.time()))
    msg = f'[{localtime}]{msg}'
    print(msg)


def main(video_path):
    # 从拉流中读取帧(暂时以本地视频替代
    cap = cv2.VideoCapture(video_path)
    # 从摄像头读取
    # cap = cv2.VideoCapture(0)

    # MediaPipe
    detector = PoseDetector()
    # 计数
    count = 0
    has_skeleton = 0
    # 跳帧
    step = 1
    # 最大帧数 应小于300
    max_frame = 200
    # 有骨骼信息的帧占最大帧数的比例
    rate = 0.5
    # 存储骨架信息
    data = {'data': []}
    # 关键帧
    shot = []
    # 是否推送

    with open('./label.json', 'r', encoding='utf-8') as f:
        label = json.load(f)
        f.close()

    while cap.isOpened():
        success, frame = cap.read()
        # 读取成功
        if success:
            count += 1
            frame = preprocess(frame)
            if count % step == 0:
                ret = get_skeleton(frame, detector)

                frame_index = int(count / step) - 1
                print_log(f'第{frame_index}帧')

                if len(ret) > 0:  # has_skeleton
                    skeleton_data = get_info(narray=ret)
                    has_skeleton += 1
                else:
                    skeleton_data = []
                data['data'].append({'frame_index': frame_index, 'skeleton': skeleton_data})
                # 取中间帧作为记录帧
                if frame_index % (max_frame / 10) == 0:
                    shot.append(frame)

            # 每处理max_frame帧骨骼信息进行一次动作识别n
            if count / step == max_frame:
                # with open('./log/data.json', 'w') as f:
                #     json.dump(data, f)
                # 占空比小于rate 进行识别
                if has_skeleton * 1.0 / count * step > rate:
                    pre = recognize(data)
                    for i in pre:
                        print(label[str(i)])
                else:  # 空白帧过多，跳过这段视频
                    pre = []
                if False:
                    push(shot, pre)
                count = 0
                has_skeleton = 0
                shot.clear()
                data['data'].clear()
        if cv2.waitKey(1) == ord('1'):
            cv2.destroyAllWindows()
            break
    cap.release()


if __name__ == '__main__':
    # main(
    #     "D:\\Users\\xiang\\Downloads\\Compressed\\tiny-Kinetics-400\\tiny-Kinetics-400\\crying\\_cE99P5nAYk_000034_000044.mp4")
    # main(
    #     "D:\\Users\\xiang\\Downloads\\Compressed\\tiny-Kinetics-400\\tiny-Kinetics-400\\drinking\\_iujb_vthv0_000011_000021.mp4")
    main(
        "D:\\Users\\xiang\\Downloads\\Compressed\\tiny-Kinetics-400\\tiny-Kinetics-400\\tai_chi\\_2zDhdZrwOc_000153_000163.mp4")
