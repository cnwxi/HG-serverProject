import json

from openpose import pyopenpose as op
import cv2
from time import perf_counter

t1 = perf_counter()

try:
    params = dict()
    params['model_folder'] = '../OpenPoseModels/'
    params['model_pose'] = 'COCO'
    params['net_resolution'] = '320x176'
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    imageToProcess = cv2.imread("../data/1.jpg")
    x, y, z = imageToProcess.shape
    imageToProcess = cv2.cvtColor(imageToProcess, cv2.COLOR_BGR2RGB)
    datum = op.Datum()
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    # print(len(datum.poseKeypoints))
    # print(datum.poseKeypoints)
    print(datum.poseKeypoints.shape)
    # print("Body keypoints: \n" + str(datum.poseKeypoints))
    # cv2.imshow("OpenPose 1.7.0", datum.cvOutputData)
    data = {'data': []}
    skeleton_data = []
    for i in datum.poseKeypoints:
        print(i.shape)
        pose = i[:, :2]
        pose = pose / [x, y]
        score = i[:, 2].tolist()
        pose = [round(i, 3) for item in pose for i in item]
        score = [round(i, 3) for i in score]
        print(type(pose))
        print(type(score))
        skeleton_data.append({'pose': pose, 'score': score})
        # break
    data['data'].append({'frame_index': 1, 'skeleton': skeleton_data})
    with open('../log/test.json', 'w') as f:
        json.dump(data, f)
        f.close()

except Exception as e:
    print(e)
t2 = perf_counter()
print('耗时：', t2 - t1)
# .\bin\OpenPoseDemo.exe --video .\examples\media\test.jpg --model_pose COCO --net_resolution 320x176
