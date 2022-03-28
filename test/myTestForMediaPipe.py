import time

import cv2
import mediapipe as mp
import numpy as np


class PoseDetector:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(True, False, True, False, 0.8, 0.5)

    def find_pose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # pose.process(imgRGB) 会识别这帧图片中的人体姿势数据，保存到self.results中
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                mp.solutions.drawing_utils.draw_landmarks(img, self.results.pose_landmarks,
                                                          mp.solutions.pose.POSE_CONNECTIONS)
        return img

    def find_positions(self, img):
        """
        获取人体姿势数据
        :param img: 一帧图像
        :param draw: 是否画出人体姿势节点和连接图
        :return: 人体姿势数据列表
        """
        # 人体姿势数据列表，每个成员由3个数字组成：id, x, y
        # id代表人体的某个关节点，x和y代表坐标位置数据
        self.lmslist = []
        if self.results.pose_landmarks:
            print(self.results.pose_landmarks)
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if lm.visibility > 0.6:
                    self.lmslist.append([cx, cy])
                else:
                    self.lmslist.append([0, 0])
                # cx, cy = lm.x, lm.y
                # self.lmslist.append([cx, cy])
        print(self.lmslist)
        return self.lmslist


def toOpenPosePoint(list):
    if len(list) == 0:
        return []
    list = np.array(list)
    Nose = list[0]
    Neck = (list[11] + list[12]) / 2
    RShoulder = list[11]
    RElbow = list[13]
    RWrist = list[15]
    LShoulder = list[12]
    LElbow = list[14]
    LWrist = list[16]
    RHip = list[23]
    RKnee = list[25]
    RAnkle = list[27]
    LHip = list[24]
    LKnee = list[26]
    LAnkle = list[28]
    REye = list[2]
    LEye = list[5]
    REar = list[7]
    LEar = list[8]
    keyPoints = [Nose, Neck, RShoulder, RElbow, RWrist, LShoulder, LElbow, LWrist, RHip, RKnee, RAnkle, LHip, LKnee,
                 LAnkle, REye, LEye, REar, LEar]
    keyPoints = [round(i, 3) for item in keyPoints for i in item]
    return keyPoints


if __name__ == '__main__':
    # 获取摄像头，传入0表示获取系统默认摄像头
    frame = cv2.imread('../data/4.jpg')
    detector = PoseDetector()

    frame = detector.find_pose(frame)
    # print(detector.find_positions())
    points = toOpenPosePoint(detector.find_positions(frame))
    print(f"points {points}")
    points = np.array(points)

    cv2.imshow("MediaPipe", frame)
    cv2.waitKey()

    # 关闭图像窗口
    cv2.destroyAllWindows()
