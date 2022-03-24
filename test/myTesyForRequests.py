import cv2
import imageio
import datetime
import base64
import numpy as np
import json

import requests


def imageToBase64(image):
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.imencode('.jpg', image)[1]
    base64Code = str(base64.b64encode(image))[2:-1]
    return base64Code


def base64ToImage(base64Code):
    img_data = base64.b64decode(base64Code)
    img_array = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_array,cv2.IMREAD_COLOR)
    return img


cap = cv2.VideoCapture('../data/video.mp4')

img = []
base64List = []
while True:
    success, frame = cap.read()
    if success is False:
        print('ERROR')
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    base = imageToBase64(frame)
    base64List.append(base)
    frame = base64ToImage(base)
    # cv2.namedWindow("a", 0)
    # cv2.resizeWindow('a', 640, 360)
    # cv2.imshow('a', frame)
    # cv2.waitKey(1)
    img.append(frame)
    if len(img) == 3:
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        name = '../log/' + str(nowTime) + '.gif'
        print(name)
        imageio.mimsave(name, img, "gif", duration=0.1)
        break
data = {
    'data': []
}
# cv2.destroyAllWindows()
cap.release()
for i in base64List[:2]:
    data['data'].append(i)

url = 'http://127.0.0.1:8000/post_case'
r = requests.post(url=url, data=json.dumps(data))
# print(r.json())
img.clear()
for i in r.json()['data']:
    frame = base64ToImage(i)
    img.append(frame)

nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
name = '../log/ret' + str(nowTime) + '.gif'
print(name)
imageio.mimsave(name, img, "gif", duration=0.1)
# cv2.destroyAllWindows()
