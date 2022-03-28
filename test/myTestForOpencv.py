import cv2

url = 'http://devimages.apple.com.edgekey.net/streaming/examples/bipbop_4x3/gear2/prog_index.m3u8'
cap = cv2.VideoCapture(url)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    cv2.namedWindow('a', 0)
    cv2.resizeWindow('a', 640, 360)
    cv2.imshow('a', frame)
    if cv2.waitKey(1) in [ord('q'), 27]:
        break
