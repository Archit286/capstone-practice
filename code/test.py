import cv2
from time import time
from face import blink
from yawn import yawn

def detect(img):
    blink_status = blink(img)
    print("Blink_Status = " + blink_status)
    yawn_status = yawn(img)
    print("Yawn Status = " + yawn_status)

for i in range(1,4):
    t0 = time()

    img = cv2.imread("./test/" + str(i) + ".jpg")
    detect(img)

    print('time taken:   ')
    print(time() - t0)