import cv2
from time import time
from face import blink
from yawn import yawn

blink_counter = 0
blink_threshold = 6
yawn_counter = 0
yawn_threshold = 6

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

def detect(img):
    blink_status = blink(img)
    print("Blink_Status = " + blink_status)
    yawn_status = yawn(img)
    print("Yawn Status = " + yawn_status)

for j in range(25):
    t0 = time()

    frame = None
    success, frame = cap.read()

    filename = "./result/" + str(j) + ".jpg"
    imwrite(filename, frame)

    if success:
        detect(frame)
    else:
        print('Error in Camera')  # For debugging purposes
        continue

    print('time taken:   ')
    print(time()-t0)