import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import random
import cv2
import tensorflow as tf
from time import time

labels = ["Closed", "Open"]
model = tf.keras.models.load_model("./input/my_model.h5")

def detect(filepath):
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eyeCascade.detectMultiScale(gray, 1.1, 4)
    if len(eyes) == 0:
        return "eyes not detected"
    else:
        for x, y, w, h in eyes:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            eyess = eyeCascade.detectMultiScale(roi_gray)
            if len(eyess) == 0:
                return "eyes not detected"
            else:
                output = 0
                for ex, ey, ew, eh in eyess:
                    eyes_roi = roi_color[ey:ey + eh, ex:ex + ew]
                    final_img = cv2.resize(eyes_roi, (224, 224))
                    final_img = np.expand_dims(final_img, axis=0)
                    final_img = final_img / 255.0
                    prediction = model.predict(final_img)
                    output = output + prediction.astype(int)[0][0]
                if output > 0:
                    return labels[1]
                else:
                    return labels[0]

def test():
    direc = "./test"
    avg_time = 0
    for image in os.listdir(direc):
        path = os.path.join(direc, image)
        t0 = time()
        output = detect(path)
        print(image + "\t" + output)
        avg_time = avg_time + (time() - t0)
    print("\nAverage Time is " + str(avg_time / 25))

test()