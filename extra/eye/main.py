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
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    new_array = cv2.resize(backtorgb, (224, 224))
    X_input = np.array(new_array).reshape(1, 224, 224, 3)
    X_input = X_input / 255.0
    prediction = model.predict(X_input)
    return labels[prediction.astype(int)[0][0]]

def test():
    direc = "./train"
    result = 0
    avg_time = 0
    for category in os.listdir(direc):
        print(category)
        path_link = os.path.join(direc, category)
        images = random.sample(os.listdir(path_link), 100)
        count = 0
        for image in images:
            path = os.path.join(path_link, image)
            t0 = time()
            output = detect(path)
            avg_time = avg_time + (time()-t0)
            if output == category:
                count = count + 1
        accuracy = count / len(images)
        print("Accuracy for " + category + " is " + str(accuracy))
        result = result + accuracy
    print("\nFinal Accuracy is " + str(result/2))
    print("\nAverage Time is " + str(avg_time / 200))

test()