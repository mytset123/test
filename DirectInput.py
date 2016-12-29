# coding: UTF-8
import numpy as np
import cv2
import os
from train import Model
from Input import read_image

import ImageGenerator as ig

model = Model()
model.load()

PATH = './input'

def OCR(img_path):  
    img = read_image(img_path)
    img = ig.invertImage(img)
    result = model.predict(img)
    print(result)

if __name__ == '__main__':
    for file in os.listdir(PATH):
        abs_path = os.path.abspath(os.path.join(PATH, file))
        print abs_path
        if os.path.isfile(abs_path):
            if abs_path.endswith("jpg") or abs_path.endswith("png"):
                OCR(abs_path)