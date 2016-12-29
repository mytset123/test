# -*- coding: utf-8 -*-
import os

import numpy as np
import cv2
import random

import ImageGenerator as ig

IMAGE_SIZE_X = 255
IMAGE_SIZE_Y = 255

def resize_with_pad(image, height=IMAGE_SIZE_Y, width=IMAGE_SIZE_X):
    
    def get_padding_size(image):
        h, w = image.shape
        longest_edge = max(h, w)
        top, bottom, left, right = (0, 0, 0, 0)
        if h < longest_edge:
            dh = longest_edge - h
            top = dh // 2
            bottom = dh - top
        elif w < longest_edge:
            dw = longest_edge - w 
            left = dw // 2 
            right = dw - left 
        else:
            pass
        return top, bottom, left, right

    top, bottom, left, right = get_padding_size(image)
    WHITE = [255, 255, 255]
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value=WHITE)

    resized_image = cv2.resize(constant, (height, width))

    return resized_image

images = []
labels = []
def traverse_dir(path):
    for file_or_dir in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file_or_dir))
        print(abs_path)
        if os.path.isdir(abs_path):  # dir
            traverse_dir(abs_path)
        else:                        # file
            if file_or_dir.endswith('.jpg') or file_or_dir.endswith('.png'):
                for image in read_image_DL(abs_path):
                    images.append(image)
                    labels.append(path)

    return images, labels

def read_image(file_path):
    image = cv2.imread(file_path)

    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, image_gray = cv2.threshold(image_gray, 200, 255, cv2.THRESH_BINARY)
    image_gray = resize_with_pad(image_gray, IMAGE_SIZE_X, IMAGE_SIZE_Y)

    return image_gray

def read_image_DL(file_path):
    image = read_image(file_path)

    images = []
    images.append(image)
    images.append(ig.addGaussianNoise(image))

    for i in range(1, 15):
        ero = ig.erosion(image, np.ones(i))
        delit = ig.delition(image, np.ones(i))
        ero_n = ig.addGaussianNoise(ero)
        delit_n = ig.addGaussianNoise(delit)

        images.append(ero)
        images.append(delit)
        images.append(ero_n)
        images.append(delit_n)

    for i, img in enumerate(images):
        images[i] = ig.invertImage(img)

    return images

def extract_data(path):
    images, labels = traverse_dir(path)

    images = np.array(images)
    labels = np.array([getLabelFromPath(label) for label in labels])

    for img, lab in zip(images, labels):
        cv2.imwrite("./tmp/" + str(lab) + "_" + str(random.randint(1,99999)) + ".jpg", img)

    return images, labels

def getLabelFromPath(path):
    if "DL_ROOT0" in path:
        return 0
    elif "DL_ROOT1" in path:
        return 1
    elif "DL_ROOT2" in path:
        return 2
    elif "DL_ROOT3" in path:
        return 3
    elif "DL_ROOT4" in path:
        return 4
    elif "DL_ROOT5" in path:
        return 5
    elif "DL_ROOT6" in path:
        return 6
    elif "DL_ROOT7" in path:
        return 7
    elif "DL_ROOT8" in path:
        return 8
    elif "DL_ROOT9" in path:
        return 9
    return -1

