# -*- coding: utf-8 -*-
import numpy as np
import cv2
import random

def invertImage(image):
    return 255 - image

def erosion(image, kernel):
    result = cv2.erode(image, kernel, iterations=1)
    return result

def delition(image, kernel):
    result = cv2.dilate(image, kernel, iterations=1)
    return result
    
# ガウシアンノイズ
def addGaussianNoise(src):
    row,col = src.shape
    mean = 0
    var = 0.1
    sigma = 15
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = src + gauss

    return noisy


def warpImage(image, rad, x, y):
    size = tuple(np.array([image.shape[1], image.shape[0]]))

    matrix = [
        [np.cos(rad), -1 * np.sin(rad), x],
        [np.sin(rad), np.cos(rad), y]
        ]
    affine_matrix = np.float32(matrix)

    warp_img = cv2.warpAffine(image, affine_matrix, size, flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255, 255))
    filename = "./tmp/" + str(rad) + "_" + str(x) + "_" + str(y) + "_" + str(random.randint(0,99999)) + ".png"
    cv2.imwrite(filename, warp_img)
    return warp_img
