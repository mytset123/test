# -*- coding: utf-8 -*-
import os

import numpy as np
import cv2
import random

IMAGE_SIZE_X = 25
IMAGE_SIZE_Y = 64

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

def read_image_DL(file_path):
    image = read_image(file_path)

    images = []
    images.append(invertImage(image))
    #images.append(addGaussianNoise(image))

    #for x in range(10):
    #    for y in range(5):
    #        for a in range(10):
    #            images.append(warpImage(image, np.pi/(50-a), x, y))
    #            images.append(warpImage(image, 2*np.pi - (np.pi/(50-a)), -x, -y))

    for i in range(1, 3):
        ero = erosion(image, np.ones(i))
        delit = delition(image, np.ones(i))

        images.append(invertImage(ero))
        images.append(invertImage(delit))
        
    return images

def invertImage(image):
    return 255 - image

def erosion(image, kernel):
    result = cv2.erode(image, kernel, iterations=1)

    filename = "./tmp/" + "erosion_" + str(kernel[0]) + "_" + str(random.randint(0,99999)) + ".png"
    cv2.imwrite(filename, result)
    return result

def delition(image, kernel):
    result = cv2.dilate(image, kernel, iterations=1)

    filename = "./tmp/" + "delate_" + str(kernel[0]) + "_" + str(random.randint(0,99999)) + ".png"
    cv2.imwrite(filename, result)
    return result

#def centroid(image):
#    mu = cv2.moments(image)
#    c = (mu["mu10"]/mu["mu00"], mu["mu01"]/mu["mu00"])

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

def read_image(file_path):
    image = cv2.imread(file_path)

    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, image_gray = cv2.threshold(image_gray, 200, 255, cv2.THRESH_BINARY)
    image_gray = resize_with_pad(image_gray, IMAGE_SIZE_X, IMAGE_SIZE_Y)

    return image_gray


def extract_data(path):
    images, labels = traverse_dir(path)

    images = np.array(images)
    labels = np.array([getLabelFromPath(label) for label in labels])

    return images, labels

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