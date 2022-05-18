import cv2
from numpy import random
import numpy as np

def add_noise(img):
    # Getting the dimensions of the image
    if(len(img.shape) == 2):
        row, col = img.shape
    else:
        row, col, ch = img.shape
    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to white
        img[y_coord][x_coord] = 255

    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to black
        img[y_coord][x_coord] = 0

    return img
def gaussian_blur(img, value):
    if(value > 0):
        value = convertValidValue(value)
        img = cv2.GaussianBlur(img, (value, value), 0)
    return img
def box_blur(img, value):
    if (value > 0):
        value = convertValidValue(value)
        img = cv2.boxFilter(img, -1, (value, value))
    return img

def median_blur(img, value):
    if (value > 0):
        value = convertValidValue(value)
        img = cv2.medianBlur(img, value)
    return img
def convertValidValue(value):
    if value % 2 == 0:
        value += 1
    return value
def sobel_filter(img, horizontal, vertical):
    Vkernel = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])

    Hkernel = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])
    if (horizontal == True and vertical == True):
        img1 = cv2.filter2D(img, ddepth=-1, kernel=Vkernel)
        img2 = cv2.filter2D(img, ddepth=-1, kernel=Hkernel)
        return img1 + img2
    elif(horizontal == True):
        return cv2.filter2D(img, ddepth=-1, kernel=Hkernel)
    elif(vertical == True):
        return cv2.filter2D(img, ddepth=-1, kernel=Vkernel)
    return img
def sobel_sharpen(img, isSobelSharpen):
    if(isSobelSharpen == True):
        return img - sobel_filter(img, True, True)
    return img

def laplace_filter(img, isEDLaplace):
    kernel = np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]])

    if(isEDLaplace == True):
        return cv2.filter2D(img, ddepth=-1, kernel=kernel)

    return img

def laplace_sharpen(img, isLaplaceSharpen):
    if(isLaplaceSharpen == True):
        return img - laplace_filter(img, True)

    return img