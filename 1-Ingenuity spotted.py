import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os

template = cv.imread("0-Templates\\Template Ingenuity 10m.png")
h, w, c = template.shape

images_filename = '0-Images\\Flight 9\\'
output_filename = '1-Ingenuity Spotted\\'

filenames = os.listdir(images_filename)
n = len(filenames)

for i in range(n):
    name = images_filename + str(i) + ".png"

    img = cv.imread(name)
    H, W, C = img.shape

    roi = [(W//4, H//3), (3*W//4, H-1)]
    ROI = img[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]

    cv.rectangle(img,roi[0], roi[1], (0,255,0), 1)

    res = cv.matchTemplate(ROI, template, cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    top_left = (max_loc[0] + roi[0][0], max_loc[1] + roi[0][1])
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(img, top_left, bottom_right, (255,0, 0), 1)

    cv.imwrite(output_filename + str(i) + ".png", img)

    print(i)
