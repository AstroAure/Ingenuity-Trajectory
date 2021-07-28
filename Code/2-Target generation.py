import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import numpy.random as rd

BLUE = (0,0,255)
RED = (255,0,0)
GREEN = (0,255,0)


def draw(img, rect, color):
    cv.rectangle(img, rect[0], rect[1], color, 1)


def find_template(img, roi, template):

    h, w, c = template.shape

    ROI = img[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]

    res = cv.matchTemplate(ROI, template, cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    top_left = (max_loc[0] + roi[0][0], max_loc[1] + roi[0][1])
    bottom_right = (top_left[0] + w, top_left[1] + h)

    return top_left, bottom_right


def gen_target_random(img, roi, target_size):

    x = rd.randint(roi[1][1]-roi[0][1])
    y = rd.randint(roi[1][0]-roi[0][0])
    target = img[roi[0][1]+y:roi[0][1]+y+target_size[1], roi[0][0]+x:roi[0][0]+x+target_size[0]]

    return target


def gen_targets(img, roi, target_size, num_targets, ingenuity):

    ROI = img[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
    ROIblur = cv.GaussianBlur(ROI, (5,5), 0)
    laplacian = 10*np.uint8(np.absolute(cv.Laplacian(ROIblur,cv.CV_64F)))
    blur = cv.GaussianBlur(laplacian, (5,5), 0)
    blurGRAY = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)

    blurGRAY[max(0,ingenuity[0][1]-roi[0][1]):min(blurGRAY.shape[0],ingenuity[1][1]-roi[0][1]), max(0,ingenuity[0][0]-roi[0][0]):min(blurGRAY.shape[0],ingenuity[1][0]-roi[0][0])] = np.zeros(blurGRAY[max(0,ingenuity[0][1]-roi[0][1]):min(blurGRAY.shape[0],ingenuity[1][1]-roi[0][1]), max(0,ingenuity[0][0]-roi[0][0]):min(blurGRAY.shape[0],ingenuity[1][0]-roi[0][0])].shape[:2]) #Hiding Ingenuity's shadow

    TARGETS = []

    for i in range(num_targets):

        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(blurGRAY)
        top_left = (roi[0][0] + max_loc[0] - target_size[0] // 2, roi[0][1] + max_loc[1] - target_size[1] // 2)
        bottom_right = (roi[0][0] + max_loc[0] + target_size[0] // 2, roi[0][1] + max_loc[1] + target_size[1] // 2)
        target = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        TARGETS.append([(top_left, bottom_right), target])

        blurGRAY[max(0,max_loc[1]-target_size[1]//2):min(blurGRAY.shape[0],max_loc[1]+target_size[1]//2), max(0,max_loc[0]-target_size[0]//2):min(blurGRAY.shape[1],max_loc[0]+target_size[0]//2)] = np.zeros(blurGRAY[max(0,max_loc[1]-target_size[1]//2):min(blurGRAY.shape[0],max_loc[1]+target_size[1]//2), max(0,max_loc[0]-target_size[0]//2):min(blurGRAY.shape[1],max_loc[0]+target_size[0]//2)].shape[:2]) #Hiding target

    return TARGETS


images_filename = '0-Images\\Flight 9\\'
output_filename = '2-Target Generated\\'

filenames = os.listdir(images_filename)
n = len(filenames)

ingenuity_template = cv.imread("0-Templates\\Template Ingenuity 10m.png")
target_size = [20, 20]


for i in range(0, n-2):

    name = images_filename + str(i) + ".png"

    img = cv.imread(name)

    H, W, C = img.shape

    roi = [(W//4, H//3), (3*W//4, H-1)]

    ingenuity_pos = find_template(img, roi, ingenuity_template)

    targets = gen_targets(img, roi, target_size, 5, ingenuity_pos)

    draw(img, roi, GREEN)
    draw(img, ingenuity_pos, BLUE)
    for target in targets:
        draw(img, target[0], RED)

    cv.imwrite(output_filename + str(i) + ".png", img)

    print(i)

###One shot
images_filename = '0-Images\\Flight 9\\'

ingenuity_template = cv.imread("0-Templates\\Template Ingenuity 10m contrasted.png")

i = 94
num_targets = 5 #min = 5
target_size = [30, 30]

name = images_filename + str(i) + ".png"

img = cv.imread(name)

H, W, C = img.shape

roi = [(W//4, H//3), (3*W//4, H-1)]

ingenuity_pos = find_template(img, roi, ingenuity_template)

TARGETS = gen_targets(img, roi, target_size, num_targets, ingenuity_pos)


for j in range(num_targets):
    plt.subplot(2, num_targets, num_targets+1+j), plt.imshow(TARGETS[j][1]), plt.title("Target " + str(j+1))

ROI = img[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
ROIblur = cv.GaussianBlur(ROI, (5,5), 0)
laplacian = 10*np.uint8(np.absolute(cv.Laplacian(ROIblur,cv.CV_64F)))
blur = cv.GaussianBlur(laplacian, (5,5), 0)
blurGRAY = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)

for target in TARGETS:
    draw(img, target[0], RED)

plt.subplot(2, num_targets, 1), plt.imshow(img), plt.title("Raw image")
plt.subplot(2, num_targets, 2), plt.imshow(ROI), plt.title("ROI")
plt.subplot(2, num_targets, 3), plt.imshow(ROIblur), plt.title("ROI blurred")
plt.subplot(2, num_targets, 4), plt.imshow(laplacian), plt.title("Laplacian")
plt.subplot(2, num_targets, 5), plt.imshow(blurGRAY), plt.title("Laplacian blurred grayscale")

plt.show()
