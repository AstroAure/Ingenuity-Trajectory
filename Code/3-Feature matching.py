import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import numpy.random as rd

BLUE = (255,0,0)
RED = (0,0,255)
GREEN = (0,255,0)
PINK = (255,0,255)
YELLOW = (0,255,255)
CYAN = (255,255,0)


def draw_rectangle(img, rect, color):
    cv.rectangle(img, rect[0], rect[1], color, 1)

def draw_arrow(img, rect1, rect2, color):
    c1 = ((rect1[0][0] + rect1[1][0])//2, (rect1[0][1] + rect1[1][1])//2)
    c2 = ((rect2[0][0] + rect2[1][0])//2, (rect2[0][1] + rect2[1][1])//2)
    cv.arrowedLine(img, c1, c2, color, tipLength=0.2)


def find_template(img, roi, template):

    h, w, c = template.shape

    ROI = img[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]

    res = cv.matchTemplate(ROI, template, cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    top_left = (max_loc[0] + roi[0][0], max_loc[1] + roi[0][1])
    bottom_right = (top_left[0] + w, top_left[1] + h)

    found = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    return (top_left, bottom_right), found


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

def feature_matching(img1, img2, roi1, roi2, target_size, num_targets):

    ingenuity1 = find_template(img1, roi1, ingenuity_template)
    ing_pos1 = ingenuity1[0]

    TARGETS_img1 = gen_targets(img1, roi1, target_size, num_targets, ing_pos1)

    srch_img = img2.copy()
    ingenuity2 = find_template(img2, roi1, ingenuity_template)
    ing_pos2 = ingenuity2[0]
    """for j in range(3):
        srch_img[ing_pos2[0][1]:ing_pos2[1][1], ing_pos2[0][0]:ing_pos2[1][0], j] = np.zeros(srch_img[ing_pos2[0][1]:ing_pos2[1][1], ing_pos2[0][0]:ing_pos2[1][0]].shape[:2])"""

    FOUND_img2 = []
    for target in TARGETS_img1:
        found = find_template(srch_img, roi2, target[1])
        FOUND_img2.append(found)

    return TARGETS_img1, FOUND_img2, ing_pos1, ing_pos2


images_filename = '0-Images\\Flight 9\\'
output_filename = '3-Feature matching\\'

filenames = os.listdir(images_filename)
n = len(filenames)

begin = 13
end = 165
num_targets = 10
target_size = [20, 20]

ingenuity_template = cv.imread("0-Templates\\Template Ingenuity 10m.png")

for i in range(begin, end):

    name1 = images_filename + str(i) + ".png"
    name2 = images_filename + str(i+1) + ".png"

    img1 = cv.imread(name1)
    img2 = cv.imread(name2)

    H, W, C = img1.shape
    roi1 = [(W//3, H//8), (2*W//3, 7*H//8)]
    roi2 = [(W//8, 0), (7*W//8, H-1)]

    TARGETS_img1, FOUND_img2, ing_pos1, ing_pos2 = feature_matching(img1, img2, roi1, roi2, target_size, num_targets)

    draw_rectangle(img2, roi1, GREEN)
    draw_rectangle(img2, roi2, CYAN)
    draw_rectangle(img2, ing_pos2, RED)

    for j in range(num_targets):
        draw_rectangle(img2, TARGETS_img1[j][0], YELLOW)
        draw_rectangle(img2, FOUND_img2[j][0], PINK)
        draw_arrow(img2, TARGETS_img1[j][0], FOUND_img2[j][0], PINK)

    cv.imwrite(output_filename + str(i) + ".png", img2)

    print(i)

###One shot
images_filename = '0-Images\\Flight 9\\'

i = 94
num_targets = 5
target_size = [20, 20]

ingenuity_template = cv.imread("0-Templates\\Template Ingenuity 10m.png")

name1 = images_filename + str(i) + ".png"
name2 = images_filename + str(i+1) + ".png"

img1 = cv.imread(name1)
img2 = cv.imread(name2)

H, W, C = img1.shape
roi1 = [(W//3, H//8), (2*W//3, 7*H//8)]
roi2 = [(W//8, 0), (7*W//8, H-1)]

TARGETS_img1, FOUND_img2, ing_pos1, ing_pos2 = feature_matching(img1, img2, roi1, roi2, target_size, num_targets)

for j in range(num_targets):
    plt.subplot(2, num_targets+1, 2+j), plt.imshow(TARGETS_img1[j][1]), plt.title("Target " + str(j+1))
    plt.subplot(2, num_targets+1, num_targets+3+j), plt.imshow(FOUND_img2[j][1]), plt.title("Target " + str(j+1) + " found")

draw_rectangle(img1, roi1, GREEN)
draw_rectangle(img2, roi2, CYAN)
draw_rectangle(img2, ing_pos2, RED)

for j in range(num_targets):
    draw_rectangle(img1, TARGETS_img1[j][0], YELLOW)
    draw_rectangle(img2, FOUND_img2[j][0], PINK)

plt.subplot(2, num_targets+1, 1), plt.imshow(img1), plt.title("Image 1")
plt.subplot(2, num_targets+1, num_targets+2), plt.imshow(img2), plt.title("Image 2")

plt.show()
