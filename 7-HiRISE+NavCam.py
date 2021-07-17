import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import numpy.random as rd
import itertools as it
import imageio

BLUE = (255,0,0)
RED = (0,0,255)
GREEN = (0,255,0)
PINK = (255,0,255)
YELLOW = (0,255,255)
CYAN = (255,255,0)
WHITE = (255,255,255)


def create_vector(rect1, rect2):
    c1 = ((rect1[0][0] + rect1[1][0])//2, (rect1[0][1] + rect1[1][1])//2)
    c2 = ((rect2[0][0] + rect2[1][0])//2, (rect2[0][1] + rect2[1][1])//2)
    return np.array([c2[0]-c1[0], c2[1]-c1[1]]), c1

def draw_rectangle(img, rect, color):
    cv.rectangle(img, rect[0], rect[1], color, 1)

def draw_vector(img, vector, color, thickness):
    vec, c1 = vector
    c2 = (c1[0]+vec[0], c1[1]+vec[1])
    cv.arrowedLine(img, c1, c2, color, thickness, tipLength=0.2)


def find_template(img, roi, template): #Search template in roi of img

    h, w, c = template.shape

    ROI = img[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]

    res = cv.matchTemplate(ROI, template, cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    top_left = (max_loc[0] + roi[0][0], max_loc[1] + roi[0][1])
    bottom_right = (top_left[0] + w, top_left[1] + h)

    found = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    return (top_left, bottom_right), found #Returns rectangle of found template and found image


def gen_targets(img, roi, target_size, num_targets, ingenuity): #Generate num_target targets of target_size size in roi of img while avoiding ingenuity

    #Image manipulation to find spots of biggest contrast
    ROI = img[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
    ROIblur = cv.GaussianBlur(ROI, (5,5), 0)
    laplacian = 10*np.uint8(np.absolute(cv.Laplacian(ROIblur,cv.CV_64F)))
    blur = cv.GaussianBlur(laplacian, (5,5), 0)
    blurGRAY = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)

    #Hiding Ingenuity's shadow
    blurGRAY[max(0,ingenuity[0][1]-roi[0][1]):min(blurGRAY.shape[0],ingenuity[1][1]-roi[0][1]), max(0,ingenuity[0][0]-roi[0][0]):min(blurGRAY.shape[0],ingenuity[1][0]-roi[0][0])] = np.zeros(blurGRAY[max(0,ingenuity[0][1]-roi[0][1]):min(blurGRAY.shape[0],ingenuity[1][1]-roi[0][1]), max(0,ingenuity[0][0]-roi[0][0]):min(blurGRAY.shape[0],ingenuity[1][0]-roi[0][0])].shape[:2])

    TARGETS = []

    for i in range(num_targets):

        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(blurGRAY)
        top_left = (roi[0][0] + max_loc[0] - target_size[0] // 2, roi[0][1] + max_loc[1] - target_size[1] // 2)
        bottom_right = (roi[0][0] + max_loc[0] + target_size[0] // 2, roi[0][1] + max_loc[1] + target_size[1] // 2)
        target = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        TARGETS.append([(top_left, bottom_right), target])

        #Hiding target
        blurGRAY[max(0,max_loc[1]-target_size[1]//2):min(blurGRAY.shape[0],max_loc[1]+target_size[1]//2), max(0,max_loc[0]-target_size[0]//2):min(blurGRAY.shape[1],max_loc[0]+target_size[0]//2)] = np.zeros(blurGRAY[max(0,max_loc[1]-target_size[1]//2):min(blurGRAY.shape[0],max_loc[1]+target_size[1]//2), max(0,max_loc[0]-target_size[0]//2):min(blurGRAY.shape[1],max_loc[0]+target_size[0]//2)].shape[:2])

    return TARGETS


def feature_matching(img1, img2, roi1, roi2, target_size, num_targets): #Find matching points between roi1 of img1 and roi2 of img2 with num_target targets of target_size size

    ingenuity1 = find_template(img1, roi1, ingenuity_template)
    ing_pos1 = ingenuity1[0]

    TARGETS_img1 = gen_targets(img1, roi1, target_size, num_targets, ing_pos1)

    srch_img = img2.copy()
    ingenuity2 = find_template(img2, roi1, ingenuity_template)
    ing_pos2 = ingenuity2[0]

    """#Blackout ingenuity
    for j in range(3):
        srch_img[ing_pos2[0][1]:ing_pos2[1][1], ing_pos2[0][0]:ing_pos2[1][0], j] = np.zeros(srch_img[ing_pos2[0][1]:ing_pos2[1][1], ing_pos2[0][0]:ing_pos2[1][0]].shape[:2])"""

    FOUND_img2 = []
    for target in TARGETS_img1:
        found = find_template(srch_img, roi2, target[1])
        FOUND_img2.append(found)

    return TARGETS_img1, FOUND_img2, ing_pos1, ing_pos2


def find_movement(ing_pos1, ing_pos2, TARGETS_img1, FOUND_img2, min_vector):

    ing_vect = create_vector(ing_pos1, ing_pos2)
    ing_center1 = ((ing_pos1[0][0] + ing_pos1[1][0])//2, (ing_pos1[0][1] + ing_pos1[1][1])//2)
    ing_center2 = ((ing_pos2[0][0] + ing_pos2[1][0])//2, (ing_pos2[0][1] + ing_pos2[1][1])//2)

    VECTORS = []
    for j in range(num_targets):
        vector = create_vector(TARGETS_img1[j][0], FOUND_img2[j][0])
        VECTORS.append(vector)

    #Research of a combination of vectors (more than min_vector) with lowest standard deviation
    min_sigma = np.inf
    min_VECTORS_sel = []
    for j in range (min_vector, len(VECTORS)):

        VECTORS_combinations = it.combinations(VECTORS, j)

        for VECTORS_sel in VECTORS_combinations:

            VECTORS_X = [vector[0][0] for vector in VECTORS_sel]
            VECTORS_Y = [vector[0][1] for vector in VECTORS_sel]

            sigma_x = np.std(VECTORS_X)
            sigma_y = np.std(VECTORS_Y)
            sigma2 = sigma_x*sigma_x + sigma_y*sigma_y

            if sigma2 <= min_sigma:
                min_sigma = sigma2
                min_VECTORS_sel = VECTORS_sel

    #Mean vector
    mean_vector = (np.array([int(np.mean([vector[0][0] for vector in min_VECTORS_sel])), int(np.mean([vector[0][1] for vector in min_VECTORS_sel]))]), ing_center1)

    #Real movement
    movement = (ing_vect[0] - mean_vector[0], ing_center2)

    return movement, VECTORS, min_VECTORS_sel


def trace_trajectory(images_filename, begin, end, ingenuity_template, num_targets, target_size, min_vector):

    print("Calculating trajectory...")

    POSITIONS = [np.array([0,0])]

    for i in range(begin, end):
        name1 = images_filename + str(i) + ".png"
        name2 = images_filename + str(i+1) + ".png"

        img1 = cv.imread(name1)
        img2 = cv.imread(name2)

        H, W, C = img1.shape
        roi1 = [(W//4, H//3), (3*W//4, 7*H//8)]
        roi2 = [(W//8, 0), (7*W//8, H-1)]

        TARGETS_img1, FOUND_img2, ing_pos1, ing_pos2 = feature_matching(img1, img2, roi1, roi2, target_size, num_targets)

        movement, VECTORS, min_VECTORS_sel = find_movement(ing_pos1, ing_pos2, TARGETS_img1, FOUND_img2, min_vector)

        POSITIONS.append(movement[0] + POSITIONS[-1])

        print(i)

    print("Trajectory calculated !")

    return POSITIONS


def anim_trajectory_HiRise(background, images_filename, begin, end, ingenuity_template, num_targets, target_size, min_vector, point_template, ingenuity_marker):

    hm, wm, cm = ingenuity_marker.shape

    POSITIONS = trace_trajectory(images_filename, begin, end, ingenuity_template, num_targets, target_size, min_vector)

    print("Creating HiRise images...")

    direction = (POSITIONS[-1] - POSITIONS[0], POSITIONS[0])

    Hbg, Wbg, Cbg = background.shape
    start = find_template(background, [(0,0), (Wbg, Hbg//2)], point_template)[0]
    finish = find_template(background, [(0,Hbg//2), (Wbg, Hbg-1)], point_template)[0]
    #start = find_template(background, [(0,0), (Wbg//2, Hbg-1)], point_template)[0]
    #finish = find_template(background, [(Wbg//2,0), (Wbg, Hbg-1)], point_template)[0]
    bg_direction = create_vector(start, finish)

    theta = np.math.atan2(np.linalg.det([direction[0], bg_direction[0]]),np.dot(direction[0], bg_direction[0]))
    alpha = np.linalg.norm(bg_direction[0]) / np.linalg.norm(direction[0])

    M = alpha * np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])


    POSITIONS_BG = [(M.dot(pos) + bg_direction[1]).astype(int) for pos in POSITIONS]

    IMAGES = []

    for i in range(1, len(POSITIONS_BG)):
        bg_copy = background.copy()

        top_left = (POSITIONS_BG[i][0] - wm//2, POSITIONS_BG[i][1] - hm)
        bottom_right = (POSITIONS_BG[i][0] + wm//2, POSITIONS_BG[i][1])
        roi = bg_copy[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        for j in range(roi.shape[0]):
            for k in range(roi.shape[1]):
                if ingenuity_marker[j,k,3] > 0.1:
                    roi[j,k] = ingenuity_marker[j,k,:3]

        cv.line(background, POSITIONS_BG[i-1], POSITIONS_BG[i], WHITE, thickness = Hbg//400)
        #cv.circle(bg_copy, POSITIONS_BG[i], Wbg//400, RED, thickness=-1)

        print(i)

        IMAGES.append(bg_copy)

    print("HiRise images created !")

    return IMAGES


#Initialisation
flight_num = 6
begin = 0
end = 100
num_targets = 10
target_size = [20, 20]
min_vector = 4
height = 720
scaling = 2

background = cv.imread("0-Backgrounds\\Flight " + str(flight_num) +".png")
ingenuity_template = cv.imread("0-Templates\\Template Ingenuity 10m.png")
point_template = cv.imread("0-Templates\\Ingenuity Point.png")
ingenuity_marker = cv.imread("0-Templates\\Ingenuity Marker.png", cv.IMREAD_UNCHANGED)
images_filename = "0-Images\\Flight " + str(flight_num) +"\\"

NAVCAM_IMAGES = [cv.imread(images_filename + str(i) + ".png") for i in range(begin+1, end)]

HIRISE_IMAGES = anim_trajectory_HiRise(background, images_filename, begin, end, ingenuity_template, num_targets, target_size, min_vector, point_template, ingenuity_marker)

print("Creating double images...")

navH, navW, navC = NAVCAM_IMAGES[0].shape
hiriseH, hiriseW, hiriseC = HIRISE_IMAGES[0].shape

nav_scale = height / navH
#hirise_scale = height / hiriseH
hirise_scale = height / (hiriseH*scaling)

IMAGES = []

for i in range(len(NAVCAM_IMAGES)):
    nav_img = cv.resize(NAVCAM_IMAGES[i], (int(navW*nav_scale), height))
    #hirise_img = cv.resize(HIRISE_IMAGES[i], (int(hiriseW*hirise_scale), height))
    #width = nav_img.shape[1] + hirise_img.shape[1]
    #img = np.zeros((height,width,3), np.uint8)
    #img[:,:hirise_img.shape[1]] = hirise_img
    #img[:,hirise_img.shape[1]:] = nav_img

    hirise_img = cv.resize(HIRISE_IMAGES[i], (int(hiriseW*hirise_scale), height//scaling))
    nav_img[:height//scaling,nav_img.shape[1]-hirise_img.shape[1]:] = hirise_img

    IMAGES.append(nav_img)

    cv.imwrite("7-HiRise + NavCam\\" + str(i) + ".png", nav_img)

    print(i)

print("Double images created !")

print("Creating GIF...")

with imageio.get_writer("7-HiRise + NavCam\\GIF.gif", mode="I", fps=5) as writer:
    for idx, frame in enumerate(IMAGES):
        print("Adding frame to GIF file: ", idx)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        writer.append_data(rgb_frame)

print("GIF created !")
