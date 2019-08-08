import os
import sys
import cv2
import glob
import csv
import numpy as np
import pickle
import imutils

from PIL import Image
import pylab
import tkinter.filedialog

def save(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

path = '/home/loki/PycharmProjects/test1/code'
os.chdir(path)

ratio = 0.75
reprojThresh = 4.0

location = load('location')
keypointsPosition = load('keypointsPosition')
descriptors = load('descriptors')
imgNames = load('imgNames')

imgReadPath = "/home/loki/Desktop/smart city literature/rui_an_20190420"
# imgNames = glob.glob(imgReadPathTrains + '/*/thumbnail.jpg')
# imgNames.sort()
picture = list(keypointsPosition.keys())

MatcherPositionData = {pic:{} for pic in picture}

for pic1 in picture:
    for pic2 in picture:
        # pic1 = picture[0]
        # pic2 = picture[1]
        kps_pic1 = keypointsPosition[pic1]
        kps_pic2 = keypointsPosition[pic2]

        # picPathName1 = imgReadPath + os.path.sep + pic1 + '/thumbnail.jpg'
        # picPathName2 = imgReadPath + os.path.sep + pic2 + '/thumbnail.jpg'
        #
        # img1 = cv2.imread(picPathName1)
        # img2 = cv2.imread(picPathName2)
        # img1 = imutils.resize(img1, width=640)
        # img2 = imutils.resize(img2, width=640)

        matcher = cv2.DescriptorMatcher_create('BruteForce')
        matches = matcher.knnMatch(descriptors[pic1], descriptors[pic2], 2)
        good = []
        for m in matches:
            if len(m) == 2 and m[0].distance < ratio * m[1].distance:
                good.append((m[0].trainIdx, m[0].queryIdx))

        good_pic1_pts = np.float32([kps_pic1[i] for (_, i) in good])
        good_pic2_pts = np.float32([kps_pic2[i] for (i, _) in good])
        MatcherPositionData[pic1][pic2]=[good_pic1_pts,good_pic2_pts]

save(MatcherPositionData,'MatcherPositionData')