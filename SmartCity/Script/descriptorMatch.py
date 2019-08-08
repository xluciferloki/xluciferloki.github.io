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
sys.path.append(path)

import Stitcher
# Class Stitcher have imported np,cv2,imutils,glob
stitcher =Stitcher.Stitcher()

def main():
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
    pic1 = picture[0]
    pic2 = picture[1]
    kps_pic1 = keypointsPosition[pic1]
    kps_pic2 = keypointsPosition[pic2]

    picPathName1 = imgReadPath + os.path.sep + pic1 + '/thumbnail.jpg'
    picPathName2 = imgReadPath + os.path.sep + pic2 + '/thumbnail.jpg'

    img1 = cv2.imread(picPathName1)
    img2 = cv2.imread(picPathName2)
    # img1 = imutils.resize(img1, width=640)
    # img2 = imutils.resize(img2, width=640)

    matcher = cv2.DescriptorMatcher_create('BruteForce')
    matches = matcher.knnMatch(descriptors[pic1], descriptors[pic2], 2)
    good = []
    for m in matches:
        if len(m) == 2 and  m[0].distance < ratio * m[1].distance:
            good.append((m[0].trainIdx, m[0].queryIdx))

    # for m in matches:
    #     # if len(m) == 2 and  m[0].distance < ratio * m[1].distance:
    #     good.append((m[0].trainIdx, m[0].queryIdx))

    good_pic1_pts = np.float32([kps_pic1[i] for (_, i) in good])
    good_pic2_pts = np.float32([kps_pic2[i] for (i, _) in good])


    (M, mask) = cv2.findHomography(good_pic1_pts, good_pic2_pts, cv2.RANSAC, reprojThresh)
    #对img1透视变换，M是ROI区域矩阵， 变换后的大小是(img1.w+img2.w, img1.h)
    # result = cv2.
    # (img1, M, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    #将img2的值赋给结果图像
    # result[0:img2.shape[0], 0:img2.shape[1]] = img2

    (hA,wA) = img1.shape[:2]
    (hB,wB) = img2.shape[:2]
    vis = np.zeros((max(hA,hB), wA+wB, 3), dtype='uint8')
    vis[0:hA, 0:wA] = img1
    vis[0:hB, wA:] = img2

    for ((trainIdx, queryIdx), s) in zip(good, mask):
        if s == 1:
            ptA = (int(kps_pic1[queryIdx][0]), int(kps_pic1[queryIdx][1]))
            ptB = (int(kps_pic2[trainIdx][0]) + wA, int(kps_pic2[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (255, 0, 0), 1)

    # for (trainIdx, queryIdx) in good:
    #     # if s == 1:
    #     ptA = (int(kps_pic1[queryIdx][0]), int(kps_pic1[queryIdx][1]))
    #     ptB = (int(kps_pic2[trainIdx][0]) + wA, int(kps_pic2[trainIdx][1]))
    #     cv2.line(vis, ptA, ptB, (255, 0, 0), 1)

    cv2.imwrite('visStitcher.jpg', vis)
    cv2.imshow('vis',vis)
    # cv2.waitKey(0)

    # im = pylab.array(Image.open(picPathName1))
    # pylab.imshow(im)
    #
    # x = good_pic1_pts[:,0]
    # y = good_pic1_pts[:,1]
    # # 使用红色星状标记绘制点
    # pylab.plot(x,y,'b*')
    # pylab.plot()
    # # pylab.imshow(vis)
    # pylab.show()

    # Stitcher.drawresultvis('matcher',[img1,img2])
    # cv2.imshow('result',result)
    # cv2.imshow('vis',vis)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()

