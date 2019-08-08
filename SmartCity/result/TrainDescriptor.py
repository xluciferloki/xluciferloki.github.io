import os
import sys
import csv
import random
import re
import pickle
import numpy as np
import pandas
import cv2
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
root = tk.Tk()
root.withdraw()
# Folderpath = filedialog.askdirectory()
# Filepath = filedialog.askopenfilename() #获得选择好的文件
Folderpath= '../../DataSet/SmartCity/scene1_jiading_lib_training'
istrain =0;coors=[]
Thumbnails={};TotalStations={}
for root,dirs,files in os.walk(Folderpath):
    for file in files:
        if re.match('thumbnail.jpg',file):
            Thumbnails[root.split('/')[-1]] = root+'/'+file
        if re.match('TotalStation.jpg',file):
            TotalStations[root.split('/')[-1]] = root+'/'+file
        if re.match('\w+coordinates.csv',file):
            istrain = 1
            coors = [root.split('/')[-1],root+'/'+file]
PICcoordinates = np.loadtxt(coors[1], dtype=np.str, delimiter=',')
PICname = PICcoordinates[1:,0]
PICcoor = PICcoordinates[1:,1:].astype(np.float)
location = {}
for pic in PICname:
    location[pic] = PICcoor
sift = cv2.xfeatures2d.SIFT_create()
imgto = {}
kps = {}
des = {}
for to in TotalStations:
    imgto[to] = cv2.imread(TotalStations[to])
    # img = cv2.resize(img,(30,40))
    (kps[to], des[to]) = sift.detectAndCompute(imgto[to], None)
kp = {};img1 = {}
for k in kps:
    kp[k] = np.float32([kp.pt for kp in kps[k]])
    img1[k] = cv2.drawKeypoints(imgto[k], kps[k], np.array([]))
    # cv2.imshow(k,img1[k])
    plt.imshow(img1[k])
    plt.ion()
matcher = cv2.DescriptorMatcher_create('BruteForce')
imgName = [k for k in imgto.keys()]
pic1 = imgName[0]
pic2 = imgName[1]
matches = matcher.knnMatch(des[pic1], des[pic2], 2)
goo = []
good = {}
ratio = 0.75
for m in matches:
    if len(m) == 2 and m[0].distance < ratio * m[1].distance:
        goo.append((m[0].queryIdx, m[0].trainIdx))
good[pic1] = [kps[pic1][i] for (i, _) in goo]
good[pic2] = [kps[pic2][i] for (_, i) in goo]
img2 = {}
for k in kps:
    imgdata = (np.ones((50, 50, 3)) * 255).astype(np.uint8)
    kp[k] = np.float32([kp.pt for kp in kps[k]])
    img2[k] = cv2.drawKeypoints(imgto[k],good[k],imgdata)
    # cv2.imshow(k,img2[k])
    plt.imshow(img2[k])
    plt.ion()
# plt.imshow(img[pic2])
plt.ioff()
plt.show()
# with open('../data/good.pkl','wb') as f:
#     pickle.dump(good,f)
# cv2.imshow('IMAGE', img)
# cv2.imwrite('SURF.png', img)
# cv2.waitKey()
if __name__ == '__main__':
    a=1
##---------------------------
import random
tmp = []
for i in range(7500):
    tmp.append(random.uniform(250,256))
imgdata = np.array(tmp, dtype=np.uint8)
imgdata = imgdata.reshape(50,50,3)
plt.imshow(imgdata)
plt.show()
plt.clf()
plt.close()