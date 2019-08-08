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
# Folderpath = filedialog.askdirecthry()
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
imgth = {}
kps = {}
des = {}
br=0
for th in Thumbnails:
    imgth[th] = cv2.imread(Thumbnails[th])
    br+=1
    if br>1:break
    # img = cv2.resize(img,(30,40))
    (kps[th], des[th]) = sift.detectAndCompute(imgth[th], None)
kp = {};img = {}
for k in kps:
    kp[k] = np.float32([kp.pt for kp in kps[k]])
    img[k] = cv2.drawKeypoints(image=imgth[k], outImage=imgth[k], keypoints=kps[k], flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,color=(51, 163, 236))
    # cv2.imshow(k,img[k])
    # plt.imshow(img[k])
    # plt.show()
imgName = [k for k in imgth.keys()]
imgName0 = imgName[0]
imgName1 = imgName[1]
img0 = imgth[imgName0]
img1 = imgth[imgName1]
img = {}
orb = cv2.ORB_create()
kp0, des0 = orb.detectAndCompute(img0, None)
kp1, des1 = orb.detectAndCompute(img1, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
# 对每个匹配选择两个最佳的匹配
matches = bf.knnMatch(des0, des1, k=2)
print(type(matches), len(matches), matches[0])
# 获取img1中的第一个描述符即[0][]在img2中最匹配即[0][0]的一个描述符  距离最小
dMatch0 = matches[0][0]
# 获取img1中的第一个描述符在img2中次匹配的一个描述符  距离次之
dMatch1 = matches[0][1]
print('knnMatches', dMatch0.distance, dMatch0.queryIdx, dMatch0.trainIdx)
print('knnMatches', dMatch1.distance, dMatch1.queryIdx, dMatch1.trainIdx)
# 将不满足的最近邻的匹配之间距离比率大于设定的阈值匹配剔除。
img3 = None
img3 = cv2.drawMatchesKnn(img0, kp0, img1, kp1, matches, img3, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
img3 = cv2.resize(img3, (1000, 400))
plt.imshow(img3)
plt.show()
# cv2.imshow('KNN', img3)
# cv2.waitKey()
# cv2.destroyAllWindows()
# with open('../data/good.pkl','wb') as f:
#     pickle.dump(good,f)
# cv2.imwrite('SURF.png', img)
if __name__ == '__main__':
    a=1

##----------------------------
import random
tmp = []
for i in range(300):
    tmp.append(random.uniform(0,255))
imgdata = np.array(tmp, dtype=np.uint8)
imgdata = imgdata.reshape(10,10,3)
# imgdata = imgdata.astype(np.str)
plt.imshow(imgdata)
plt.show()