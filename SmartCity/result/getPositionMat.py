# coding: utf-8
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
from matplotlib.image import imread
from PIL import Image
from selenium.webdriver.common.action_chains import ActionChains
import termios

root = tk.Tk()
root.withdraw()
folderPath= '../../DataSet/SmartCity/scene1_jiading_lib_training'
Thumbnails = {}
Thumbnailloc = {}
for item in os.walk(folderPath):
    for file in item[2]:
        if re.match('thumbnail.jpg', file):
            Thumbnails[item[0].split('/')[-1]] = item[0] + '/' + file
        if re.match('thumbnail.txt', file):
            Thumbnailloc[item[0].split('/')[-1]] = item[0] + '/' + file
br=0
pos = {}
for th in Thumbnails:
    txt = Thumbnails[th].split('.jpg')[0] + '.txt'    # im = Image.open(Thumbnails[th])
    # im = cv2.imread(Thumbnails[th])
    if not os.path.exists(txt):
        im = imread(Thumbnails[th])
        # plt.ion()
        plt.imshow(im)
        # cv2.waitKey(0)
        plt.pause(10)
        while input('输入1,记录坐标')==1:break
        plt.pause(10)
        # plt.show()
        tmp = []
        # while True:
        #     pg = plt.ginput(3)
        #     if input('输入2,保存坐标')==2:
        #         for tm in pg:
        #             tmp.append((int(tm[0]), int(tm[1])))
        #         break
        for tm in plt.ginput(3):
            tmp.append((int(tm[0]), int(tm[1])))
        pos[th] = tmp
        np.savetxt(txt,tmp)
        # br += 1
        # if br > 0: break
with open('../data/pos.pkl','wb') as f:
    pickle.dump(pos,f)
if __name__ == '__main__':
    a=1
###--------------------------
# # 使用我们上面说的灰度图
# cmap = plt.get_cmap('Greys')
# # cmap = plt.cm.Greys #也可以这么写
# # 利用normlize来标准化颜色的值
# norm = plt.Normalize(vmin=-3, vmax=3)
# plt.imshow(im, cmap=cmap, norm=norm)
# plt.show()
# pos = plt.ginput(3)
# print(pos)
# plt.imshow(im, cmap=plt.get_cmap("gray"), vmin=100, vmax=150)
# plt.show()
# im_1 = np.transpose(im[:,:,0])
# im_1 = im[:,:,0]
# plt.imshow(im_1)
# plt.show()
# # 此时会发现显示的是热量图，不是我们预想的灰度图，可以添加 cmap 参数，有如下几种添加方法：
# plt.imshow(im_1, cmap='Greys_r')
# plt.show()
# img = plt.imshow(im_1)
# img.set_cmap('gray') # 'hot' 是热量图
# plt.show()
# b="%d"%2.34
plt.plot([1,2,3],[5,6,7])