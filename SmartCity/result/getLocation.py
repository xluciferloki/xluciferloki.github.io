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
root = tk.Tk()
root.withdraw()
folderPath= '../../DataSet/SmartCity/scene1_jiading_lib_training'
Thumbnails = {}
for item in os.walk(folderPath):
    for file in item[2]:
        if re.match('thumbnail.jpg', file):
            Thumbnails[item[0].split('/')[-1]] = item[0] + '/' + file
br=0
for th in Thumbnails:
    img = cv2.imread(Thumbnails[th])
    br+=1
    if br>1:break
# print img.shape
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)


cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", img)

while (True):
    try:
        cv2.waitKey(100)
    except Exception:
        cv2.destroyAllWindows()
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
