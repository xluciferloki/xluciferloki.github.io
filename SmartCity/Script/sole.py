# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 21:35:36 2019

@author: Nanan-COLL123456
"""

import cv2
import imutils
import glob

imgsReadPath = "../result"
imgsName = glob.glob(imgsReadPath + '/undistortIMG*.jpg')
img1 = cv2.imread(imgsName[0])
img2 = cv2.imread(imgsName[1])
img1 = imutils.resize(img1, width=480)
img2 = imutils.resize(img2, width=480)

img = img2

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
surf = cv2.xfeatures2d.SURF_create(5000)
keypoints, descriptor = surf.detectAndCompute(gray, None)
keypoint = [kp.pt for kp in keypoints]
img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        color=(51, 163, 236))
cv2.imshow('IMAGE', img)
cv2.imwrite('SURF2.png', img)
while True:
    if cv2.waitKey() & 0xff == ord('q'):
        break
cv2.destroyAllWindows()
