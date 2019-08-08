# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 10:55:37 2019

@author: Nanan-COLL123456
"""

import numpy as np
import cv2
import imutils
import glob

class Stitcher:
    def __init__(self):
        self.isv3 = imutils.is_cv3()

    def stitch(self, imgs, ratio=0.75, reprojThresh = 4.0, showMatches = False):
        print('A')
        (img1, img2) = imgs
        #获取关键点和描述子
        (kp1, des1) = self.detectAndDescribe(img1)
        (kp2, des2) = self.detectAndDescribe(img2)
        print(len(kp1),len(des1))
        print(len(kp2), len(des2))
        R = self.matchKeyPoints(kp1, kp2, des1, des2, ratio, reprojThresh)

        #如果没有足够的最佳匹配点，M为None
        if R is None:
            return  None
        (good, M, mask) = R
        print(M)
        #对img1透视变换，M是ROI区域矩阵， 变换后的大小是(img1.w+img2.w, img1.h)
        result = cv2.warpPerspective(img1, M, (img1.shape[1] + img2.shape[1], img1.shape[0]))
        #将img2的值赋给结果图像
        result[0:img2.shape[0], 0:img2.shape[1]] = img2

        #是否需要显示ROI区域
        if showMatches:
            vis = self.drawMatches1(img1, img2, kp1, kp2, good, mask)
            return (result, vis)

        return result

    def keypointsDescriptor(self,img):
        print('B')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #检查我们使用的是否是penCV3.x
        if self.isv3:
            sift = cv2.xfeatures2d.SIFT_create()
            (kps, des) = sift.detectAndCompute(img, None)
        else:
            sift = cv2.FastFeatureDetector_create('SIFT')
            kps = sift.detect(gray)
            des = sift.compute(gray, kps)

        #kp = np.float32([kp.pt for kp in kps]) #    **********************************
        #返回关键点和描述符
        return (kps, des)

    def detectAndDescribe(self,img):
        print('B')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #检查我们使用的是否是penCV3.x
        if self.isv3:
            sift = cv2.xfeatures2d.SIFT_create()
            (kps, des) = sift.detectAndCompute(img, None)
        else:
            sift = cv2.FastFeatureDetector_create('SIFT')
            kps = sift.detect(gray)
            des = sift.compute(gray, kps)

        kp = np.float32([kp.pt for kp in kps]) #    **********************************
        #返回关键点和描述符
        return (kp, des)

    def matchKeyPoints(self,kp1, kp2, des1, des2, ratio, reprojThresh):
        print('C')
        #初始化BF,因为使用的是SIFT ，所以使用默认参数
        matcher = cv2.DescriptorMatcher_create('BruteForce')
        # bf = cv2.BFMatcher()
        # matches = bf.knnMatch(des1, des2, k=2)
        matches = matcher.knnMatch(des1, des2, 2)  #***********************************

        #获取理想匹配
        good = []
        for m in matches:
            if len(m) == 2 and  m[0].distance < ratio * m[1].distance:
                good.append((m[0].trainIdx, m[0].queryIdx))

        print(len(good))
        #最少要有四个点才能做透视变换
        if len(good) > 4:
            #获取关键点的坐标
            # src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            # dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            src_pts = np.float32([kp1[i] for (_, i) in good])
            dst_pts = np.float32([kp2[i] for (i, _) in good])

            #通过两个图像的关键点计算变换矩阵
            (M, mask) = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reprojThresh)

            #返回最佳匹配点、变换矩阵和掩模
            return (good, M, mask)
        #如果不满足最少四个 就返回None
        return None

    def drawMatches(img1, img2, kp1, kp2, matches, mask, M):
        # 获得原图像的高和宽
        h, w = img1.shape[:2]
        # 使用得到的变换矩阵对原图像的四个角进行变换，获得目标图像上对应的坐标
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        matchesMask = mask.ravel().tolist()

        draw_params = dict(matchColor = (0, 255, 0),
                           singlePointColor = None,
                           matchesMask = matchesMask,
                           flags = 2)
        img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)

        return img

    def drawMatches1(self,img1, img2, kp1, kp2, metches,mask):
        print('D')
        (hA,wA) = img1.shape[:2]
        (hB,wB) = img2.shape[:2]
        vis = np.zeros((max(hA,hB), wA+wB, 3), dtype='uint8')
        vis[0:hA, 0:wA] = img1
        vis[0:hB, wA:] = img2
        for ((trainIdx, queryIdx),s) in zip(metches, mask):
            if s == 1:
                ptA = (int(kp1[queryIdx][0]), int(kp1[queryIdx][1]))
                ptB = (int(kp2[trainIdx][0])+wA, int(kp2[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        return vis

def drawkeypoint(st,img):

    stitcher = Stitcher()
    keypoints,_=stitcher.keypointsDescriptor(img)

    img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                            color=(51, 163, 236))

    cv2.imshow(st, img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


def drawresultvis(st,img):
    stitcher = Stitcher()

    (result, vis) = stitcher.stitch(img, showMatches=True)
    # (result, vis) = stitch([img1,img2], showMatches=True)
    cv2.imshow(st + 'keyPoint Matches', vis)

    cv2.imshow(st + 'Result', result)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

# def imgRead(path):

def getKeypointDescriptor(img):
    print('B')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #检查我们使用的是否是penCV3.x
    if imutils.is_cv3():
        sift = cv2.xfeatures2d.SIFT_create()
        (kps, des) = sift.detectAndCompute(img, None)
    else:
        sift = cv2.FastFeatureDetector_create('SIFT')
        kps = sift.detect(gray)
        des = sift.compute(gray, kps)

    kp = np.float32([kp.pt for kp in kps]) #    **********************************
    #返回关键点和描述符
    return (kp, des)

def getGoodMatchPoint(kp1, kp2, des1, des2, ratio, reprojThresh):
    print('C')
    #初始化BF,因为使用的是SIFT ，所以使用默认参数
    matcher = cv2.DescriptorMatcher_create('BruteForce')
    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(des1, des2, k=2)
    matches = matcher.knnMatch(des1, des2, 2)  #***********************************

    #获取理想匹配
    good = []
    for m in matches:
        if len(m) == 2 and  m[0].distance < ratio * m[1].distance:
            good.append((m[0].trainIdx, m[0].queryIdx))

    src_pts = np.float32([kp1[i] for (_, i) in good])
    dst_pts = np.float32([kp2[i] for (i, _) in good])

    return [src_pts,dst_pts]
 
if __name__=='__main__':
    imgReadPath = "/home/loki/Desktop/smart city literature/rui_an_20190420"
    imgNames = glob.glob(imgReadPath + '/*/thumbnail.jpg')
    imgNames.sort()
    img1 = cv2.imread(imgNames[0])
    img2 = cv2.imread(imgNames[1])
    img1 = imutils.resize(img1, width=720)
    img2 = imutils.resize(img2, width=720)
    drawresultvis('Beforedraw', [img1, img2])
    drawkeypoint('img1',img1)
    #drawkeypoint('img2',img2)
    #drawresultvis('Afterdraw',[img1,img2])
    cv2.waitKey(0)
