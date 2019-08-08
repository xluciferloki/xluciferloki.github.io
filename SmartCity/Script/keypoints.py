import os
import sys
import cv2
import glob
import csv
import numpy as np
import pickle
def save(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


os.chdir(os.getcwd())
sys.path.append('code')

import Stitcher
# Class Stitcher have imported np,cv2,imutils,glob

def datamain():
    imgReadPathTrains = "/home/loki/Desktop/smart city literature/rui_an_20190420"
    imgNames = glob.glob(imgReadPathTrains + '/*/thumbnail.jpg')
    imgNames.sort()
    save(imgNames,'imgNames')

    stitcher = Stitcher.Stitcher()

    imgName = imgNames[0]
    csvWritePath = '/home/loki/Desktop/smart city literature/kps_des_rui_an'
    if not os.path.exists(csvWritePath):
        os.makedirs(csvWritePath)

    keypointsPosition = {}
    keypoints = {}
    descriptors = {}
    for imgName in imgNames:
        img = cv2.imread(imgName)
        imglist = imgName.split('/')
        sift = cv2.xfeatures2d.SIFT_create()
        (kps, des) = sift.detectAndCompute(img, None)
        kp = np.float32([kp.pt for kp in kps])

        keypointsPosition[imglist[6]] = kp
        keypoints[imglist[6]] = kps
        descriptors[imglist[6]] = des

        # exec('keypoint,descriptor  = Stitcher.getKeypointDescriptor(img)')
        # with open(csvWriteName, 'w', newline='') as file:
        #     csvwriter = csv.writer(file)
        #     csvwriter.writerows(keypoint)
        # csvWriteName = csvWritePath + os.path.sep + imglist[6] + '_keypoints.csv'
        # np.savetxt(csvWriteName,kps,delimiter=',')

        csvWriteName = csvWritePath + os.path.sep + imglist[6] + '_keypointsPosition.csv'
        np.savetxt(csvWriteName, kp, delimiter=',')

        csvWriteName = csvWritePath + os.path.sep + imglist[6] + '_descriptors.csv'
        np.savetxt(csvWriteName, des, delimiter=',')
    save(keypointsPosition,'keypointsPosition')
    # save(keypoints, 'keypoints')
    save(descriptors, 'descriptors')


    locationDataPath = '/home/loki/Desktop/smart city literature/rui_an_20190420'
    locationDataName = 'rui_an_20190420_coordinates.csv'
    data_rui_an = np.loadtxt(locationDataPath + os.path.sep + locationDataName, dtype=np.str, delimiter=',')
    locationPicture = data_rui_an[1:,0]
    locationPosition = data_rui_an[1:,1:].astype(np.float)
    location = {}
    for i in range(locationPicture.shape[0]):
        location[locationPicture[i]] = locationPosition[i,:]

    save(location,'location')
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__=='__main__':
    datamain()