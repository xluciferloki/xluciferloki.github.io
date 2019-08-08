import cv2
assert cv2.__version__[0] == '3'
import numpy as np
import os
import glob
from os.path import join as pjoin
from scipy import misc



def get_K_and_D(checkerboard, imgsPath):

    CHECKERBOARD = checkerboard
    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    _img_shape = None
    objpoints = [] 
    imgpoints = [] 
    


    
    images = cv2.imread("F:/S---city/dataset--/rui_an_20190420/PIC_20190420_162718/origin_1.jpg")
    for fname in images:
        img = cv2.imread(fname)
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."
        
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            imgpoints.append(corners)
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    rms, _, _, _, _ = \
    cv2.fisheye.calibrate(
                                objpoints,
                                imgpoints,
                                gray.shape[::-1],
                                K,
                                D,
                                rvecs,
                                tvecs,
                                calibration_flags,
                                (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
                                )
    DIM = _img_shape[::-1]
    print("Found " + str(N_OK) + " valid images for calibration")
    print("DIM=" + str(_img_shape[::-1]))
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")
    
    return DIM, K, D

def undistort(K,D,img_path):
    img = cv2.imread(img_path)
    DIM = img.shape[:2]
    img = cv2.resize(img, DIM)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM,cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)    
    cv2.imwrite('unfisheyeImage.png', undistorted_img)


def show():
    from PIL import Image
    from numpy import asarray
    imgsReadPath = "/home/loki/DataSet/SmartCity/rui_an_20190420"
    imgsName = glob.glob(imgsReadPath + '/*/*.jpg')
    for imgName in imgsName:
        #imgName =imgsName[3]
        img=cv2.imread(imgName)

        _img_shape=img.shape[:2]
        #DIM=_img_shape[::-1]
        DIM=(640,480)
        img = cv2.resize(img, DIM)
        print(DIM)

        K=np.array([[194.938,0,319.337],[0,194.393,240.487],[0,0,1]])
        D=np.array([-0.0312506,-0.00373898,-0.000650069,0.000174612])

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3),K,img.shape[:2],cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)

        IMG = Image.fromarray(undistorted_img)
        IMG.show()
        imgSplit = imgName.split('/')
        # imgsWritePath0 = os.getcwd()
        #imgsWritePath = '~/Desktop/smart city literature/' + 'undistorted_rui_an_20190420/' + imgSplit[6]+'/'
        # imgsWritePath = './undistorted_rui_an_20190420/'+imgSplit[6]
        # if not os.path.exists(imgsWritePath):
        #     os.makedirs(imgsWritePath, mode=0o777)
        # os.chdir(imgsWritePath)
        # cv2.imwrite(imgSplit[7], undistorted_img)
        # os.chdir(imgsWritePath0)
    #cv2.imshow('110',undistorted_img)
    #cv2.imshow('resize img',img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #print(undistorted_img.shape)


if __name__=='__main__':
    show()