#/home/loki/anaconda3/envs/cv/bin/python
#-*-coding: utf-8-*-
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
import matplotlib.pyplot as plt
from matplotlib.image import imread
from PIL import Image
from selenium.webdriver.common.action_chains import ActionChains
import termios
def tkopen(f=1):
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    if f==1:
        folderPath = tk.filedialog.askdirectory()
    if f!=1:
        folderPath = tk.filedialog.askopenfilename() #获得选择好的文件
    return folderPath
def readfolderfile(reg, path, isFile=True):
    from os import walk
    from re import match
    if isFile:
        File = {}
        for item in walk(path):
            for file in item[2]:
                if match(reg, file):
                    File[item[0].split('/')[-1]] = item[0] + '/' + file
        return File
    else:
        directory = {}
        for root, dirs, files in walk(path):
            for dr in dirs:
                if match(reg,dr):
                    directory[dr] = root + '/' + dr
        return directory
def pklopen(pklpath):
    import pickle,os
    if os.path.exists(pklpath):
        with open(pklpath, 'rb') as f:
            plf = pickle.load(f)
        return plf
    return {}
def pkldump(obj,fopen):
    import pickle
    with open(fopen, 'wb') as f:
        plf = pickle.dump(obj,f)
def imgread(imgspath,br=None):
    from PIL import Image
    if br is None:
        Img = {}
        for imgp in imgspath:
            Img[imgp] = Image.open(imgspath[imgp])
    if br is not None:
        Img = []
        for imgp in imgspath:
            Img.append(Image.open(imgspath[imgp]))
            br[0]+=1
            if br[0]>br[1]:break
    return Img
def csvread(csvpath,f=1):
    from os.path import exists
    from numpy import loadtxt
    if not exists(csvpath):return {}
    tmp = loadtxt(csvpath, dtype=str, delimiter=',')
    name = tmp[1:,0]
    data = tmp[1:,1:].astype(float)
    if f == 1:
        rt = {}
        for i in range(len(name)):
            rt[name[i]] = data[i]
        return rt
    else:
        return name,data
def markloc(imgpath, num, delay=16):
    import matplotlib.pyplot as plt
    import matplotlib.image as matimg
    img = matimg.imread(imgpath)
    plt.imshow(img)
    plt.pause(delay)
    pos = plt.ginput(num)
    return pos
def markdefect(posdata, imgname, num, size=(1920,960),ischeckPoint = False):
    detect = []
    for pd in posdata:
        if len(posdata[pd]) != num:
            detect.append(pd)
        if ischeckPoint and len(posdata[pd]) == num:
            mx1 = 0
            mx2 = 0
            for j in range(num-1):
                mx1 = max(abs(sinxs(posdata[pd][j][0], size[0])-sinxs(posdata[pd][j+1][0],size[0])),mx1)
                mx2 = max(abs(sinxs(posdata[pd][j][1], size[1])-sinxs(posdata[pd][j+1][1],size[1])),mx2)
            if mx1 > 1.2 or mx2 > 1.2:
                detect.append(pd)
    for i in imgname:
        if i not in posdata:
            detect.append(i)
    return detect
def remark(posdata, imgpath, num=2, size=(1920,960),ischeckPoint=False,delay=16):
    '''
    :param posdata: {pic1:[(1,2),(3,4),(3,5)],[]}
    :param imgpath: {pic1:'/home/loki/...',...}
    :param leval:
    :param size:
    :return:
    '''
    imgname = [im for im in imgpath]
    md = markdefect(posdata,imgname,num, size=size, ischeckPoint=ischeckPoint)
    if md:
        for i in md:
            pos = markloc(imgpath[i],num,delay=delay)
            key = input('press any key to continue...')
            if key not in ['n','N']:
                posdata[i] = pos
            if key in ['q','Q']:
                break
    return posdata
def markpic(imgpath,num=2):
    import matplotlib.pyplot as plt
    import matplotlib.image as matimg
    plt.ion()
    pos = {}
    for i in imgpath:
        img = matimg.imread(imgpath[i])
        plt.imshow(img)
        plt.pause(5)
        while input('input 1 to click') == 1:break
        tmp = plt.ginput(num)
        while input('input 2 to save') == 2:pos[i]=tmp
    plt.ioff()
    return pos
def sinxs(x,s,l=1):
    import numpy as np
    tmx0 = np.sin(x/s*2*np.pi)**l
    return tmx0
def cosxs(x,s,l=1):
    import numpy as np
    tmx0 = np.cos(x/s*2*np.pi)**l
    return tmx0
def rousinxs(x,s,l=1):
    # tmx0 = sinxs(x,s,l)
    tmx0 = np.sqrt((x-s/2)**2)*sinxs(x,s,l)
    return tmx0
def roucosxs(x,s,l=1):
    # tmx0 = cosxs(x,s,l)
    tmx0 = np.sqrt((x-s/2)**2)*cosxs(x,s,l)
    return tmx0
def rouscCalc(x,s,key='s',l=1):
    if key == 's':
        return rousinxs(x,s,l)
    if key == 'c':
        return roucosxs(x,s,l)
    return None
def rouscFunDes(lx,lv,pt,key,arg):
    strt = ''
    if key == 's':
        strt = str(arg) + 'sqrt((x' + str(lx) + str(pt) + '-size(' + str(pt) + ')/2)^2)*sin(x' + str(lx) + str(pt) + '/size[0]*2*pi)^' + str(lv)
    if key == 'c':
        strt = str(arg) + 'sqrt((x' + str(lx) + str(pt) + '-size(' + str(pt) + ')/2)^2)*cin(x' + str(lx) + str(pt) + '/size[0]*2*pi)^' + str(lv)
    return strt
def mnglct(postotal,csvdata):
    LocationTrain = []
    for pt in postotal:
        tmp = []
        for i in range(3):
            tmp.append(postotal[pt][i][0])
            tmp.append(postotal[pt][i][1])
        for i in range(3):
            tmp.append(csvdata[pt][i])
        LocationTrain.append(tmp)
    return LocationTrain
def presentAnotherXcoor(postotal,size=(1920,960)):
    tsc = {}
    for pt in postotal:
        tmp = []
        for p in postotal[pt]:
            tm1 = roucosxs(p[0],size[0])
            tm2 = rousinxs(p[0],size[0])
            tm3 = roucosxs(p[1],size[1])
            tm4 = rousinxs(p[1],size[1])
            tp1 = ((tm1,tm2),(tm3,tm4))
            tmp.append(tp1)
        tsc[pt] = tmp
    return tsc
def mngFunDes(lenx,leval,args=''):
    rt = ''
    # for lx in lenx:
    #     for le in range(leval):
    #         rt = rt + 'sqrt((x' + str(lx) + str(0) + '-size(0)/2)^2)*sin(x' + str(lx) + str(0) + '/size[0]*2*pi)^' + str(le + 1)
    #         rt = rt + 'sqrt((x' + str(lx) + str(0) + '-size(0)/2)^2)*cos(x' + str(lx) + str(0) + '/size[0]*2*pi)^' + str(le + 1)
    #         rt = rt + 'sqrt((x' + str(lx) + str(1) + '-size(1)/2)^2)*sin(x' + str(lx) + str(1) + '/size[1]*2*pi)^' + str(le + 1)
    #         rt = rt + 'sqrt((x' + str(lx) + str(1) + '-size(1)/2)^2)*cos(x' + str(lx) + str(1) + '/size[1]*2*pi)^' + str(le + 1)
    #     for le in range(leval-1):
    #         rt = rt + 'sqrt((x' + str(lx) + str(0) + '-size(0)/2)^2)*sin(x' + str(lx) + str(0) + '/size[0]*2*pi)^' + str(le + 1) + '*sqrt((x' + str(lx) + str(1) + '-size(1)/2)^2)*sin(x' + str(lx) + str(1) + '/size[1]*2*pi)^' + str(leval - le - 1)
    #         rt = rt + 'sqrt((x' + str(lx) + str(0) + '-size(0)/2)^2)*sin(x' + str(lx) + str(0) + '/size[0]*2*pi)^' + str(le + 1) + '*sqrt((x' + str(lx) + str(1) + '-size(1)/2)^2)*sin(x' + str(lx) + str(1) + '/size[1]*2*pi)^' + str(leval - le - 1)
    #         rt = rt + 'sqrt((x' + str(lx) + str(0) + '-size(0)/2)^2)*cos(x' + str(lx) + str(0) + '/size[0]*2*pi)^' + str(le + 1) + '*sqrt((x' + str(lx) + str(1) + '-size(1)/2)^2)*cos(x' + str(lx) + str(1) + '/size[1]*2*pi)^' + str(leval - le - 1)
    #         rt = rt + 'sqrt((x' + str(lx) + str(0) + '-size(0)/2)^2)*cos(x' + str(lx) + str(0) + '/size[0]*2*pi)^' + str(le + 1) + '*sqrt((x' + str(lx) + str(1) + '-size(1)/2)^2)*cos(x' + str(lx) + str(1) + '/size[1]*2*pi)^' + str(leval - le - 1)
    # for lx1 in range(lenx):
    #     for lx2 in range(lenx):
    #         for le1 in range(leval):
    #             for le2 in range(leval):
    #                 if le1+le2+2 <= leval:
    #                     rt = rt + 'sqrt((x' + str(lx1) + str(0) + '-size(0)/2)^2)*sin(x' + str(lx1) + str(0) + '/size[0]*2*pi)^' + str(le1 + 1) + '*sqrt((x' + str(lx2) + str(0) + '-size(0)/2)^2)*sin(x' + str(lx2) + str(0) + '/size[0]*2*pi)^' + str(le2 + 1)
    #                     rt = rt + 'sqrt((x' + str(lx1) + str(0) + '-size(0)/2)^2)*sin(x' + str(lx1) + str(0) + '/size[0]*2*pi)^' + str(le1 + 1) + '*sqrt((x' + str(lx2) + str(0) + '-size(0)/2)^2)*cos(x' + str(lx2) + str(0) + '/size[0]*2*pi)^' + str(le2 + 1)
    #                     rt = rt + 'sqrt((x' + str(lx1) + str(0) + '-size(0)/2)^2)*sin(x' + str(lx1) + str(0) + '/size[0]*2*pi)^' + str(le1 + 1) + '*sqrt((x' + str(lx2) + str(1) + '-size(1)/2)^2)*sin(x' + str(lx2) + str(1) + '/size[1]*2*pi)^' + str(le2 + 1)
    #                     rt = rt + 'sqrt((x' + str(lx1) + str(0) + '-size(0)/2)^2)*sin(x' + str(lx1) + str(0) + '/size[0]*2*pi)^' + str(le1 + 1) + '*sqrt((x' + str(lx2) + str(1) + '-size(1)/2)^2)*cos(x' + str(lx2) + str(1) + '/size[1]*2*pi)^' + str(le2 + 1)
    #                     rt = rt + 'sqrt((x' + str(lx1) + str(0) + '-size(0)/2)^2)*cos(x' + str(lx1) + str(0) + '/size[0]*2*pi)^' + str(le1 + 1) + '*sqrt((x' + str(lx2) + str(0) + '-size(0)/2)^2)*sin(x' + str(lx2) + str(0) + '/size[0]*2*pi)^' + str(le2 + 1)
    #                     rt = rt + 'sqrt((x' + str(lx1) + str(0) + '-size(0)/2)^2)*cos(x' + str(lx1) + str(0) + '/size[0]*2*pi)^' + str(le1 + 1) + '*sqrt((x' + str(lx2) + str(0) + '-size(0)/2)^2)*cos(x' + str(lx2) + str(0) + '/size[0]*2*pi)^' + str(le2 + 1)
    #                     rt = rt + 'sqrt((x' + str(lx1) + str(0) + '-size(0)/2)^2)*cos(x' + str(lx1) + str(0) + '/size[0]*2*pi)^' + str(le1 + 1) + '*sqrt((x' + str(lx2) + str(1) + '-size(1)/2)^2)*sin(x' + str(lx2) + str(1) + '/size[1]*2*pi)^' + str(le2 + 1)
    #                     rt = rt + 'sqrt((x' + str(lx1) + str(0) + '-size(0)/2)^2)*cos(x' + str(lx1) + str(0) + '/size[0]*2*pi)^' + str(le1 + 1) + '*sqrt((x' + str(lx2) + str(1) + '-size(1)/2)^2)*cos(x' + str(lx2) + str(1) + '/size[1]*2*pi)^' + str(le2 + 1)
    #                     rt = rt + 'sqrt((x' + str(lx1) + str(1) + '-size(1)/2)^2)*sin(x' + str(lx1) + str(1) + '/size[1]*2*pi)^' + str(le1 + 1) + '*sqrt((x' + str(lx2) + str(0) + '-size(0)/2)^2)*sin(x' + str(lx2) + str(0) + '/size[0]*2*pi)^' + str(le2 + 1)
    #                     rt = rt + 'sqrt((x' + str(lx1) + str(1) + '-size(1)/2)^2)*sin(x' + str(lx1) + str(1) + '/size[1]*2*pi)^' + str(le1 + 1) + '*sqrt((x' + str(lx2) + str(0) + '-size(0)/2)^2)*cos(x' + str(lx2) + str(0) + '/size[0]*2*pi)^' + str(le2 + 1)
    #                     rt = rt + 'sqrt((x' + str(lx1) + str(1) + '-size(1)/2)^2)*sin(x' + str(lx1) + str(1) + '/size[1]*2*pi)^' + str(le1 + 1) + '*sqrt((x' + str(lx2) + str(1) + '-size(1)/2)^2)*sin(x' + str(lx2) + str(1) + '/size[1]*2*pi)^' + str(le2 + 1)
    #                     rt = rt + 'sqrt((x' + str(lx1) + str(1) + '-size(1)/2)^2)*sin(x' + str(lx1) + str(1) + '/size[1]*2*pi)^' + str(le1 + 1) + '*sqrt((x' + str(lx2) + str(1) + '-size(1)/2)^2)*cos(x' + str(lx2) + str(1) + '/size[1]*2*pi)^' + str(le2 + 1)
    #                     rt = rt + 'sqrt((x' + str(lx1) + str(1) + '-size(0)/2)^2)*cos(x' + str(lx1) + str(1) + '/size[1]*2*pi)^' + str(le1 + 1) + '*sqrt((x' + str(lx2) + str(0) + '-size(0)/2)^2)*sin(x' + str(lx2) + str(0) + '/size[0]*2*pi)^' + str(le2 + 1)
    #                     rt = rt + 'sqrt((x' + str(lx1) + str(1) + '-size(0)/2)^2)*cos(x' + str(lx1) + str(1) + '/size[1]*2*pi)^' + str(le1 + 1) + '*sqrt((x' + str(lx2) + str(0) + '-size(0)/2)^2)*cos(x' + str(lx2) + str(0) + '/size[0]*2*pi)^' + str(le2 + 1)
    #                     rt = rt + 'sqrt((x' + str(lx1) + str(1) + '-size(0)/2)^2)*cos(x' + str(lx1) + str(1) + '/size[1]*2*pi)^' + str(le1 + 1) + '*sqrt((x' + str(lx2) + str(1) + '-size(1)/2)^2)*sin(x' + str(lx2) + str(1) + '/size[1]*2*pi)^' + str(le2 + 1)
    #                     rt = rt + 'sqrt((x' + str(lx1) + str(1) + '-size(0)/2)^2)*cos(x' + str(lx1) + str(1) + '/size[1]*2*pi)^' + str(le1 + 1) + '*sqrt((x' + str(lx2) + str(1) + '-size(1)/2)^2)*cos(x' + str(lx2) + str(1) + '/size[1]*2*pi)^' + str(le2 + 1)
    i=0
    for lx1 in range(lenx):
        for lx2 in range(lenx):
            for lv1 in range(leval):
                for lv2 in range(leval):
                    if lv1+lv2+2 <= leval:
                        for pt1 in [0,1]:
                            for sc1 in ['s','c']:
                                for pt2 in [0,1]:
                                    for sc2 in ['s','c']:
                                        if args:
                                            rt = rt + str(args[i]) + rouscFunDes(lx1,lv1,pt1,sc1, '') + '*' + rouscFunDes(lx2, lv2, pt2, sc2, '') + ' + '
                                            i+=1
                                        else:
                                            rt = rt + rouscFunDes(lx1, lv1, pt1, sc1, '') + '*' + rouscFunDes(lx2, lv2, pt2, sc2, '') + ' + '
    return rt
def mngDataCalc(x, leval=3, size=(1920,960)):
    '''
    :param x: [(1,2),(3,4),...]
    :param leval:
    :param size:
    :return: [...]
    '''
    lenx = len(x)
    # tmp = []
    # if lenx == 1:
    #     for le in range(leval):
    #         tmp.append(rousinxs(x[0], size[0],le+1))
    #         tmp.append(roucosxs(x[0], size[0],le+1))
    #         tmp.append(rousinxs(x[1], size[1],le+1))
    #         tmp.append(roucosxs(x[1], size[1],le+1))
    #     for le in range(leval-1):
    #         tmp.append(rousinxs(x[0], size[0], le + 1) * rousinxs(x[1], size[1], leval - le - 1))
    #         tmp.append(rousinxs(x[0], size[0], le + 1) * roucosxs(x[1], size[1], leval - le - 1))
    #         tmp.append(roucosxs(x[0], size[0], le + 1) * roucosxs(x[1], size[1], leval - le - 1))
    #         tmp.append(roucosxs(x[0], size[0], le + 1) * roucosxs(x[1], size[1], leval - le - 1))
    # else:
    #     for lx in range(lenx):
    #         for le in range(leval):
    #             tmp.append(rousinxs(x[lx][0],size[0],le+1))
    #             tmp.append(roucosxs(x[lx][0],size[0],le+1))
    #             tmp.append(rousinxs(x[lx][1],size[1],le+1))
    #             tmp.append(roucosxs(x[lx][1],size[1],le+1))
    #         for le in range(leval-1):
    #             tmp.append(rousinxs(x[lx][0], size[0], le + 1) * rousinxs(x[lx][1], size[1], leval - le - 1))
    #             tmp.append(rousinxs(x[lx][0], size[0], le + 1) * roucosxs(x[lx][1], size[1], leval - le - 1))
    #             tmp.append(roucosxs(x[lx][0], size[0], le + 1) * roucosxs(x[lx][1], size[1], leval - le - 1))
    #             tmp.append(roucosxs(x[lx][0], size[0], le + 1) * roucosxs(x[lx][1], size[1], leval - le - 1))
    #     for lx1 in range(lenx):
    #         for lx2 in range(lenx):
    #             if lx1 != lx2:
    #                 for le1 in range(leval-1):
    #                     for le2 in range(leval-1):
    #                         if le1+le2+2 <= leval:
    #                             tmp.append(rousinxs(x[lx1][0], size[0], le1 + 1) * rousinxs(x[lx2][0], size[0], le2 + 1))
    #                             tmp.append(rousinxs(x[lx1][0], size[0], le1 + 1) * roucosxs(x[lx2][0], size[0], le2 + 1))
    #                             tmp.append(rousinxs(x[lx1][0], size[0], le1 + 1) * rousinxs(x[lx2][1], size[1], le2 + 1))
    #                             tmp.append(rousinxs(x[lx1][0], size[0], le1 + 1) * roucosxs(x[lx2][1], size[1], le2 + 1))
    #                             tmp.append(roucosxs(x[lx1][0], size[0], le1 + 1) * rousinxs(x[lx2][0], size[0], le2 + 1))
    #                             tmp.append(roucosxs(x[lx1][0], size[0], le1 + 1) * roucosxs(x[lx2][0], size[0], le2 + 1))
    #                             tmp.append(roucosxs(x[lx1][0], size[0], le1 + 1) * rousinxs(x[lx2][1], size[1], le2 + 1))
    #                             tmp.append(roucosxs(x[lx1][0], size[0], le1 + 1) * roucosxs(x[lx2][1], size[1], le2 + 1))
    #                             tmp.append(rousinxs(x[lx1][1], size[1], le1 + 1) * rousinxs(x[lx2][0], size[0], le2 + 1))
    #                             tmp.append(rousinxs(x[lx1][1], size[1], le1 + 1) * roucosxs(x[lx2][0], size[0], le2 + 1))
    #                             tmp.append(rousinxs(x[lx1][1], size[1], le1 + 1) * rousinxs(x[lx2][1], size[1], le2 + 1))
    #                             tmp.append(rousinxs(x[lx1][1], size[1], le1 + 1) * roucosxs(x[lx2][1], size[1], le2 + 1))
    #                             tmp.append(roucosxs(x[lx1][1], size[1], le1 + 1) * rousinxs(x[lx2][0], size[0], le2 + 1))
    #                             tmp.append(roucosxs(x[lx1][1], size[1], le1 + 1) * roucosxs(x[lx2][0], size[0], le2 + 1))
    #                             tmp.append(roucosxs(x[lx1][1], size[1], le1 + 1) * rousinxs(x[lx2][1], size[1], le2 + 1))
    #                             tmp.append(roucosxs(x[lx1][1], size[1], le1 + 1) * roucosxs(x[lx2][1], size[1], le2 + 1))
    tmp = []
    # for lx1 in range(lenx):
    #     for lx2 in range(lenx):
    #         for le1 in range(leval):
    #             for le2 in range(leval):
    #                 if le1+le2+2 <= leval:
    #                     tmp.append(rousinxs(x[lx1][0], size[0], le1 + 1) * rousinxs(x[lx2][0], size[0], le2 + 1))
    #                     tmp.append(rousinxs(x[lx1][0], size[0], le1 + 1) * roucosxs(x[lx2][0], size[0], le2 + 1))
    #                     tmp.append(rousinxs(x[lx1][0], size[0], le1 + 1) * rousinxs(x[lx2][1], size[1], le2 + 1))
    #                     tmp.append(rousinxs(x[lx1][0], size[0], le1 + 1) * roucosxs(x[lx2][1], size[1], le2 + 1))
    #                     tmp.append(roucosxs(x[lx1][0], size[0], le1 + 1) * rousinxs(x[lx2][0], size[0], le2 + 1))
    #                     tmp.append(roucosxs(x[lx1][0], size[0], le1 + 1) * roucosxs(x[lx2][0], size[0], le2 + 1))
    #                     tmp.append(roucosxs(x[lx1][0], size[0], le1 + 1) * rousinxs(x[lx2][1], size[1], le2 + 1))
    #                     tmp.append(roucosxs(x[lx1][0], size[0], le1 + 1) * roucosxs(x[lx2][1], size[1], le2 + 1))
    #                     tmp.append(rousinxs(x[lx1][1], size[1], le1 + 1) * rousinxs(x[lx2][0], size[0], le2 + 1))
    #                     tmp.append(rousinxs(x[lx1][1], size[1], le1 + 1) * roucosxs(x[lx2][0], size[0], le2 + 1))
    #                     tmp.append(rousinxs(x[lx1][1], size[1], le1 + 1) * rousinxs(x[lx2][1], size[1], le2 + 1))
    #                     tmp.append(rousinxs(x[lx1][1], size[1], le1 + 1) * roucosxs(x[lx2][1], size[1], le2 + 1))
    #                     tmp.append(roucosxs(x[lx1][1], size[1], le1 + 1) * rousinxs(x[lx2][0], size[0], le2 + 1))
    #                     tmp.append(roucosxs(x[lx1][1], size[1], le1 + 1) * roucosxs(x[lx2][0], size[0], le2 + 1))
    #                     tmp.append(roucosxs(x[lx1][1], size[1], le1 + 1) * rousinxs(x[lx2][1], size[1], le2 + 1))
    #                     tmp.append(roucosxs(x[lx1][1], size[1], le1 + 1) * roucosxs(x[lx2][1], size[1], le2 + 1))
    for lx1 in range(lenx):
        for lx2 in range(lenx):
            for lv1 in range(leval):
                for lv2 in range(leval):
                    if lv1+lv2+2 <= leval:
                        for pt1 in [0,1]:
                            for sc1 in ['s','c']:
                                for pt2 in [0,1]:
                                    for sc2 in ['s','c']:
                                        tmp.append(rouscCalc(x[lx1][pt1],size[pt1],sc1,lv1+1)*rouscCalc(x[lx2][pt2],size[pt2],sc2,lv2+1))
    return tmp
def mngData(postotal,leval=3,size=(1920,960)):
    tmp = {}
    for pic in postotal:
        tmp[pic] = (mngDataCalc(postotal[pic],leval,size))
    return tmp
def getRou(length,humanHeigh=1.5, f=0.2, alpha=1):
    '''
    :param ps: int, float
    :param humanHeigh:
    :param f:
    :param alpha:
    :return:
    '''
    if length:
        return humanHeigh*f/alpha*length
    return None
def getTheta(x,width=1920):
    from numpy import pi
    if x:
        return x/width*2*pi
    return None
def getAbsPosTxTy(rou,theta,origin=(100,100,1.5)):
    from numpy import sin,cos
    posx = origin[0] + rou * sin(theta)
    posy = origin[1] + rou * cos(theta)
    return [posx,posy]
def getPosTz(y,heigh=960):
    hz = y/heigh
    return hz
def getAbsPos(pos,humanHeigh=1.5, f=0.2,alpha=1,size=(1920,960),origin=(100,100,1.5)):
    '''
    :param pos: {pic:((1.2,3.2),(8.3,9.6))}
    :param humanHeigh:
    :param f:
    :param alpha:
    :param size:
    :param origin:
    :return:
    '''
    from numpy import sqrt
    rtrn = {}
    for ps in pos:
        rou = getRou(sqrt((pos[ps][1][0]-pos[ps][1][0])**2+(pos[ps][0][1]-pos[ps][1][1])**2),humanHeigh=humanHeigh,f=f,alpha=alpha)
        theta = getTheta(pos[ps][0][0],width=size[0])
        loc = getAbsPosTxTy(rou,theta,origin=origin)
        loc.append(getPosTz(pos[ps][0][1],heigh=size[1]))
        rtrn[ps] = loc
    return rtrn
def getRltPos(ps1,ps2,ps1_loc,referHeigh=1.5, f=0.2, pixel=1,size=(1920,960)):
    '''
    :param ps1: [(1,2),(3,4)]
    :param referHeigh:
    :param f:
    :param pixelM:
    :param size:
    :return: ps2_loc
    '''
    from numpy import sqrt,sin,cos
    rou1 = getRou(sqrt((ps1[0][0]-ps1[1][0])**2+(ps1[0][1]-ps1[1][1])**2),humanHeigh=referHeigh,f=f,alpha=pixel)
    rou2 = getRou(sqrt((ps2[0][0]-ps2[1][0])**2+(ps2[0][1]-ps2[1][1])**2),humanHeigh=referHeigh,f=f,alpha=pixel)
    theta1 = getTheta(ps1[0][0],width=size[0])
    theta2 = getTheta(ps2[0][0],width=size[0])
    diffx = rou1 * sin(theta1) - rou2 * sin(theta2)
    diffy = rou1 * cos(theta1) - rou2 * cos(theta2)
    locx = ps1_loc[0] + diffx
    locy = ps1_loc[1] + diffy
    locz = rou2/rou1*ps1_loc[2] - rou2/f*(getPosTz(ps1[0][1],heigh=size[1]) - getPosTz(ps2[0][1],heigh=size[1]))
    return (locx,locy,locz)
def dict2csv(csvpath, dictdata, fmt='%.4f'):
    from csv import writer
    with open(csvpath, 'w', newline='') as csvF:
        csvW = writer(csvF, delimiter=',')
        for pda in dictdata:
            tmp = []
            tmp.append(pda)
            for pa in dictdata[pda]:
                if len(pa) ==1:
                    tmp.append(fmt % pa)
                else:
                    for ipa in pa:
                        tmp.append(fmt%ipa)
            csvW.writerow(tmp)
def imgpltshow(pos,imgpath):
    import matplotlib.pyplot as plt
    from PIL import Image
    for im in pos:
        fig = plt.figure()
        plt.imshow(Image.open(imgpath[im]))
        for ixy in range(len(pos[im])):
            plt.scatter(pos[im][ixy][0],pos[im][ixy][1])
        plt.pause(0.5)
        if input('press any key to continue...,or input q to quit.') == 'q': break
        plt.close()
    return None
def trasfercoor(locdata):
    return None
def coorTrasferMatrix(posAbs,traincoor):
    from numpy import array
    trainMat = [[] for i in range(3)]
    posMat = []
    picnameMat = []
    for ipa in posAbs:
        picnameMat.append(ipa)
        for il in range(3):
            trainMat[il].append(traincoor[ipa][il])
        posMat.append(posAbs[ipa])
    trainMat = array(trainMat).T
    posMat = array(posMat)
    picnameMat =array(picnameMat)
    return (posMat,trainMat,picnameMat)
def getLinarCoefMat(posMat,trainMat):
    from numpy.linalg import inv
    sym = posMat.T.dot(posMat)
    invsym = inv(sym)
    xyzCoef = invsym.dot(posMat.T.dot(trainMat))
    return xyzCoef
def getTwoCoef(posMat,trainMat):
    return None
def chooseDir(reg,path):
    folderPaths = readfolderfile(reg, path, isFile=False)
    foldername = [fp for fp in folderPaths]
    num = 0
    for fp in foldername:
        print(num, ' : ', fp)
        num+=1
    cnum = [c for c in range(num)]
    key = int(input('input a number in ' + str(cnum)))
    if key not in cnum:return None
    return folderPaths[foldername[key]]
def undistort(img,KD=None):
    if KD is None:
        K = np.array([[194.938, 0, 319.337], [0, 194.393, 240.487], [0, 0, 1]])
        D = np.array([-0.0312506, -0.00373898, -0.000650069, 0.000174612])
    else:
        K = KD[0]
        D = KD[1]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, img.shape[:2], cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img
if __name__ == '__main__':
    dataPath = '~/DataSet/SmartCity'
    pathreg = 'scene\d_\w+'
    folderPath = chooseDir(pathreg,dataPath)
    # folderPath = '~/DataSet/SmartCity/scene1' + '' + '_jiading_lib_training'
    sceneName = folderPath.split('/SmartCity/')[1]
    pklPath = '../data/' + sceneName + '.pkl'
    csvWritePath = '../data/result_' + sceneName + '.csv'
    csvReadPath = folderPath + os.path.sep + sceneName + '_coordinates.csv'
    filereg = 'thumbnail.jpg'
    imgpath = readfolderfile(filereg, folderPath)
    imgname = [th for th in imgpath]
    # img0, img1 = imgread(imgpath, [0, 1])
    pos = pklopen(pklPath)
    pos = remark(pos, imgpath, num=2, delay=12)
    pkldump(pos, pklPath)
    # traincoor = csvread(csvReadPath, f=1)
    # posAbs = getAbsPos(pos)
    # posMat, trainMat, picnameMat = coorTrasferMatrix(posAbs, traincoor)
    # LocationTrain = mnglct(postotal,traincoor)
    # train = mngData(postotal,leval=2,size=(1920,960))
    # fundes = mngFunDes(lenx=3,leval=2)

###########-----------test-------------###################