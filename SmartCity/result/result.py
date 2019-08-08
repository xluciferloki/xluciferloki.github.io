# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 18:02:46 2019

@author: hszzjs

@e-mail: hushaozhe@stu.xidian.edu.cn
"""
import os,sys
import csv
import random
import re
import pickle
# def file_name(file_dir):
#     res=[]
#     i=0
#     for root, dirs, files in os.walk(file_dir):
#         #当前路径下所有子目录
#         if(len(dirs) and i):
#             res+=dirs
#         i+=1
#     return res
    
# 打开一个csv文件对象

file_dir='../../DataSet/SmartCity/'
Dirs=[];PIC=[];Coors=[];TS=[]
for root, dirs, files in os.walk(file_dir):
    #当前路径下全景图像文件
    for f in files:
        if re.match('thumbnail.jpg',f):
            PIC.append(root+'/'+f)
        if re.match('TotalStation.jpg',f):
            TS.append(root+'/'+f)
        if re.match('\w+coordinates.csv',f):
            Coors.append(root+'/'+f)
    #当前路径下图片目录名称
    for d in dirs:
        if re.match('PIC\w+',d):
            Dirs.append(d)
with open('../data/PIC_PATH.pkl', 'wb') as f:
    pickle.dump(PIC,f)
with open('../data/TS_PATH.pkl', 'wb') as f:
    pickle.dump(TS,f)
with open('../data/Coors_PATH.pkl', 'wb') as f:
    pickle.dump(Coors,f)
# data = open('test.txt','r')#打开文件
# buf = []#缓存文件中数据的变量
# for lines in data:
#     value = lines.split('\n')[0]#读出每行
#     buf.append(value)#转换为数字list并缓存
# data.close()
# i=0
detect=[]
increase=[]
for dir1 in Dirs:
    if re.findall(dir1, str(PIC)).__len__() is 0:
        detect.append(dir1)
    if re.findall(dir1,str(Dirs)).__len__() is not 1:
        increase.append(dir1)

with open('../data/result.csv','w', newline='') as csvfile:
    # 创建一个写入对象
    resultwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    for name in Dirs:
        x_coor = random.uniform(0, 200)
        y_coor = random.uniform(0, 200)
        z_coor = random.uniform(0, 3)

        resultwriter.writerow([name,round(x_coor, 4),round(y_coor, 4),round(z_coor, 4)])

if __name__ == __main__:
    a=1