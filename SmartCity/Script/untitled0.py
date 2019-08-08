# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 18:02:46 2019

@author: hszzjs

@e-mail: hushaozhe@stu.xidian.edu.cn
"""
import csv,os
import random

def file_name(file_dir):
    res=[]
    i=0
    for root, dirs, files in os.walk(file_dir):
        
         #当前路径下所有子目录  
        if(len(dirs) and i):
            res+=dirs
            print(dirs)
        i+=1
    return res
    
# 打开一个csv文件对象


#file_dir='./data/'
#tmp=file_name(file_dir)
data = open('test.txt','r')#打开文件
buf = []#缓存文件中数据的变量
for lines in data:
    value = lines.split('\n')[0]#读出每行
    buf.append(value)#转换为数字list并缓存
data.close()
i=0
with open('test1.csv','w', newline='') as csvfile:

    # 创建一个写入对象

    spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    for name in buf:
        i+=1
        a = random.uniform(70, 100)
        a=round(a, 4)
        b = random.uniform(90, 120)
        b=round(b, 4)
        c = random.uniform(1.3, 1.7)
        c=round(c, 4)
        if(i==374):
            break
        #向csv文件里写入第一行
        spamwriter.writerow([name]+[str('%.4f'%a)]+[str('%.4f'%b)]+[str('%.4f'%c)])
#    spamwriter.writerow(['spam']*5 + ['Baked Beans'])
  
    # 向csv文件里写入第二行
#    spamwriter.writerow(['spam', 'Lovely Spam', 'Wonderful Spam'])