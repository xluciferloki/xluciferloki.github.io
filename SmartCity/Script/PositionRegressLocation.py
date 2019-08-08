import os
import sys
import cv2
import glob
import csv
import numpy as np
import pickle
import imutils

from PIL import Image
import pylab
import tkinter.filedialog

from sklearn.model_selection import train_test_split

def save(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

path = '/home/loki/PycharmProjects/test1/code'
os.chdir(path)

picture = load('picNames')

ref_pic_name = picture[0]


mpd = load('databaseRef')

# tra_pic = picture[0:int(picture.__len__()*2/3)]
# pre_pic = picture[(int(picture.__len__()*2/3)+1):]

# picNames = list(mpd.keys())
# save(picNames,"picNames")

# databaseRef = {}
# for pic in picture:
#     databaseRef[pic] = mpd[pic][ref_pic_name]
#
# save(databaseRef,'databaseRef')

location = load('location')

x=[]
y=[]
# pic = picture[0]
for pic in picture:
    temp = []
    for i in range(1):
        temp.append(mpd[pic][0][i, 0])
        temp.append(mpd[pic][0][i, 1])
        t0 = (mpd[pic][1][i, 0] - mpd[pic][0][i, 0])
        t1 = (mpd[pic][1][i, 1] - mpd[pic][0][i, 1])
        temp.append(t0)
        temp.append(t1)
    x.append(temp)
    y.append(location[pic])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

import numpy as np
from sklearn import datasets,linear_model

# 定义训练数据
X_train = np.array(x_train)
Y_train = np.array(y_train)

regr = linear_model.LinearRegression()
Y_test = [[] for i in range(3)]

regr.fit(X_train,Y_train[:,0])
Y_test[0] = regr.predict(x_test)

regr.fit(X_train,Y_train[:,1])
Y_test[1] = regr.predict(x_test)

regr.fit(X_train,Y_train[:,2])
Y_test[2] = regr.predict(x_test)

# print('coefficients(b1,b2...):',regr.coef_)
# print('intercept(b0):',regr.intercept_)

Y = np.array(y_test)

from matplotlib import pylab as pl

y_test = np.array(y_test)
Y_test = np.array(Y_test)
Test_mean = [[] for i in range(3)]
Test_var = [[] for i in range(3)]
for i in range(3):
    Test_mean[i] = np.mean(y_test[:,i]-Y_test[i,:])
    Test_var[i] = np.var(y_test[:,i]-Y_test[i,:])


def test():
    a=1

if __name__=="__main__":
    import numpy
    test()
