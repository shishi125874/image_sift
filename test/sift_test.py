# -*- coding: utf-8 -*-
# @Time    : 2018/8/29 上午10:31
# @Author  : Shql
# @Site    : 
# @File    : sift_test.py
# @Software: PyCharm
import cv2
import numpy as np

sift = cv2.xfeatures2d.SIFT_create()
count = 1
img = cv2.imread('/Users/shi/work/code/image/image/2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kp, des = sift.detectAndCompute(gray, None)
des = np.array(des)
print des.shape

img1 = cv2.imread('/Users/shi/work/code/image/image/1.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
kp1, des1 = sift.detectAndCompute(gray, None)
des2 = np.array(des1)
print des2.shape

result = np.concatenate((des, des2), axis=0)
# result = np.append(img, img1, axis=1)
print result.shape
