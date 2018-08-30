# -*- coding: utf-8 -*-
# @Time    : 2018/7/31 下午5:24
# @Author  : Shql
# @Site    : 
# @File    : sift-test.py
# @Software: PyCharm
import cv2
import os
import numpy as np
import time

sift = cv2.xfeatures2d.SIFT_create()

start = time.clock()
img = cv2.imread('/Users/shi/Downloads/VOCdevkit/VOC2012/JPEGImages/2010_003345.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kp, des = sift.detectAndCompute(gray, None)
end = time.clock()
print end - start
print len(kp)
# cv2.imshow('test', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
