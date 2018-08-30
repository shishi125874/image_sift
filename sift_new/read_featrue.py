# -*- coding: utf-8 -*-
# @Time    : 2018/8/6 下午2:18
# @Author  : Shql
# @Site    : 
# @File    : read_featrue.py
# @Software: PyCharm
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# path = '/Users/shi/work/code/image/1528341681367.jpg'
# sift = cv2.xfeatures2d.SIFT_create()
# img = cv2.imread(path)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# kp, des = sift.detectAndCompute(gray, None)
# des = np.array(des)
# print des[0]
#
# # os.path.basename(path).splitext()
# img_name = os.path.splitext(os.path.basename(path))[0]
# output = os.path.join('/Users/shi/work/code/image', img_name)
# fp = open(output, 'w')
# np.savez(output, img=des)

# r = np.load(output + '.npz')

X = np.random.randint(25, 50, (25, 3))
Y = np.random.randint(60, 85, (25, 3))
Z = np.vstack((X, Y))

# convert to np.float32
Z = np.float32(Z)

# define criteria and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret, label, center = cv2.kmeans(Z, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Now separate the data, Note the flatten()
A = Z[label.ravel() == 0]
B = Z[label.ravel() == 1]
print label
print Z[label]

# Plot the data
plt.scatter(A[:, 0], A[:, 1])
plt.scatter(B[:, 0], B[:, 1], c='r')
plt.scatter(center[:, 0], center[:, 1], s=80, c='y', marker='s')
plt.xlabel('Height'), plt.ylabel('Weight')
plt.show()

# r = np.load('/Users/shi/work/data/tain/fearture/2008_001460.npz')
#
# result = r['img']
#
# print result.shape
# print result
