# -*- coding: utf-8 -*-
# @Time    : 2018/8/22 下午2:53
# @Author  : Shql
# @Site    : 
# @File    : search.py
# @Software: PyCharm

import argparse as ap
import cv2
import numpy as np
import os
from sklearn.externals import joblib
from scipy.cluster.vq import *

from sklearn import preprocessing
import numpy as np

from pylab import *
from PIL import Image

# Get query image path
# image_path = args["image"]
image_path = '/Users/shi/Downloads/VOCdevkit/VOC2012/JPEGImages/2007_002445.jpg'

# Load the classifier, class names, scaler, number of clusters and vocabulary
im_features, image_paths, idf, numWords, voc = joblib.load("bof6000.pkl")

# Create feature extraction and keypoint detector objects
sift = cv2.xfeatures2d.SIFT_create()
# fea_det = cv2.FeatureDetector_create("SIFT")
# des_ext = cv2.DescriptorExtractor_create("SIFT")

# List where all the descriptors are stored
des_list = []

im = cv2.imread(image_path)
grays = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# kpts = fea_det.detect(im)
# kpts, des = des_ext.compute(im, kpts)
kpts, des = sift.detectAndCompute(grays, None)

# rootsift
# rs = RootSIFT()
# des = rs.compute(kpts, des)

des_list.append((image_path, des))

# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]

#
test_features = np.zeros((1, numWords), "float32")
words, distance = vq(descriptors, voc)
for w in words:
    test_features[0][w] += 1

# Perform Tf-Idf vectorization and L2 normalization
test_features = test_features * idf
test_features = preprocessing.normalize(test_features, norm='l2')

score = np.dot(test_features, im_features.T)
rank_ID = np.argsort(-score)

# Visualize the results
# figure()
# gray()
# subplot(5, 4, 1)
# imshow(im[:, :, ::-1])
# axis('off')
cv2.imshow('yuan', cv2.imread(image_path))
# print image_paths
for i, ID in enumerate(rank_ID[0][0:5]):
    img_nps = image_paths[ID]
    img_name = os.path.splitext(os.path.split(img_nps)[1])[0] + '.jpg'
    img_path = os.path.join('/Users/shi/Downloads/VOCdevkit/VOC2012/JPEGImages', img_name)
    img = cv2.imread(img_path)
    cv2.imshow(str(i), img)
    # img = Image.open(img_path)
    # gray()
    # subplot(5, 4, i + 5)
    # imshow(img)
    # axis('off')
cv2.waitKey(0)
cv2.destroyAllWindows()

# show()
