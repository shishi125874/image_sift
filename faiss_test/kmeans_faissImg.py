# -*- coding: utf-8 -*-
# @Time    : 2018/7/30 下午3:05
# @Author  : Shql
# @Site    : 
# @File    : kmeans_faissImg.py
# @Software: PyCharm
import faiss
import numpy as np

# f = open(''siftEm.npz'')

r = np.load('/Users/shi/work/code/image/siftEm.npz')

# sift = r['labels']
sift = r['img']

# np.load('siftEm.npz')

print len(sift)
