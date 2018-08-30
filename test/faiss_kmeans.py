# -*- coding: utf-8 -*-
# @Time    : 2018/8/24 上午10:23
# @Author  : Shql
# @Site    : 
# @File    : faiss_kmeans.py
# @Software: PyCharm

import faiss
import numpy as np
from faiss import swigfaiss
from scipy.cluster.vq import *

d = 64
nb = 100000
nq = 10000
np.random.seed(1234)
xb = np.random.random((nb, d)).astype('float32')
# print xb
xb[:, 0] += np.arange(nb) / 1000
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000
# print xq

# swigfaiss.kmeans_clustering(d, nb, 100, float(d * nb), float(d * 100))

# 100个族，维度为d，迭代次数niter=10，
kmeans = faiss.Kmeans(d, 100, niter=10, verbose=True)

result = kmeans.train(xb)
print kmeans.centroids
print kmeans.assign(xq)
# print result
word, lenth = vq(xq, kmeans.centroids)
print word

# ## 向量集构建IndexFlatL2索引，它是最简单的索引类型，只执行强力L2距离搜索
# index = faiss.IndexFlatL2(d)  # build the index
# print(index.is_trained)
# index.add(xb)  # add vectors to the index
# print(index.ntotal)
#
# ## 进行简单的k-近邻搜索
# k = 4  # we want to see 4 nearest neighbors
# D, I = index.search(xb[:5], k)  # sanity check
# print(I)
# print(D)
# D, I = index.search(xq, k)  # actual search
# print(I[:5])  # neighbors of the 5 first queries
# print(I[-5:])  # neighbors of the 5 last queries
