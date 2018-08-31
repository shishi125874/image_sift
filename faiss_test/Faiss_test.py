# -*- coding: utf-8 -*-
# @Time    : 2018/7/18 下午6:58
# @Author  : Shql
# @Site    : 
# @File    : Faiss_test.py
# @Software: PyCharm
import faiss
import numpy as np

d = 64
nb = 100000
nq = 10000
np.random.seed(1234)
xb = np.random.random((nb, d)).astype('float32')
# print xb
xb[:, 0] += np.arange(nb) / 1000
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000

print xq.shape

## 向量集构建IndexFlatL2索引，它是最简单的索引类型，只执行强力L2距离搜索
index = faiss.IndexFlatL2(d)  # build the index
print(index.is_trained)
index.add(xb)  # add vectors to the index
print(index.ntotal)

k = 4                          # we want to see 4 nearest neighbors
D, I = index.search(xq, k)     # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(D[-5:])

## 进行简单的k-近邻搜索
k = 4  # we want to see 4 nearest neighbors
D, I = index.search(xb[:5], k)  # sanity check
# print(I)
# print(D)
D, I = index.search(xq, k)  # actual search
# print(I[:5])  # neighbors of the 5 first queries
# print(I[-5:])  # neighbors of the 5 last queries
